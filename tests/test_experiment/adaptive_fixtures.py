"""Deterministic fixtures for the adaptive MVP (spec 18, PR-026).

No model is trained and no LLM is involved: what is under test is the control
plane carrying one experiment through training, recovery, evaluation,
decision, planning, branching and a second candidate's training.

```text
LoRACompiler      the native compiler, for a LoRA candidate up to max_rank:
                  the plan records the adapter rank, and the candidate's own
                  fingerprint, so the run realizes exactly the node it names
AdaptiveRuntime   scripted workloads, idempotent by operation id:
                  training at oom_rank, first attempt   checkpoint, CUDA OOM, exit 1
                  any other training attempt            restores a checkpoint if
                                                        directed, publishes a model
                  evaluation                            task_success by the model's rank
```
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.evaluation_fixtures import evaluation
from tests.test_checkpoints.helpers import make_bundle
from tests.test_experiment.test_host_behaviour import _spec
from xaytune.compilation import SupportResult
from xaytune.compilation.attempt_resolution import training_execution_fingerprint
from xaytune.compilation.native import NativeCompiler
from xaytune.core.domain.candidate import AdapterSpec, CandidateSpec
from xaytune.core.domain.evaluation import MetricResult
from xaytune.core.domain.objective import BudgetSpec, Objective, ObjectiveMetric
from xaytune.core.domain.specs import PlannerSpec
from xaytune.core.ids import ArtifactId
from xaytune.core.immutable import FrozenDict, thaw
from xaytune.core.refs import ArtifactRef, CheckpointRef, RuntimeRef
from xaytune.core.telemetry import (
    ArtifactProducedPayload,
    EvaluationCompletedPayload,
    IncidentObservedPayload,
)
from xaytune.runtimes import (
    EvaluationEventPayload,
    OperationOutcome,
    RuntimeEventEnvelope,
    RuntimeStatus,
    TrainingEventPayload,
)
from xaytune.runtimes.local.runtime import LocalRuntime

GROW_2 = FrozenDict({"rules": [{"kind": "increase-lora-rank", "factor": 2, "max_rank": 64}]})
SEED = 42


def _without_adapter(candidate: CandidateSpec) -> CandidateSpec:
    training = candidate.training.model_copy(update={"adapter": None})
    return candidate.model_copy(update={"training": training})


class LoRACompiler(NativeCompiler):
    """The native compiler, accepting a LoRA adapter up to *max_rank*."""

    def __init__(self, max_rank: int = 64) -> None:
        self.max_rank = max_rank

    def supports(self, candidate: CandidateSpec) -> SupportResult:
        adapter = candidate.training.adapter
        if adapter is not None and (adapter.rank is None or adapter.rank > self.max_rank):
            return SupportResult(
                supported=False,
                reasons=(f"LoRA rank {adapter.rank} is above this compiler's {self.max_rank}",),
            )
        return super().supports(_without_adapter(candidate))

    def compile(self, candidate: CandidateSpec, context: Any) -> Any:
        spec = super().compile(_without_adapter(candidate), context)
        adapter = candidate.training.adapter
        return spec.model_copy(
            update={
                "candidate_fingerprint": candidate.candidate_fingerprint(),
                "config": FrozenDict(
                    {**thaw(spec.config), "adapter_rank": None if adapter is None else adapter.rank}
                ),
            }
        )


class AdaptiveRuntime:
    """Scripted training and evaluation, keyed by operation id like a real runtime."""

    descriptor = LocalRuntime.descriptor

    def __init__(
        self,
        manager: Any,
        tmp_path: Path,
        *,
        values: dict[int, float] | None = None,
        oom_rank: int | None = 16,
    ) -> None:
        self.manager = manager
        self.tmp_path = tmp_path
        self.values = values or {16: 0.79, 32: 0.83}
        self.oom_rank = oom_rank
        self.submitted: dict[str, RuntimeRef] = {}
        self.training_plans: list[Any] = []
        self.evaluation_plans: list[Any] = []
        self.events: dict[str, tuple[RuntimeEventEnvelope, ...]] = {}
        self.outcomes: dict[str, RuntimeStatus] = {}
        self.models: dict[str, int] = {}
        self.restore_context: Any = None
        self.restored: list[CheckpointRef] = []

    def capabilities(self) -> Any:
        base = LocalRuntime.capabilities(self)  # type: ignore[arg-type]
        assert base.checkpoint is not None
        return base.model_copy(
            update={
                "extensions": {},
                "checkpoint": base.checkpoint.model_copy(
                    update={"atomic_commit": True, "full_exact_restore": True, "formats": ()}
                ),
            }
        )

    async def lookup_operation(self, operation_id: Any) -> OperationOutcome | None:
        reference = self.submitted.get(str(operation_id))
        if reference is None:
            return None
        return OperationOutcome(
            operation_id=operation_id, disposition="accepted", runtime_ref=reference
        )

    async def submit_or_get(self, operation_id: Any, plan: Any) -> RuntimeRef:
        if str(operation_id) in self.submitted:
            return self.submitted[str(operation_id)]
        target = plan.target
        reference = RuntimeRef(backend="local", external_id=target.id)
        self.submitted[str(operation_id)] = reference
        if target.kind == "training-attempt":
            await self._train(plan, target)
        else:
            self._evaluate(plan, target)
        return reference

    async def _train(self, plan: Any, target: Any) -> None:
        self.training_plans.append(plan)
        rank = plan.spec.config["adapter_rank"]
        if rank == self.oom_rank:
            self.oom_rank = None
            state, context, restore = make_bundle(
                self.tmp_path / f"capture-{target.id}",
                attempt_id=target.id,
                candidate=plan.spec.candidate_fingerprint,
                execution=training_execution_fingerprint(plan),
            )
            self.restore_context = restore
            checkpoint = await self.manager.save(state, context)
            manifest = (await self.manager.store.get(checkpoint)).manifest
            self.events[target.id] = (
                self._training(target, 0, manifest.committed_payload(checkpoint)),
                self._training(target, 1, IncidentObservedPayload(reason="cuda-oom")),
            )
            self.outcomes[target.id] = RuntimeStatus(state="failed", exit_code=1)
            return
        restore = plan.runtime_options.get("checkpoint_restore")
        if restore is not None:
            checkpoint_ref = CheckpointRef.model_validate(restore)
            await self.manager.restore(checkpoint_ref, self.restore_context)
            self.restored.append(checkpoint_ref)
        uri = str(self.tmp_path / "models" / target.id)
        self.models[uri] = rank
        model = ArtifactRef(id=ArtifactId.generate(), kind="model", uri=uri)
        self.events[target.id] = (
            self._training(target, 0, ArtifactProducedPayload(artifact_ref=model)),
        )
        self.outcomes[target.id] = RuntimeStatus(state="succeeded", exit_code=0)

    def _evaluate(self, plan: Any, target: Any) -> None:
        self.evaluation_plans.append(plan)
        config = plan.spec.config
        rank = self.models[config["subject_uri"]]
        metric = MetricResult(
            name="task_success",
            value=self.values[rank],
            sample_count=4,
            evaluator_name="scripted",
            evaluator_version=config["evaluator_version"],
            seed=config["seed"],
        )
        self.events[target.id] = (
            RuntimeEventEnvelope(
                event_id=f"{target.id}-completed",
                sequence=0,
                protocol_version="xaytune.telemetry/v1alpha3",
                target=target,
                payload=EvaluationEventPayload(data=EvaluationCompletedPayload(metrics=(metric,))),
            ),
        )
        self.outcomes[target.id] = RuntimeStatus(state="succeeded", exit_code=0)

    @staticmethod
    def _training(target: Any, sequence: int, data: Any) -> RuntimeEventEnvelope:
        return RuntimeEventEnvelope(
            event_id=f"{target.id}-{sequence}",
            sequence=sequence,
            target=target,
            payload=TrainingEventPayload(data=data),
        )

    async def watch(self, reference: RuntimeRef, cursor: Any) -> Any:
        for event in self.events[reference.external_id]:
            yield event

    async def get_status(self, reference: RuntimeRef) -> RuntimeStatus:
        return self.outcomes[reference.external_id]

    def close(self) -> None:
        pass


def adaptive_spec(
    tmp_path: Path,
    *,
    budget: BudgetSpec | None = None,
    planner_config: FrozenDict = GROW_2,
) -> Any:
    """Spec 18, small: LoRA 16, micro-batch 4 × accumulation 8, task_success >= 0.82."""
    base = _spec(
        tmp_path,
        objective=Objective(
            primary=ObjectiveMetric(name="task_success", direction="maximize"), target=0.82
        ),
        evaluation=evaluation(),
        planner=PlannerSpec(kind="rule-based", config=planner_config),
        budget=budget or BudgetSpec(max_runs=4),
        seed=SEED,
    )
    training = base.candidate.training.model_copy(
        update={
            "adapter": AdapterSpec(type="lora", rank=16, alpha=32.0),
            "optimization": base.candidate.training.optimization.model_copy(
                update={"micro_batch_size": 4, "gradient_accumulation": 8}
            ),
        }
    )
    return base.model_copy(
        update={"candidate": base.candidate.model_copy(update={"training": training})}
    )
