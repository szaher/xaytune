"""EmbeddedControllerHost: the controller, running inside the caller's process.

```text
submit(spec)
   │ validate                           nothing is written for a spec that
   │                                    cannot run
   ├── Experiment, Node, Run            each with its event
   ├── compile                          the compiler the spec names
   ├── Attempt + INTENDED operation     one commit (ADR-005 §4)
   ├── runtime.submit_or_get()          the only external effect
   ├── operation CONFIRMED + RuntimeRef
   └── observe                          telemetry → durable transitions
```

**Intent before effect, always.** The attempt and its submit operation commit
before the runtime is asked for anything, so a crash between the two leaves
intent with no effect -- which a later reconciler can resolve -- and never an
effect with no intent, which nothing could find.

**Durable-state-backed, not restart-reconciling.** Everything this host knows
is in the record, so a handle from ``attach()`` in another process reads the
same answers. But an attempt that was in flight when the submitting process
died is not adopted by a new host: nothing here resumes its telemetry or
settles its outcome. That is PR-012a, and until then ``wait()`` on such an
experiment says so rather than waiting forever.

The controller loop only turns observations into transitions. It does not
decide scientific outcomes: a successful run leaves the node ``ACTIVE``,
because the next thing that can happen to it is evaluation, and the experiment
``ACTIVE``, because ending an experiment is a decision (implementation plan,
PR-012).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from xaytune.compilation import (
    CompilationContext,
    TrainerCompiler,
    UnsupportedCandidateError,
)
from xaytune.core.domain.event import DomainEvent
from xaytune.core.domain.experiment import (
    CandidateSpecSnapshot,
    Experiment,
    ExperimentNode,
)
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.domain.run import Run, RunAttempt
from xaytune.core.domain.specs import CompilerSpec, RuntimeSpec
from xaytune.core.errors import XaytuneError
from xaytune.core.execution import ResolvedExecutionPlan
from xaytune.core.ids import (
    ExperimentId,
    ExperimentNodeId,
    RunAttemptId,
    RunId,
)
from xaytune.core.refs import Actor, ArtifactRef, ControllerHostRef, RuntimeRef
from xaytune.core.sqlite import connect
from xaytune.core.state.machines import ATTEMPT_MACHINE, RUN_MACHINE
from xaytune.core.state.status import (
    ExperimentNodeStatus,
    ExperimentStatus,
    RunAttemptStatus,
    RunStatus,
)
from xaytune.core.telemetry import (
    ArtifactProducedPayload,
    TrainingStartedPayload,
    WorkerReadyPayload,
)
from xaytune.experiment.handle import (
    ExperimentHandle,
    ExperimentResult,
    NextStage,
    NodeOutcome,
    RunOutcome,
)
from xaytune.experiment.spec import ExperimentSpec
from xaytune.runtimes import RuntimeBackend, RuntimeStatus
from xaytune.storage.control_plane import ControlPlaneRepository
from xaytune.storage.migrations import migrate

__all__ = [
    "ControllerNotRunningError",
    "EmbeddedControllerHost",
    "UnknownImplementationError",
]

_ACTOR = Actor(type="system", id="embedded-controller")

_NODE_ACTIVATION = (
    ExperimentNodeStatus.PLANNED,
    ExperimentNodeStatus.READY,
    ExperimentNodeStatus.ACTIVE,
)
"""A submitted candidate is planned, ready and active at once: there is no
planner deciding whether to run it, and nothing to wait for before it can.
Each step is still its own transition with its own event, so the history does
not claim the node skipped states the machine requires."""

_ATTEMPT_PATH = (
    RunAttemptStatus.QUEUED,
    RunAttemptStatus.STARTING,
    RunAttemptStatus.RUNNING,
)
"""The non-terminal states an attempt passes through, in order."""

_RUNTIME_OUTCOME: Mapping[str, tuple[RunAttemptStatus, RunStatus]] = {
    "succeeded": (RunAttemptStatus.SUCCEEDED, RunStatus.SUCCEEDED),
    "failed": (RunAttemptStatus.FAILED, RunStatus.FAILED),
    "cancelled": (RunAttemptStatus.CANCELLED, RunStatus.CANCELLED),
}


class UnknownImplementationError(XaytuneError):
    """A spec names a compiler or runtime this host cannot resolve."""


class ControllerNotRunningError(XaytuneError):
    """Work is unsettled and nothing in this process is driving it.

    The submitting process has gone, or never was this one. Adopting an
    in-flight attempt -- finding its workload, resuming its telemetry, settling
    its outcome -- is PR-012a's reconciliation, and waiting here instead would
    wait forever.
    """


def _default_compilers() -> dict[str, Callable[[], TrainerCompiler]]:
    from xaytune.compilation.native import NativeCompiler
    from xaytune.compilation.trl import TRLCompiler

    return {"native": NativeCompiler, "trl": TRLCompiler}


def _local_runtime(config: Mapping[str, Any]) -> RuntimeBackend:
    from xaytune.runtimes.local import LocalRuntime

    unexpected = sorted(set(config) - {"root"})
    if unexpected:
        raise ValueError(f"the local runtime takes no configuration {unexpected}")
    root = config.get("root")
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise ValueError(
            f"the local runtime needs config.root as an absolute path, not {root!r}: "
            f"it is where the runtime's durable registry lives"
        )
    return LocalRuntime(root)  # type: ignore[return-value]


class EmbeddedControllerHost:
    """Runs the controller for experiments inside this process.

    Args:
        state_path: The control-plane database. Created and migrated if new.
        compilers: Compiler factories by name. Defaults to the built-in
            ``native`` and ``trl``; the registry the durable
            :class:`CompilerSpec` resolves through.
        runtimes: Runtime factories by kind, each taking the spec's config.
            Defaults to ``local``.
    """

    def __init__(
        self,
        state_path: Path | str,
        *,
        compilers: Mapping[str, Callable[[], TrainerCompiler]] | None = None,
        runtimes: Mapping[str, Callable[[Mapping[str, Any]], RuntimeBackend]] | None = None,
    ) -> None:
        if str(state_path) != ":memory:":
            # A new state database is a normal first run, not an error: SQLite
            # creates the file but not the directory it lives in.
            Path(state_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = connect(state_path)
        migrate(self._connection)
        self.repository = ControlPlaneRepository(self._connection)
        self._compilers = dict(compilers or _default_compilers())
        self._runtime_factories = dict(runtimes or {"local": _local_runtime})
        self._runtimes: dict[str, RuntimeBackend] = {}
        self._controllers: dict[str, asyncio.Task[None]] = {}
        self._reference = ControllerHostRef(kind="embedded", id=f"embedded-{uuid.uuid4().hex}")

    # ---- the public surface ---------------------------------------------

    async def submit(self, spec: ExperimentSpec) -> ExperimentHandle:
        """Record the experiment, start its first run, and return its handle.

        Returns once the runtime has accepted the workload and that is
        recorded; training continues under a controller task in this process.

        Raises:
            UnknownImplementationError: If the spec names a compiler or runtime
                this host cannot resolve.
            UnsupportedCandidateError: If the compiler cannot run the candidate
                exactly as declared. Nothing is recorded in either case: a spec
                that cannot run is refused at submission, not after.
        """
        compiler = self._compiler(spec.compiler.name)
        support = compiler.supports(spec.candidate)
        if not support:
            raise UnsupportedCandidateError(compiler.descriptor.name, support.reasons)
        runtime = self._runtime(spec.runtime)

        experiment = self._record_experiment(spec, compiler, runtime)
        node = self._record_node(experiment, spec)
        run = self._record_run(node, spec.seed)

        attempt = RunAttempt(id=RunAttemptId.generate(), run_id=run.id, attempt_number=1)
        plan = ResolvedExecutionPlan(
            spec=compiler.compile(
                spec.candidate,
                CompilationContext(
                    run_id=str(run.id),
                    seed=spec.seed,
                    output_uri=str(Path(spec.artifact_root) / str(run.id)),
                ),
            ),
            runtime=spec.runtime.kind,
            target=RuntimeOperationTarget(kind="training-attempt", id=str(attempt.id)),
        )

        attempt, operation = self.repository.create_attempt_with_submit_intent(
            attempt, request_digest=plan.request_digest("submit"), actor=_ACTOR
        )

        try:
            reference = await runtime.submit_or_get(operation.id, plan)
        except Exception as exc:
            if _is_refusal(exc):
                # A known negative: the runtime refused the plan and nothing
                # runs. Anything else is an unknown outcome, and stays INTENDED
                # for reconciliation rather than being written down as failed.
                self.repository.fail_operation(
                    operation.id, expected_revision=operation.revision, actor=_ACTOR
                )
                self._settle(attempt.id, run.id, RunAttemptStatus.FAILED, RunStatus.FAILED)
                return ExperimentHandle(experiment.id, self)
            raise

        self.repository.confirm_operation(
            operation.id,
            expected_revision=operation.revision,
            actor=_ACTOR,
            runtime_ref=reference,
        )
        self._advance_attempt(attempt.id, RunAttemptStatus.QUEUED)

        self._controllers[str(experiment.id)] = asyncio.create_task(
            self._observe(experiment.id, attempt.id, run.id, runtime, reference),
            name=f"xaytune-controller-{experiment.id}",
        )
        return ExperimentHandle(experiment.id, self)

    async def attach(self, experiment_id: ExperimentId | str) -> ExperimentHandle:
        """A handle to an experiment already in the record.

        Raises:
            AggregateNotFoundError: If no such experiment exists.
        """
        experiment = self.repository.aggregates.load_experiment(str(experiment_id))
        return ExperimentHandle(experiment.id, self)

    async def close(self) -> None:
        """Stop observing, and release the database and runtimes.

        A controller task still running is cancelled: its workload keeps
        running, because the runtime owns it, and the record keeps the intent
        and the reference. Picking it up again is PR-012a.
        """
        for task in self._controllers.values():
            task.cancel()
        await asyncio.gather(*self._controllers.values(), return_exceptions=True)
        self._controllers.clear()
        for runtime in self._runtimes.values():
            close = getattr(runtime, "close", None)
            if close is not None:
                close()
        self._runtimes.clear()
        self._connection.close()

    # ---- what the handle asks -------------------------------------------

    def _experiment_status(self, experiment_id: ExperimentId) -> ExperimentStatus:
        return self.repository.aggregates.load_experiment(str(experiment_id)).status

    def _events_after(self, experiment_id: ExperimentId, sequence: int) -> tuple[DomainEvent, ...]:
        return self.repository.events.events_for_experiment_after(str(experiment_id), sequence)

    async def _wait(self, experiment_id: ExperimentId) -> ExperimentResult:
        controller = self._controllers.get(str(experiment_id))
        if controller is not None:
            # Shielded: a caller that stops waiting must not stop the
            # controller, which other handles may be waiting on too.
            await asyncio.shield(controller)
        result = self._result(experiment_id)
        if not result.quiescent:
            raise ControllerNotRunningError(
                f"experiment {experiment_id} has unsettled work and no controller in this "
                f"process is driving it; adopting in-flight work after a restart is PR-012a"
            )
        return result

    def _result(self, experiment_id: ExperimentId) -> ExperimentResult:
        """Where the experiment stands, read entirely from the record."""
        aggregates = self.repository.aggregates
        experiment = aggregates.load_experiment(str(experiment_id))

        nodes: list[NodeOutcome] = []
        settled = True
        trained = False
        for node in aggregates.nodes_for_experiment(str(experiment_id)):
            runs: list[RunOutcome] = []
            for run in aggregates.runs_for_node(str(node.id)):
                attempts = aggregates.attempts_for_run(str(run.id))
                final = max(attempts, key=lambda a: a.attempt_number) if attempts else None
                settled = settled and RUN_MACHINE.is_terminal(run.status)
                trained = trained or (
                    run.status is RunStatus.SUCCEEDED and node.status is ExperimentNodeStatus.ACTIVE
                )
                runs.append(
                    RunOutcome(
                        run_id=run.id,
                        status=run.status,
                        attempt_status=final.status if final else None,
                        artifacts=final.artifact_refs if final else (),
                    )
                )
            nodes.append(NodeOutcome(node_id=node.id, status=node.status, runs=tuple(runs)))

        next_stage: NextStage | None
        if experiment.is_terminal:
            next_stage = None
        elif trained:
            next_stage = "evaluation"
        else:
            next_stage = "failure-handling"

        return ExperimentResult(
            experiment_id=experiment.id,
            status=experiment.status,
            quiescent=settled,
            next_stage=next_stage,
            nodes=tuple(nodes),
        )

    # ---- the controller loop --------------------------------------------

    async def _observe(
        self,
        experiment_id: ExperimentId,
        attempt_id: RunAttemptId,
        run_id: RunId,
        runtime: RuntimeBackend,
        reference: RuntimeRef,
    ) -> None:
        """Turn one attempt's telemetry into durable transitions, then settle it.

        Reads the stream until the runtime ends it, then asks the runtime how
        the workload ended. The stream says what happened along the way; the
        status says how it finished -- a worker can exit non-zero after
        reporting ``TrainingCompleted`` (publication failed), and the outcome
        is the exit, not the last observation.
        """
        async for envelope in runtime.watch(reference):
            observation = envelope.payload.data
            if isinstance(observation, WorkerReadyPayload):
                self._advance_attempt(attempt_id, RunAttemptStatus.STARTING)
            elif isinstance(observation, TrainingStartedPayload):
                self._advance_attempt(attempt_id, RunAttemptStatus.RUNNING)
            elif isinstance(observation, ArtifactProducedPayload):
                self._record_artifact(attempt_id, observation.artifact_ref)

        status = await runtime.get_status(reference)
        outcome = _RUNTIME_OUTCOME.get(status.state)
        if outcome is None:
            # The workload is live or unknown although its stream ended: the
            # supervisor is gone. Deciding what that means -- adopt, advance
            # the telemetry generation, escalate -- is PR-012a, so the attempt
            # is left unsettled and wait() says so.
            return
        self._settle(attempt_id, run_id, *outcome, status=status)
        self._reconcile_cancellations(experiment_id)

    # ---- cancellation (ADR-013 §6) -----------------------------------------

    async def _cancel(self, experiment_id: ExperimentId, *, reason: str) -> None:
        """Record the cancellation saga, then carry out its effects.

        Intent first, in one commit: the experiment's Action and a child
        Action and cancel operation per live attempt. Then each effect,
        confirmed once the runtime has accepted it. Nothing here moves the
        experiment: it reaches ``CANCELLED`` when reconciliation sees that no
        attempt is live, which for a running workload is after the controller
        has observed it stop.

        An effect that cannot be issued -- no recorded reference, a runtime
        that raised -- is left ``INTENDED``, and the experiment stays
        ``ACTIVE``. Never timed out into ``CANCELLED``: that would claim a
        workload stopped exactly when it is most likely still running.
        """
        experiment = self.repository.aggregates.load_experiment(str(experiment_id))
        parent, children = self.repository.request_experiment_cancellation(
            experiment.id, reason=reason, actor=_ACTOR
        )
        if experiment.runtime is not None:
            runtime = self._runtime(experiment.runtime)
            for child, operation in children:
                if operation is None or operation.state != "intended":
                    continue
                reference = self._submitted_reference(child.target.id)
                if reference is None:
                    continue
                await runtime.cancel(reference, operation.id)
                self.repository.confirm_operation(
                    operation.id, expected_revision=operation.revision, actor=_ACTOR
                )
        self.repository.reconcile_experiment_cancellation(parent.id, actor=_ACTOR)

    def _reconcile_cancellations(self, experiment_id: ExperimentId) -> None:
        """Settle any experiment cancellation in flight, now an attempt has ended."""
        for action in self.repository.actions.for_target("experiment", str(experiment_id)):
            if action.type == "cancel-experiment" and not action.is_terminal:
                self.repository.reconcile_experiment_cancellation(action.id, actor=_ACTOR)

    def _submitted_reference(self, attempt_id: str) -> RuntimeRef | None:
        """The runtime reference the attempt's confirmed submission recorded."""
        for operation in self.repository.operations.for_target("training-attempt", attempt_id):
            if operation.type == "submit" and operation.runtime_ref is not None:
                return operation.runtime_ref
        return None

    # ---- durable writes --------------------------------------------------

    def _record_experiment(
        self, spec: ExperimentSpec, compiler: TrainerCompiler, runtime: RuntimeBackend
    ) -> Experiment:
        experiment = Experiment(
            id=ExperimentId.generate(),
            name=spec.name,
            objective=spec.objective,
            controller_host=self._reference,
            # ADR-016: the version is resolved here and recorded, so the
            # record says which implementation ran, not only which name.
            compiler=CompilerSpec(
                name=spec.compiler.name, version=compiler.descriptor.plugin_version
            ),
            runtime=spec.runtime.model_copy(
                update={"version": runtime.descriptor.plugin_version}  # type: ignore[attr-defined]
            ),
            artifact_root=spec.artifact_root,
        )
        self.repository.create_experiment(experiment, actor=_ACTOR)
        return self.repository.transition_experiment(
            experiment.id,
            expected_revision=experiment.revision,
            new_status=ExperimentStatus.ACTIVE,
            actor=_ACTOR,
        )

    def _record_node(self, experiment: Experiment, spec: ExperimentSpec) -> ExperimentNode:
        node = ExperimentNode(
            id=ExperimentNodeId.generate(),
            experiment_id=experiment.id,
            hypothesis=spec.hypothesis,
            reason="submitted",
            candidate=CandidateSpecSnapshot(candidate=spec.candidate),
            candidate_fingerprint=spec.candidate.candidate_fingerprint(),
            created_by=_ACTOR,
        )
        self.repository.create_node(node, actor=_ACTOR)
        for status in _NODE_ACTIVATION:
            node = self.repository.transition_node(
                node.id, expected_revision=node.revision, new_status=status, actor=_ACTOR
            )
        return node

    def _record_run(self, node: ExperimentNode, seed: int) -> Run:
        run = Run(
            id=RunId.generate(),
            node_id=node.id,
            experiment_id=node.experiment_id,
            seed=seed,
            replicate=1,
            candidate_fingerprint=node.candidate_fingerprint,
        )
        self.repository.create_run(run, actor=_ACTOR)
        return self.repository.transition_run(
            run.id, expected_revision=run.revision, new_status=RunStatus.ACTIVE, actor=_ACTOR
        )

    def _advance_attempt(self, attempt_id: RunAttemptId, target: RunAttemptStatus) -> RunAttempt:
        """Move an attempt forward to *target*, through each state in between.

        Only ever forward: an observation that arrives late -- a
        ``WorkerReady`` read after ``TrainingStarted`` was already applied --
        is a no-op, not a regression.
        """
        attempt = self.repository.aggregates.load_attempt(str(attempt_id))
        if attempt.is_terminal:
            return attempt
        path = _ATTEMPT_PATH
        position = path.index(attempt.status) if attempt.status in path else -1
        for status in path[position + 1 : path.index(target) + 1]:
            attempt = self.repository.transition_attempt(
                attempt.id, expected_revision=attempt.revision, new_status=status, actor=_ACTOR
            )
        return attempt

    def _record_artifact(self, attempt_id: RunAttemptId, artifact: ArtifactRef) -> None:
        attempt = self.repository.aggregates.load_attempt(str(attempt_id))
        if any(existing.id == artifact.id for existing in attempt.artifact_refs):
            return
        self.repository.record_artifact(
            attempt.id, artifact, expected_revision=attempt.revision, actor=_ACTOR
        )

    def _settle(
        self,
        attempt_id: RunAttemptId,
        run_id: RunId,
        attempt_status: RunAttemptStatus,
        run_status: RunStatus,
        *,
        status: RuntimeStatus | None = None,
    ) -> None:
        """Record how an attempt, and so its run, ended.

        A successful attempt passes through ``RUNNING`` if telemetry never
        reported the start: it cannot have succeeded without running, and the
        machine requires the state. A failed or cancelled one goes straight to
        its outcome from wherever it was.
        """
        attempt = self.repository.aggregates.load_attempt(str(attempt_id))
        if not attempt.is_terminal:
            if attempt_status is RunAttemptStatus.SUCCEEDED:
                attempt = self._advance_attempt(attempt_id, RunAttemptStatus.RUNNING)
            if ATTEMPT_MACHINE.can(attempt.status, attempt_status):
                self.repository.transition_attempt(
                    attempt.id,
                    expected_revision=attempt.revision,
                    new_status=attempt_status,
                    actor=_ACTOR,
                )
        run = self.repository.aggregates.load_run(str(run_id))
        if not RUN_MACHINE.is_terminal(run.status):
            self.repository.transition_run(
                run.id, expected_revision=run.revision, new_status=run_status, actor=_ACTOR
            )

    # ---- resolving specs -------------------------------------------------

    def _compiler(self, name: str) -> TrainerCompiler:
        factory = self._compilers.get(name)
        if factory is None:
            raise UnknownImplementationError(
                f"no compiler named {name!r}; this host knows {sorted(self._compilers)}"
            )
        return factory()

    def _runtime(self, spec: RuntimeSpec) -> RuntimeBackend:
        factory = self._runtime_factories.get(spec.kind)
        if factory is None:
            raise UnknownImplementationError(
                f"no runtime of kind {spec.kind!r}; this host knows "
                f"{sorted(self._runtime_factories)}"
            )
        key = spec.model_dump_json(exclude={"version"})
        if key not in self._runtimes:
            self._runtimes[key] = factory(spec.config)
        return self._runtimes[key]


def _is_refusal(exc: BaseException) -> bool:
    """Whether *exc* is a runtime's definitive refusal of a plan."""
    from xaytune.runtimes.local import UnsupportedPlanError

    return isinstance(exc, UnsupportedPlanError)
