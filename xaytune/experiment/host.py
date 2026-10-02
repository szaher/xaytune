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

**Restart-safe, not yet owned.** Everything this host knows is in the record,
so when the process that submitted an experiment dies, ``attach()`` from a new
host adopts its unsettled work (PR-012a): a live workload is observed again
from the durable telemetry cursor, an unconfirmed submission is looked up, and
a submission is issued only when the runtime says it never received it. What
it does not have is ownership. Two hosts attached to one experiment *at the
same time* would both adopt it -- no second workload, since adoption never
issues one, but two observers whose writes would conflict. Leases that make
one host the owner belong to the daemon host, not to this one.

The controller loop turns observations into transitions, and leaves
scientific outcomes to the decision engine. A successful run leaves the node
``ACTIVE`` when no evaluation is configured, because the next thing that can
happen to it is evaluation, and the experiment ``ACTIVE``, because ending an
experiment is a decision (implementation plan, PR-012).

**Evaluation, when the spec asks for it** (ADR-015, PR-013):

```text
training run SUCCEEDED
   ├── node ACTIVE → EVALUATING, cycle n, EvaluationRun   one commit
   ├── EvaluationAttempt + INTENDED operation              one commit
   ├── runtime.submit_or_get()                             the same journal
   ├── observe: EvaluationCompleted(metrics) held durably,
   │   with the cursor advanced to it                      one commit
   ├── the runtime says the workload succeeded:
   │   result + attempt + run SUCCEEDED                    one commit
   └── node EVALUATING → DECIDING                          reconciled, not assumed
```

**Then the decision** (PR-015): the context is assembled from the record --
the objective and the cycle's results -- and the engine's decision is
recorded with what it causes, node ``COMPLETED`` / ``REJECTED`` and the
experiment ``SUCCEEDED`` / ``FAILED``, in one commit. An undecidable cycle
stays ``DECIDING``, with a ``DecisionDeferred`` event. An evaluation in
flight is adopted after a restart exactly as training is; a node left in
``DECIDING`` by a crash is decided when the experiment is attached.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from xaytune.checkpoints import CheckpointManager
from xaytune.compilation import (
    CompilationContext,
    TrainerCompiler,
    UnsupportedCandidateError,
)
from xaytune.compilation.attempt_resolution import (
    resolve_training_attempt,
    training_execution_fingerprint,
)
from xaytune.core.domain.action import ActionStatus
from xaytune.core.domain.actions import ActionSpec
from xaytune.core.domain.budget import (
    BudgetExhaustedError,
    CapacityUnavailableError,
    UnsupportedBudgetError,
    budget_refusals,
)
from xaytune.core.domain.decision import DecisionContext
from xaytune.core.domain.evaluation import (
    EvaluationAttempt,
    EvaluationResult,
    EvaluationRun,
    EvaluationSpec,
    EvaluatorSpec,
    result_provenance_problems,
)
from xaytune.core.domain.event import DomainEvent
from xaytune.core.domain.experiment import (
    CandidateSpecSnapshot,
    Experiment,
    ExperimentNode,
)
from xaytune.core.domain.incident import Incident, IncidentCategory
from xaytune.core.domain.intervention import InterventionDirective, TrainingPosition
from xaytune.core.domain.intervention_replay import InterventionReplayError
from xaytune.core.domain.numerical_recovery import (
    NONFINITE_CATEGORIES,
    NumericalEscalation,
    UnsupportedNumericalRecoveryError,
)
from xaytune.core.domain.oom_recovery import (
    OOMEscalation,
    OOMRecoveryInputsV1,
    OOMResizeProposal,
    PriorOOMResize,
)
from xaytune.core.domain.operation import RuntimeOperation, RuntimeOperationTarget
from xaytune.core.domain.planning import settled_for_planning
from xaytune.core.domain.policy import GovernedAction
from xaytune.core.domain.recovery import RecoveryRequest
from xaytune.core.domain.run import Run, RunAttempt
from xaytune.core.domain.specs import CompilerSpec, PlannerSpec, RuntimeSpec
from xaytune.core.errors import XaytuneError
from xaytune.core.execution import PythonModuleEntrypoint, ResolvedExecutionPlan
from xaytune.core.execution_controls import MANAGED_NUMERICAL_RECOVERY, TRAINING_INTERVENTIONS
from xaytune.core.ids import (
    ActionId,
    EvaluationAttemptId,
    EvaluationId,
    EvaluationRunId,
    ExperimentId,
    ExperimentNodeId,
    RunAttemptId,
    RunId,
)
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import Actor, ArtifactRef, ControllerHostRef, RuntimeRef
from xaytune.core.sqlite import connect
from xaytune.core.state.machines import (
    ATTEMPT_MACHINE,
    EVALUATION_ATTEMPT_MACHINE,
    RUN_MACHINE,
)
from xaytune.core.state.status import (
    EvaluationAttemptStatus,
    EvaluationRunStatus,
    ExperimentNodeStatus,
    ExperimentStatus,
    RunAttemptStatus,
    RunStatus,
)
from xaytune.core.telemetry import (
    ArtifactProducedPayload,
    CheckpointCommittedPayload,
    EvaluationCompletedPayload,
    EvaluationStartedPayload,
    InterventionAppliedPayload,
    TrainingStartedPayload,
    WorkerReadyPayload,
)
from xaytune.decision import DecisionEngine, ThresholdDecisionEngine, UndecidableError
from xaytune.evaluation import (
    EvaluationContext,
    Evaluator,
    ResolvableEvaluator,
    UnsupportedEvaluationError,
)
from xaytune.experiment.handle import (
    EvaluationOutcome,
    ExperimentHandle,
    ExperimentResult,
    NextStage,
    NodeOutcome,
    RunOutcome,
)
from xaytune.experiment.spec import ExperimentSpec
from xaytune.planning import PLANNERS, Planner
from xaytune.policy import DenyAllPolicy, PolicyEngine
from xaytune.resilience import IncidentClassifier
from xaytune.resilience.numerical import NumericalRecoveryPlanner
from xaytune.resilience.numerical_execution import (
    NumericalCheckpointUnavailableError,
    NumericalRecoveryExecutor,
)
from xaytune.resilience.oom import OOMRecoveryPlanner
from xaytune.resilience.oom_execution import (
    OOMCheckpointUnavailableError,
    OOMRecoveryExecutor,
)
from xaytune.resilience.recovery import RecoveryCoordinator
from xaytune.runtimes import RuntimeBackend, RuntimeEventEnvelope, RuntimeStatus, StreamCursor
from xaytune.storage.control_plane import (
    ControlPlaneRepository,
    EvaluationReconciliation,
    ProvenanceError,
    StalePolicyContextError,
    StaleRecoveryContextError,
)
from xaytune.storage.journal import IdempotencyConflictError
from xaytune.storage.migrations import migrate

__all__ = [
    "ControllerNotRunningError",
    "EmbeddedControllerHost",
    "ImplementationMismatchError",
    "ReconciliationEscalatedError",
    "UnknownImplementationError",
]

_ACTOR = Actor(type="system", id="embedded-controller")
_INCIDENT_CLASSIFIER = IncidentClassifier()

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

_LIVE_STATES = frozenset({"pending", "queued", "starting", "running"})
_OUTCOME_POLL_SECONDS = 0.5
# How often a training attempt waiting for a parallel-run slot looks again.
_CAPACITY_POLL_SECONDS = 0.5

# A candidate whose decision settled what it is: rejected on a constraint, or
# completed short of the target. A COMPLETED node under an ACTIVE experiment
# can only be a BRANCH -- STOP_SUCCEEDED ends the experiment with it.
_SCIENTIFICALLY_SETTLED = frozenset({ExperimentNodeStatus.REJECTED, ExperimentNodeStatus.COMPLETED})

_RUNTIME_OUTCOME: Mapping[str, tuple[RunAttemptStatus, RunStatus]] = {
    "succeeded": (RunAttemptStatus.SUCCEEDED, RunStatus.SUCCEEDED),
    "failed": (RunAttemptStatus.FAILED, RunStatus.FAILED),
    "cancelled": (RunAttemptStatus.CANCELLED, RunStatus.CANCELLED),
}

_EVALUATION_PATH = (
    EvaluationAttemptStatus.QUEUED,
    EvaluationAttemptStatus.STARTING,
    EvaluationAttemptStatus.RUNNING,
)

_EVALUATION_OUTCOME: Mapping[str, tuple[EvaluationAttemptStatus, EvaluationRunStatus]] = {
    "failed": (EvaluationAttemptStatus.FAILED, EvaluationRunStatus.FAILED),
    "cancelled": (EvaluationAttemptStatus.CANCELLED, EvaluationRunStatus.CANCELLED),
    # Preempted: the attempt says so, and the run fails -- there is no retry
    # policy yet, and a run left ACTIVE over a preempted attempt would have
    # nothing executing it. A retry, when there is one, is a new attempt; it
    # never recovers this one.
    "preempted": (EvaluationAttemptStatus.PREEMPTED, EvaluationRunStatus.FAILED),
}
"""How an evaluation that did not succeed ends. Success is not here: it is
recorded only with its result, through ``record_evaluation_result``."""


class UnknownImplementationError(XaytuneError):
    """A spec names a compiler or runtime this host cannot resolve."""


class ImplementationMismatchError(XaytuneError):
    """The record names a compiler or runtime version this host cannot provide.

    Raised only where that implementation would be used, not merely because
    the record names it:

    - a **runtime** mismatch refuses interacting with the recorded external
      effect -- looking it up, adopting it, cancelling it -- because a
      different runtime version may track workloads differently;
    - a **compiler** mismatch refuses rebuilding a request to re-issue it,
      because a different compiler version may compile the same candidate
      into a different request.

    Nothing is adopted or issued when it is raised. Reading the record needs
    neither: a settled experiment attaches with no runtime or compiler at all.
    """


class ReconciliationEscalatedError(XaytuneError):
    """Reconciliation reached a question it must not answer by guessing.

    The work is left exactly as recorded -- an unresolved operation stays
    unresolved -- and a person, or a later and better-informed controller,
    has to decide.
    """


class ControllerNotRunningError(XaytuneError):
    """Work is unsettled, nothing is driving it, and this host cannot adopt it.

    The record names no runtime to adopt it with -- an experiment recorded
    before a host drove experiments. Waiting instead would wait forever.
    """


def _default_compilers() -> dict[str, Callable[[], TrainerCompiler]]:
    from xaytune.compilation.native import NativeCompiler
    from xaytune.compilation.trl import TRLCompiler

    return {"native": NativeCompiler, "trl": TRLCompiler}


def _default_evaluators() -> dict[str, Callable[[], Evaluator]]:
    from xaytune.evaluation.lmeval import LMEvalEvaluator
    from xaytune.evaluation.native import NativeEvaluator

    return {"native": NativeEvaluator, "lm-eval": LMEvalEvaluator}


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
        evaluators: Evaluator factories by name, the registry a recorded
            ``EvaluatorSpec`` resolves through. Defaults to the built-in
            ``native``.
        decision_engine: What decides an evaluated candidate. Defaults to
            :class:`~xaytune.decision.ThresholdDecisionEngine`, for which a
            missed target ends the experiment;
            :class:`~xaytune.decision.AdaptiveThresholdDecisionEngine` reads
            it as ``BRANCH`` and leaves the experiment open for another
            candidate.
        policy: What authorizes a proposed action
            (:meth:`ExperimentHandle.propose`). ``None`` means
            :class:`~xaytune.policy.DenyAllPolicy`: with no policy configured,
            every proposal is denied, with a durable decision saying why.
            Cancellation is never governed by it.
        planners: Planner factories by kind, each binding a ``PlannerSpec``.
            Defaults to :data:`~xaytune.planning.PLANNERS` (``rule-based`` and
            ``no-op``). A spec's planner is bound at submission and recorded;
            nothing invokes it yet.
    """

    def __init__(
        self,
        state_path: Path | str,
        *,
        compilers: Mapping[str, Callable[[], TrainerCompiler]] | None = None,
        runtimes: Mapping[str, Callable[[Mapping[str, Any]], RuntimeBackend]] | None = None,
        evaluators: Mapping[str, Callable[[], Evaluator]] | None = None,
        decision_engine: DecisionEngine | None = None,
        policy: PolicyEngine | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        recovery_request_for_incident: Callable[[Incident], RecoveryRequest | None] | None = None,
        planners: Mapping[str, Callable[[PlannerSpec], Planner]] | None = None,
    ) -> None:
        if str(state_path) != ":memory:":
            # A new state database is a normal first run, not an error: SQLite
            # creates the file but not the directory it lives in.
            Path(state_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = connect(state_path)
        migrate(self._connection)
        self.repository = ControlPlaneRepository(self._connection)
        # ``None`` means the defaults; an empty mapping means none. Treating an
        # empty registry as "use the defaults" would hand a caller who removed
        # every compiler the built-in ones anyway.
        self._compilers = dict(_default_compilers() if compilers is None else compilers)
        self._runtime_factories = dict({"local": _local_runtime} if runtimes is None else runtimes)
        self._evaluators = dict(_default_evaluators() if evaluators is None else evaluators)
        self._planners = dict(PLANNERS if planners is None else planners)
        self._decision_engine = (
            ThresholdDecisionEngine() if decision_engine is None else decision_engine
        )
        self._policy: PolicyEngine = DenyAllPolicy() if policy is None else policy
        self._checkpoint_manager = checkpoint_manager
        self._recovery_request_for_incident = recovery_request_for_incident
        self._runtimes: dict[str, RuntimeBackend] = {}
        # One observer per attempt, keyed by experiment then attempt: an
        # experiment can have several live attempts, and each must stay
        # tracked -- for wait() to wait on and close() to stop.
        self._controllers: dict[str, dict[str, asyncio.Task[None]]] = {}
        self._escalations: dict[str, str] = {}
        self._reference = ControllerHostRef(kind="embedded", id=f"embedded-{uuid.uuid4().hex}")

    # ---- the public surface ---------------------------------------------

    async def submit(self, spec: ExperimentSpec) -> ExperimentHandle:
        """Record the experiment, start its first run, and return its handle.

        Returns once the runtime has accepted the workload and that is
        recorded; training continues under a controller task in this process.

        Raises:
            UnknownImplementationError: If the spec names a compiler, runtime,
                evaluator or planner this host cannot resolve.
            PlannerConfigurationError: If the planner refuses its spec's
                version or configuration.
            UnsupportedCandidateError: If the compiler cannot run the candidate
                exactly as declared.
            UnsupportedEvaluationError: If the evaluator cannot run the
                evaluation exactly as declared.
            UnsupportedBudgetError: If the budget limits something nothing
                measures. Nothing is recorded in any of these cases: a spec
                that cannot run is refused at submission, not after.

        A budget that has nothing left for the first run is not refused: the
        experiment is recorded, and ends ``BUDGET_EXHAUSTED`` without starting
        anything, which the returned handle reports.
        """
        compiler = self._compiler(spec.compiler.name)
        support = compiler.supports(spec.candidate)
        if not support:
            raise UnsupportedCandidateError(compiler.descriptor.name, support.reasons)
        if spec.budget is not None:
            refused = budget_refusals(spec.budget)
            if refused:
                raise UnsupportedBudgetError(refused)
        runtime = self._runtime(spec.runtime)
        if spec.numerical_recovery is not None:
            refused = self._numerical_recovery_refusals(spec, compiler, runtime)
            if refused:
                raise UnsupportedNumericalRecoveryError(refused)
        evaluation = None if spec.evaluation is None else self._bind_evaluation(spec.evaluation)
        planner = None if spec.planner is None else self._planner(spec.planner).spec

        experiment = self._record_experiment(spec, compiler, runtime, evaluation, planner)
        node = self._record_node(experiment, spec)
        try:
            run = self._record_run(node, spec.seed)
            attempt_id = RunAttemptId.generate()
            plan = self._plan(experiment, run, attempt_id, compiler)
            attempt = RunAttempt(
                id=attempt_id,
                run_id=run.id,
                attempt_number=1,
                execution_fingerprint=training_execution_fingerprint(plan),
            )
            attempt, operation = await self._create_attempt(attempt, plan.request_digest("submit"))
        except BudgetExhaustedError as exhausted:
            self.repository.exhaust_budget(experiment.id, reasons=exhausted.reasons, actor=_ACTOR)
            return ExperimentHandle(experiment.id, self)

        await self._issue(experiment.id, run.id, attempt.id, operation, plan, runtime)
        return ExperimentHandle(experiment.id, self)

    def _settled_for_planning(self, node: NodeOutcome) -> bool:
        decisions = self.repository.aggregates.decisions_for_node(str(node.node_id))
        latest = max(decisions, key=lambda d: d.evaluation_cycle) if decisions else None
        return settled_for_planning(node.status, None if latest is None else latest.outcome)

    def _numerical_recovery_refusals(
        self, spec: ExperimentSpec, compiler: TrainerCompiler, runtime: RuntimeBackend
    ) -> tuple[str, ...]:
        """Why numerical recovery cannot be armed for *spec* here, before anything is recorded.

        Arming makes a nonfinite loss fail the attempt, so it is refused unless
        that failure can actually be recovered: a managed-checkpoint Native
        plan, a runtime that declares the worker requests for that worker, and a
        host with a checkpoint manager and an explicit first RecoveryRequest.
        """
        reasons: list[str] = []
        if self._checkpoint_manager is None:
            reasons.append("the host has no checkpoint manager to validate a restore")
        if self._recovery_request_for_incident is None:
            reasons.append("the host has no recovery_request_for_incident for the first decision")
        if spec.candidate.training.optimization.learning_rate is None:
            reasons.append("the candidate declares no learning rate to reduce")
        probe = compiler.compile(
            spec.candidate,
            CompilationContext(
                run_id="numerical-recovery-probe",
                seed=spec.seed,
                output_uri=str(Path(spec.artifact_root) / "numerical-recovery-probe"),
                checkpoint_store_uri=str(Path(spec.artifact_root) / "checkpoints"),
            ),
        )
        entrypoint = probe.entrypoint
        if probe.checkpoint.format != "native-torch/v1" or not isinstance(
            entrypoint, PythonModuleEntrypoint
        ):
            reasons.append("the compiled plan has no managed Native checkpoints")
        else:
            declared = runtime.capabilities().extensions.get("worker_requests", {})
            supported = set(declared.get(entrypoint.module, ()))
            if not {MANAGED_NUMERICAL_RECOVERY, TRAINING_INTERVENTIONS} <= supported:
                reasons.append(
                    f"the runtime does not declare the numerical-recovery worker requests "
                    f"for {entrypoint.module}"
                )
        return tuple(reasons)

    async def attach(self, experiment_id: ExperimentId | str) -> ExperimentHandle:
        """A handle to an experiment already in the record, adopting its work.

        If nothing in this process is driving the experiment -- the process that
        submitted it has gone -- its unsettled work is reconciled first (see
        :meth:`_reconcile`): a live workload is adopted and observed from the
        durable cursor, an unconfirmed submission is looked up, and only a
        submission the runtime never received is issued, under its recorded
        identity.

        Raises:
            AggregateNotFoundError: If no such experiment exists.
            ImplementationMismatchError: If unsettled work needs the recorded
                runtime -- to look up, adopt or cancel an effect -- and this
                host provides a different version; or if a submission must be
                re-issued and this host provides a different compiler version
                to rebuild it with. Nothing is adopted or issued.
            UnknownImplementationError: In the same cases, when this host has
                no such runtime or compiler at all. An experiment with nothing
                left to reconcile needs neither.
        """
        experiment = self.repository.aggregates.load_experiment(str(experiment_id))
        if not self._driving(experiment.id):
            await self._reconcile(experiment)
        return ExperimentHandle(experiment.id, self)

    async def close(self) -> None:
        """Stop observing, and release the database and runtimes.

        A controller task still running is cancelled: its workload keeps
        running, because the runtime owns it, and the record keeps the intent
        and the reference, so a later ``attach()`` adopts it.
        """
        tasks = [task for observers in self._controllers.values() for task in observers.values()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._controllers.clear()
        for runtime in self._runtimes.values():
            close = getattr(runtime, "close", None)
            if close is not None:
                close()
        self._runtimes.clear()
        self._connection.close()

    # ---- what the handle asks -------------------------------------------

    async def approve_action(
        self, action_id: ActionId | str, *, approver: Actor, reason: str
    ) -> GovernedAction:
        """A human approves an action awaiting approval. Nothing is carried out.

        Approves the proposal policy judged, without judging again. Raises what
        :meth:`~xaytune.storage.ControlPlaneRepository.approve_action` raises.
        """
        self.repository.approve_action(action_id, approver=approver, reason=reason)
        numerical = self.repository.numerical_recovery_bindings.for_action(str(action_id))
        if numerical is not None:
            episode = self.repository.recovery_episodes.get(str(numerical.episode_id))
            assert episode is not None
            await self._drive_numerical_recovery(
                episode.context.experiment_id,
                RunId(episode.context.run_id),
                RunAttemptId(episode.context.target.id),
            )
        binding = self.repository.recovery_action_bindings.for_action(str(action_id))
        if binding is not None:
            episode = self.repository.recovery_episodes.get(str(binding.episode_id))
            assert episode is not None
            await self._drive_oom_recovery(
                episode.context.experiment_id,
                RunId(episode.context.run_id),
                RunAttemptId(episode.context.target.id),
            )
        return self.repository.governed_action(action_id)

    async def reject_action(
        self, action_id: ActionId | str, *, approver: Actor, reason: str
    ) -> GovernedAction:
        """A human refuses an action awaiting approval."""
        self.repository.reject_action(action_id, approver=approver, reason=reason)
        numerical = self.repository.numerical_recovery_bindings.for_action(str(action_id))
        if numerical is not None:
            try:
                self.repository.abandon_numerical_recovery_action(
                    ActionId(str(action_id)), actor=approver, reason=reason
                )
            except StaleRecoveryContextError:
                episode = self.repository.recovery_episodes.get(str(numerical.episode_id))
                assert episode is not None
                await self._drive_numerical_recovery(
                    episode.context.experiment_id,
                    RunId(episode.context.run_id),
                    RunAttemptId(episode.context.target.id),
                )
        binding = self.repository.recovery_action_bindings.for_action(str(action_id))
        if binding is not None:
            try:
                self.repository.abandon_oom_recovery_action(
                    ActionId(str(action_id)), actor=approver, reason=reason
                )
            except StaleRecoveryContextError:
                episode = self.repository.recovery_episodes.get(str(binding.episode_id))
                assert episode is not None
                await self._drive_oom_recovery(
                    episode.context.experiment_id,
                    RunId(episode.context.run_id),
                    RunAttemptId(episode.context.target.id),
                )
        return self.repository.governed_action(action_id)

    # ---- governed actions -----------------------------------------------

    _STALE_RETRIES = 3

    def _propose(
        self, experiment_id: ExperimentId, spec: ActionSpec, *, reason: str, proposed_by: Actor
    ) -> GovernedAction:
        """Validate, authorize and record *spec*, against the experiment's runtime.

        The runtime's declared capabilities join the snapshot policy judges;
        an experiment whose record names no runtime declares none. A snapshot
        that changed before it could be recorded is judged again, a few times.
        """
        experiment = self.repository.aggregates.load_experiment(str(experiment_id))
        capabilities = (
            None if experiment.runtime is None else self._runtime(experiment.runtime).capabilities()
        )
        for attempt in range(self._STALE_RETRIES):
            try:
                return self.repository.propose_action(
                    spec,
                    experiment_id=experiment_id,
                    proposed_by=proposed_by,
                    reason=reason,
                    policy=self._policy,
                    capabilities=capabilities,
                )
            except StalePolicyContextError:
                if attempt == self._STALE_RETRIES - 1:
                    raise
        raise AssertionError("unreachable")

    def _actions(self, experiment_id: ExperimentId) -> tuple[GovernedAction, ...]:
        return tuple(
            self.repository.governed_action(action.id)
            for action in self.repository.actions.for_experiment(str(experiment_id))
        )

    def _experiment_status(self, experiment_id: ExperimentId) -> ExperimentStatus:
        return self.repository.aggregates.load_experiment(str(experiment_id)).status

    def _events_after(self, experiment_id: ExperimentId, sequence: int) -> tuple[DomainEvent, ...]:
        return self.repository.events.events_for_experiment_after(str(experiment_id), sequence)

    async def _wait(self, experiment_id: ExperimentId) -> ExperimentResult:
        # Every attempt's observer, including any adopted while waiting.
        # Shielded: a caller that stops waiting must not stop the observers,
        # which other handles may be waiting on too.
        awaited: set[asyncio.Task[None]] = set()
        while True:
            observers = self._controllers.get(str(experiment_id), {}).values()
            pending = [task for task in observers if task not in awaited]
            if not pending:
                break
            awaited.update(pending)
            await asyncio.gather(*(asyncio.shield(task) for task in pending))
        escalation = self._escalations.get(str(experiment_id))
        if escalation is not None:
            raise ReconciliationEscalatedError(escalation)
        result = self._result(experiment_id)
        if not result.quiescent:
            raise ControllerNotRunningError(
                f"experiment {experiment_id} has unsettled work, no controller is driving it, "
                f"and this host cannot adopt it: the record names no runtime spec"
            )
        return result

    def _result(self, experiment_id: ExperimentId) -> ExperimentResult:
        """Where the experiment stands, read entirely from the record."""
        aggregates = self.repository.aggregates
        experiment = aggregates.load_experiment(str(experiment_id))

        nodes: list[NodeOutcome] = []
        settled = True
        trained = deciding = evaluating = False
        awaiting_approval, awaiting_execution = self.repository.resting_actions(str(experiment_id))
        approval_targets = set()
        for action in awaiting_approval:
            binding = self.repository.recovery_action_bindings.for_action(
                str(action.id)
            ) or self.repository.numerical_recovery_bindings.for_action(str(action.id))
            if binding is not None and self.repository.recovery_plans.is_effective_and_fresh(
                str(binding.plan_id)
            ):
                episode = self.repository.recovery_episodes.get(str(binding.episode_id))
                if episode is not None:
                    approval_targets.add(episode.context.target.id)
        for node in aggregates.nodes_for_experiment(str(experiment_id)):
            runs: list[RunOutcome] = []
            for run in aggregates.runs_for_node(str(node.id)):
                attempts = aggregates.attempts_for_run(str(run.id))
                final = max(attempts, key=lambda a: a.attempt_number) if attempts else None
                settled = settled and (
                    RUN_MACHINE.is_terminal(run.status)
                    or (
                        final is not None
                        and final.is_terminal
                        and str(final.id) in approval_targets
                    )
                )
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
            evaluations: list[EvaluationOutcome] = []
            for evaluation_run in aggregates.evaluation_runs_for_node(str(node.id)):
                evaluation_attempts = aggregates.evaluation_attempts_for_run(str(evaluation_run.id))
                settled = settled and evaluation_run.is_terminal
                evaluating = evaluating or not evaluation_run.is_terminal
                evaluations.append(
                    EvaluationOutcome(
                        evaluation_run_id=evaluation_run.id,
                        evaluation_cycle=evaluation_run.evaluation_cycle,
                        status=evaluation_run.status,
                        attempt_status=(
                            evaluation_attempts[-1].status if evaluation_attempts else None
                        ),
                        result=aggregates.evaluation_result_for_run(str(evaluation_run.id)),
                    )
                )
            deciding = deciding or node.status is ExperimentNodeStatus.DECIDING
            nodes.append(
                NodeOutcome(
                    node_id=node.id,
                    status=node.status,
                    runs=tuple(runs),
                    evaluations=tuple(evaluations),
                )
            )

        # Quiescent means no control work is unresolved -- not only that every
        # run has ended. An effect with no known outcome, or an action still in
        # flight, is work somebody has to finish, and a run can be terminal
        # while it remains (a cancellation that raced natural completion).
        operations, actions = self.repository.unsettled_work(str(experiment_id))
        settled = settled and not operations and not actions

        next_stage: NextStage | None
        if experiment.is_terminal:
            next_stage = None
        elif awaiting_approval:
            next_stage = "action-approval"
        elif awaiting_execution:
            next_stage = "action-execution"
        elif deciding:
            next_stage = "decision"
        elif trained or evaluating:
            next_stage = "evaluation"
        elif nodes and all(node.status in _SCIENTIFICALLY_SETTLED for node in nodes):
            if all(self._settled_for_planning(node) for node in nodes):
                # Every candidate was evaluated and decided on its merits --
                # rejected for a violated constraint, or completed short of
                # the target (BRANCH) -- and the experiment is still open:
                # another candidate is what comes next, a planner's work.
                # Only a scientific outcome leads here. A candidate that
                # failed or was cancelled did not establish a result, and is
                # failure-handling.
                next_stage = "planning"
            else:
                # Settled, but not in a way that asks for another candidate: a
                # STOP decision the experiment did not apply -- it was paused
                # when the decision was made, and resuming does not apply it
                # -- or a node settled with no decision. Somebody must decide
                # what the experiment's outcome is.
                next_stage = "decision"
        else:
            # Training failed or was cancelled, or an evaluation ended without
            # a result and the node's cycle stalled.
            next_stage = "failure-handling"

        return ExperimentResult(
            experiment_id=experiment.id,
            status=experiment.status,
            quiescent=settled,
            next_stage=next_stage,
            nodes=tuple(nodes),
            budget=self.repository.budget_status(experiment.id),
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
        generation, sequence = self.repository.aggregates.telemetry_position(str(attempt_id))
        cursor = StreamCursor(generation=generation, sequence=sequence)
        async for envelope in runtime.watch(reference, cursor):
            observation = envelope.payload.data
            position = (envelope.stream_generation, envelope.sequence)
            if isinstance(observation, WorkerReadyPayload):
                self._advance_attempt(attempt_id, RunAttemptStatus.STARTING, position)
            elif isinstance(observation, TrainingStartedPayload):
                self._advance_attempt(attempt_id, RunAttemptStatus.RUNNING, position)
            elif isinstance(observation, ArtifactProducedPayload):
                self._record_artifact(attempt_id, observation.artifact_ref, position)
            elif isinstance(observation, InterventionAppliedPayload):
                self._record_intervention_applied(experiment_id, attempt_id, observation, position)
            elif isinstance(observation, CheckpointCommittedPayload):
                self.repository.record_checkpoint(
                    attempt_id,
                    observation,
                    evidence=FrozenDict(envelope.model_dump(mode="json")),
                    actor=_ACTOR,
                )
            else:
                self._record_incident(
                    RuntimeOperationTarget(kind="training-attempt", id=str(attempt_id)), envelope
                )

        status = await runtime.get_status(reference)
        if status.state in _LIVE_STATES:
            # The stream ended but the workload did not: its supervisor is
            # gone (ADR-014 §1a). The same workload, unobserved -- so the
            # attempt's stream moves to a new generation and the gap is
            # recorded, and no attempt is created. Its outcome is then read
            # from the runtime alone.
            self.repository.record_telemetry_degraded(
                attempt_id,
                reason=status.detail or "the telemetry stream ended while the workload ran",
                actor=_ACTOR,
            )
            while status.state in _LIVE_STATES:
                await asyncio.sleep(_OUTCOME_POLL_SECONDS)
                status = await runtime.get_status(reference)

        outcome = _RUNTIME_OUTCOME.get(status.state)
        if outcome is None:
            # Nothing observed how it ended. It may have succeeded and
            # published, or failed halfway: guessing either would write a
            # fact nobody established, so the attempt stays unsettled and
            # wait() escalates.
            self._escalations[str(experiment_id)] = (
                f"attempt {attempt_id} ended with no recorded outcome "
                f"({status.detail or status.state}); it is left unsettled rather than guessed"
            )
            return
        run_outcome: RunStatus | None = outcome[1]
        oom_observed = numerical_observed = False
        if run_outcome is RunStatus.SUCCEEDED and (
            unconfirmed := self.repository.intervention_directives.unconfirmed_for_attempt(
                str(attempt_id)
            )
        ):
            # The process exited cleanly, but the control plane cannot vouch
            # for the trajectory it trained: a directed intervention was never
            # confirmed. The attempt's mechanical outcome stays truthful; the
            # Run fails and nothing is evaluated.
            run_outcome = RunStatus.FAILED
            self._escalations[str(experiment_id)] = (
                f"attempt {attempt_id} succeeded without confirming intervention "
                f"application(s) {', '.join(str(d.application_id) for d in unconfirmed)}; "
                f"its Run failed rather than accept an unknown trajectory"
            )
        if outcome[0] is RunAttemptStatus.FAILED:
            target = RuntimeOperationTarget(kind="training-attempt", id=str(attempt_id))
            oom_observed = any(
                IncidentCategory.CUDA_OOM
                in {candidate.category for candidate in observed.candidates}
                for observed in self.repository.incidents.for_attempt(target)
            )
            numerical_observed = (
                not oom_observed
                and self._numerical_incident(attempt_id) is not None
                and self.repository.aggregates.load_experiment(
                    str(experiment_id)
                ).numerical_recovery
                is not None
            )
            if (oom_observed or numerical_observed) and self._checkpoint_manager is not None:
                # Recovery may still create a successor under this logical Run.
                # The recovery workflow later decides whether to continue or fail it.
                run_outcome = None
        self._settle(attempt_id, run_id, outcome[0], run_outcome, status=status)
        self._reconcile_cancellations(experiment_id)
        if oom_observed and run_outcome is None:
            await self._drive_oom_recovery(experiment_id, run_id, attempt_id)
        elif numerical_observed and run_outcome is None:
            await self._drive_numerical_recovery(experiment_id, run_id, attempt_id)
        if run_outcome is RunStatus.SUCCEEDED:
            run = self.repository.aggregates.load_run(str(run_id))
            await self._continue_to_evaluation(experiment_id, run.node_id)

    # ---- issuing and reconciling submissions (ADR-013) -----------------------

    def _oom_incident(self, attempt_id: RunAttemptId) -> bool:
        target = RuntimeOperationTarget(kind="training-attempt", id=str(attempt_id))
        return any(
            candidate.category is IncidentCategory.CUDA_OOM
            for observed in self.repository.incidents.for_attempt(target)
            for candidate in observed.candidates
        )

    def _numerical_incident(self, attempt_id: RunAttemptId) -> Incident | None:
        """The attempt's first nonfinite incident, if numerical recovery is its concern."""
        target = RuntimeOperationTarget(kind="training-attempt", id=str(attempt_id))
        return next(
            (
                observed
                for observed in self.repository.incidents.for_attempt(target)
                if any(
                    candidate.category in NONFINITE_CATEGORIES for candidate in observed.candidates
                )
            ),
            None,
        )

    def _record_intervention_applied(
        self,
        experiment_id: ExperimentId,
        attempt_id: RunAttemptId,
        observation: InterventionAppliedPayload,
        position: tuple[int, int],
    ) -> None:
        """Record a worker-confirmed effect, and only for a directive this attempt carried.

        The application and the telemetry cursor commit together: a crash
        either loses both, and the event is redelivered, or keeps both.
        """
        directive = self.repository.intervention_directives.get(observation.application_id)
        if (
            directive is None
            or directive.attempt_id != attempt_id
            or str(directive.intervention_id) != observation.intervention_id
        ):
            self._escalations[str(experiment_id)] = (
                f"attempt {attempt_id} reported an intervention application "
                f"{observation.application_id} that no directive of it names"
            )
            return
        try:
            self.repository.record_intervention_application(
                directive.intervention_id,
                application_id=directive.application_id,
                attempt_id=attempt_id,
                position=TrainingPosition(optimizer_step=observation.optimizer_step),
                observed_previous_value=observation.previous_value,
                applied_value=observation.applied_value,
                actor=_ACTOR,
                telemetry_position=position,
            )
        except (ProvenanceError, IdempotencyConflictError) as error:
            self._escalations[str(experiment_id)] = str(error)

    async def _drive_numerical_recovery(
        self, experiment_id: ExperimentId, run_id: RunId, attempt_id: RunAttemptId
    ) -> None:
        """Govern and consume one numerical episode as a checkpoint-backed successor.

        Spec 08 §9a. The decision is a governed ``ChangeLearningRate``; its
        authorized outcome is a ``TrainingIntervention``; the successor attempt
        restores a validated checkpoint from before the unsafe step and carries
        the intervention -- with any re-application a rollback requires -- as
        directives the worker applies and confirms. An approval-pending Action
        leaves the Run active; approval or attach resumes from the record.
        """
        observed = self._numerical_incident(attempt_id)
        experiment = self.repository.aggregates.load_experiment(str(experiment_id))
        policy = experiment.numerical_recovery
        if observed is None or policy is None:
            return
        target = RuntimeOperationTarget(kind="training-attempt", id=str(attempt_id))
        episode = self.repository.recovery_episodes.for_attempt(target)
        if self._checkpoint_manager is None:
            if episode is not None:
                self._escalations[str(experiment_id)] = (
                    f"numerical episode {episode.id} needs its checkpoint manager to resume"
                )
            else:
                self._settle(attempt_id, run_id, RunAttemptStatus.FAILED, RunStatus.FAILED)
            return
        request = None
        if episode is None and self._recovery_request_for_incident is not None:
            request = self._recovery_request_for_incident(observed)
        if episode is None and request is None:
            self._escalations[str(experiment_id)] = (
                f"numerical attempt {attempt_id} needs an explicit RecoveryRequest "
                "before its first recovery decision"
            )
            return
        coordinator = RecoveryCoordinator(self.repository, self._checkpoint_manager)
        plan = await coordinator.plan(str(observed.id), request)
        if not self.repository.recovery_plans.is_effective_and_fresh(str(plan.id)):
            self._escalations[str(experiment_id)] = (
                f"numerical episode {plan.episode_id} has uncovered or changed evidence"
            )
            return
        run = self.repository.aggregates.load_run(str(run_id))
        assert experiment.compiler is not None
        compiler = self._compiler(experiment.compiler.name)
        _require_version("compiler", experiment.compiler, compiler.descriptor.plugin_version)
        runtime = self._recorded_runtime(experiment)
        capabilities = runtime.capabilities()

        def resolver(candidate_attempt: RunAttempt, **options: Any) -> ResolvedExecutionPlan:
            return self._plan(experiment, run, candidate_attempt, compiler, **options)

        existing = self.repository.numerical_recovery_bindings.for_plan(str(plan.id))
        if existing is None:
            try:
                inputs = self.repository.numerical_recovery_inputs(str(plan.id), policy)
            except (ProvenanceError, ValidationError) as error:
                self._escalations[str(experiment_id)] = (
                    f"numerical episode {plan.episode_id} cannot be planned: {error}"
                )
                return
            proposed = NumericalRecoveryPlanner().plan(inputs)
            if isinstance(proposed, NumericalEscalation):
                self._settle(attempt_id, run_id, RunAttemptStatus.FAILED, RunStatus.FAILED)
                return
            governed = self.repository.propose_numerical_recovery_action(
                inputs,
                proposed,
                proposed_by=_ACTOR,
                reason="numerical recovery: lower the learning rate and continue the run",
                policy=self._policy,
                capabilities=capabilities,
            )
        else:
            governed = self.repository.governed_action(existing.action_id)
        action = governed.action
        if action.status is ActionStatus.APPROVAL_PENDING:
            return
        if action.status is ActionStatus.REJECTED:
            self.repository.abandon_numerical_recovery_action(
                action.id, actor=_ACTOR, reason="numerical recovery Action was rejected"
            )
            return
        if action.status not in (ActionStatus.VALIDATED, ActionStatus.APPROVED):
            # EXECUTING: the successor and its intent exist; submission
            # reconciliation and the worker's confirmation own the next step.
            return
        try:
            if self.repository.training_interventions.for_action(str(action.id)) is None:
                self.repository.record_numerical_intervention(
                    action.id,
                    actor=_ACTOR,
                    rationale="loss became nonfinite; lower the learning rate and continue",
                )
        except StaleRecoveryContextError as error:
            self._escalations[str(experiment_id)] = str(error)
            return
        executor = NumericalRecoveryExecutor(
            self.repository, self._checkpoint_manager, resolver, capabilities=capabilities
        )
        while True:
            try:
                successor, operation, _ = await executor.execute(action.id, actor=_ACTOR)
                break
            except CapacityUnavailableError:
                await asyncio.sleep(_CAPACITY_POLL_SECONDS)
            except (NumericalCheckpointUnavailableError, BudgetExhaustedError) as error:
                self.repository.abandon_numerical_recovery_action(
                    action.id, actor=_ACTOR, reason=str(error)
                )
                return
            except (
                StaleRecoveryContextError,
                StalePolicyContextError,
                InterventionReplayError,
            ) as error:
                # Nothing was committed. Unknown lineage fails closed and waits
                # for a human; a changed decision is replanned on attach.
                self._escalations[str(experiment_id)] = str(error)
                return
        await self._issue(
            experiment_id, run_id, successor.id, operation, resolver(successor), runtime
        )

    async def _drive_oom_recovery(
        self, experiment_id: ExperimentId, run_id: RunId, attempt_id: RunAttemptId
    ) -> None:
        """Repair, govern and consume one CUDA OOM episode, or settle its Run.

        All checkpoint I/O is in the executor outside SQLite. An approval-pending
        Action deliberately leaves the Run active; a later approval or attach
        resumes from the stored episode request and Action binding.
        """
        target = RuntimeOperationTarget(kind="training-attempt", id=str(attempt_id))
        incidents = self.repository.incidents.for_attempt(target)
        oom = next(
            (
                observed
                for observed in incidents
                if any(
                    candidate.category is IncidentCategory.CUDA_OOM
                    for candidate in observed.candidates
                )
            ),
            None,
        )
        if oom is None:
            return
        episode = self.repository.recovery_episodes.for_attempt(target)
        if self._checkpoint_manager is None:
            if episode is not None:
                self._escalations[str(experiment_id)] = (
                    f"CUDA OOM episode {episode.id} needs its checkpoint manager to resume"
                )
            else:
                self._settle(attempt_id, run_id, RunAttemptStatus.FAILED, RunStatus.FAILED)
            return
        request = None
        if episode is None and self._recovery_request_for_incident is not None:
            request = self._recovery_request_for_incident(oom)
        if episode is None and request is None:
            self._escalations[str(experiment_id)] = (
                f"CUDA OOM attempt {attempt_id} needs an explicit RecoveryRequest "
                "before its first recovery decision"
            )
            return
        coordinator = RecoveryCoordinator(self.repository, self._checkpoint_manager)
        plan = await coordinator.plan(str(oom.id), request)
        if not self.repository.recovery_plans.is_effective_and_fresh(str(plan.id)):
            self._escalations[str(experiment_id)] = (
                f"CUDA OOM episode {plan.episode_id} has uncovered or changed evidence"
            )
            return
        self.repository.supersede_stale_oom_actions(plan.episode_id, actor=_ACTOR)
        experiment = self.repository.aggregates.load_experiment(str(experiment_id))
        run = self.repository.aggregates.load_run(str(run_id))
        source = self.repository.aggregates.load_attempt(str(attempt_id))
        assert experiment.compiler is not None
        compiler = self._compiler(experiment.compiler.name)
        _require_version("compiler", experiment.compiler, compiler.descriptor.plugin_version)
        runtime = self._recorded_runtime(experiment)
        capabilities = runtime.capabilities()

        def resolver(candidate_attempt: RunAttempt) -> ResolvedExecutionPlan:
            return self._plan(experiment, run, candidate_attempt, compiler)

        source_plan = resolver(source)
        optimization = source_plan.spec.config.get("optimization")
        if not isinstance(optimization, FrozenDict):
            raise ProvenanceError("compiled OOM source has no optimization configuration")
        micro = optimization.get("micro_batch_size")
        accumulation = optimization.get("gradient_accumulation")
        if type(micro) is not int or type(accumulation) is not int:
            raise ProvenanceError("compiled OOM source has invalid batch configuration")
        prior_receipt = self.repository.recovery_execution_receipts.for_successor(str(attempt_id))
        prior_resize = None
        if prior_receipt is not None:
            prior_binding = self.repository.recovery_action_bindings.for_action(
                str(prior_receipt.action_id)
            )
            assert prior_binding is not None
            assert prior_binding.proposal.action_spec.gradient_accumulation is not None
            prior_resize = PriorOOMResize(
                action_id=prior_receipt.action_id,
                successor_attempt_id=attempt_id,
                prior_execution_state_fingerprint=prior_binding.source_execution_state_fingerprint,
                promised_micro_batch_size=prior_binding.proposal.action_spec.micro_batch_size,
                promised_gradient_accumulation=prior_binding.proposal.action_spec.gradient_accumulation,
            )
        try:
            inputs = OOMRecoveryInputsV1(
                plan=plan,
                run_id=run.id,
                candidate_fingerprint=run.candidate_fingerprint,
                execution_state_fingerprint=plan.execution_state_fingerprint,
                current_micro_batch_size=micro,
                current_gradient_accumulation=accumulation,
                world_size=source_plan.spec.resources.workers or 1,
                prior_resize=prior_resize,
            )
        except ValidationError:
            # Another accepted diagnosis blocks this specialised autonomous
            # path; no generic resize Action may bypass episode arbitration.
            self._escalations[str(experiment_id)] = (
                f"CUDA OOM episode {plan.episode_id} requires non-automatic review"
            )
            return
        existing_binding = self.repository.recovery_action_bindings.for_plan(str(plan.id))
        if existing_binding is None:
            proposed = OOMRecoveryPlanner().plan(inputs)
            if isinstance(proposed, OOMEscalation):
                self._settle(attempt_id, run_id, RunAttemptStatus.FAILED, RunStatus.FAILED)
                return
            assert isinstance(proposed, OOMResizeProposal)
            governed = self.repository.propose_oom_recovery_action(
                inputs,
                proposed,
                proposed_by=_ACTOR,
                reason="adaptive CUDA OOM recovery preserving effective batch",
                policy=self._policy,
                capabilities=capabilities,
            )
        else:
            governed = self.repository.governed_action(existing_binding.action_id)
        action = governed.action
        if action.status is ActionStatus.APPROVAL_PENDING:
            return
        if action.status is ActionStatus.REJECTED:
            self.repository.abandon_oom_recovery_action(
                action.id, actor=_ACTOR, reason="recovery Action was rejected"
            )
            return
        if action.status not in (ActionStatus.VALIDATED, ActionStatus.APPROVED):
            # EXECUTING means the durable receipt/operation already exists and
            # submission reconciliation, not proposal, owns the next step.
            return
        executor = OOMRecoveryExecutor(
            self.repository,
            self._checkpoint_manager,
            resolver,
            capabilities=capabilities,
        )
        while True:
            try:
                successor, operation, _ = await executor.execute(action.id, actor=_ACTOR)
                break
            except CapacityUnavailableError:
                # Another live attempt may release the slot. Re-entering the
                # executor also revalidates checkpoint bytes and all guards.
                await asyncio.sleep(_CAPACITY_POLL_SECONDS)
            except (OOMCheckpointUnavailableError, BudgetExhaustedError) as error:
                self.repository.abandon_oom_recovery_action(
                    action.id, actor=_ACTOR, reason=str(error)
                )
                return
            except (StaleRecoveryContextError, StalePolicyContextError) as error:
                # Another writer changed the decision or its governed context.
                # No recovery effect was committed; restart/attach can replan.
                self._escalations[str(experiment_id)] = str(error)
                return
        await self._issue(
            experiment_id, run_id, successor.id, operation, resolver(successor), runtime
        )

    def _plan(
        self,
        experiment: Experiment,
        run: Run,
        attempt: RunAttempt | RunAttemptId,
        compiler: TrainerCompiler,
        *,
        directives: tuple[InterventionDirective, ...] | None = None,
        restore_action_id: str | None = None,
    ) -> ResolvedExecutionPlan:
        """The attempt's execution plan, built from the durable record alone.

        Submission and reconciliation both use this, so a submission re-issued
        after a restart is the same request by construction: the same candidate
        snapshot, seed, output location and target, compiled by the same compiler
        version, then resolved through that attempt's durable overrides,
        checkpoint binding and intervention directives. Its digest is checked
        before anything is issued.

        The managed numerical-recovery control is armed from the experiment's
        recorded ``numerical_recovery`` policy, for a compiled plan with managed
        Native checkpoints -- the only worker that honours it. *directives* and
        *restore_action_id* are given only for a successor not yet recorded;
        otherwise both are read from the record.
        """
        node = self.repository.aggregates.load_node(str(run.node_id))
        assert experiment.runtime is not None and experiment.artifact_root is not None
        assert run.seed is not None
        if isinstance(attempt, RunAttemptId):
            attempt = self.repository.aggregates.get_attempt(str(attempt)) or RunAttempt(
                id=attempt, run_id=run.id, attempt_number=1
            )
        if attempt.run_id != run.id:
            raise ValueError("attempt belongs to a different logical run")
        spec = compiler.compile(
            node.candidate.candidate,
            CompilationContext(
                run_id=str(run.id),
                seed=run.seed,
                output_uri=str(Path(experiment.artifact_root) / str(run.id)),
                checkpoint_store_uri=str(Path(experiment.artifact_root) / "checkpoints"),
            ),
        )
        if directives is None:
            directives = self.repository.intervention_directives.for_attempt(str(attempt.id))
        if restore_action_id is None:
            receipt = self.repository.numerical_recovery_executions.for_successor(str(attempt.id))
            restore_action_id = None if receipt is None else str(receipt.action_id)
        return resolve_training_attempt(
            spec,
            attempt,
            experiment.runtime.kind,
            directives=directives,
            numerical_recovery=experiment.numerical_recovery is not None
            and spec.checkpoint.format == "native-torch/v1",
            restore_action_id=restore_action_id,
        )

    async def _issue(
        self,
        experiment_id: ExperimentId,
        run_id: RunId | EvaluationRunId,
        attempt_id: RunAttemptId | EvaluationAttemptId,
        operation: RuntimeOperation,
        plan: ResolvedExecutionPlan,
        runtime: RuntimeBackend,
    ) -> None:
        """Issue a recorded submission, record its outcome, and observe it.

        The same for training and evaluation: the plan's target says which
        attempt it is for. A runtime's definitive refusal fails the operation
        and settles the attempt. Any other error leaves the operation INTENDED
        -- an unknown outcome is for reconciliation, not to be written down as
        a failure.
        """
        kind = plan.target.kind
        try:
            reference = await runtime.submit_or_get(operation.id, plan)
        except Exception as exc:
            if _is_refusal(exc):
                self.repository.fail_operation(
                    operation.id, expected_revision=operation.revision, actor=_ACTOR
                )
                self._settle_refused(kind, experiment_id, run_id, attempt_id)
                return
            raise
        operation = self.repository.confirm_operation(
            operation.id,
            expected_revision=operation.revision,
            actor=_ACTOR,
            runtime_ref=reference,
        )
        self._adopt(kind, experiment_id, run_id, attempt_id, runtime, reference)

    def _adopt(
        self,
        kind: str,
        experiment_id: ExperimentId,
        run_id: RunId | EvaluationRunId,
        attempt_id: RunAttemptId | EvaluationAttemptId,
        runtime: RuntimeBackend,
        reference: RuntimeRef,
    ) -> None:
        """Observe a workload the runtime has, whoever issued it."""
        observe: Any
        if kind == "training-attempt":
            self._advance_attempt(RunAttemptId(attempt_id), RunAttemptStatus.QUEUED)
            observe = self._observe(
                experiment_id, RunAttemptId(attempt_id), RunId(run_id), runtime, reference
            )
        else:
            self._advance_evaluation_attempt(
                EvaluationAttemptId(attempt_id), EvaluationAttemptStatus.QUEUED
            )
            observe = self._observe_evaluation(
                experiment_id,
                EvaluationAttemptId(attempt_id),
                EvaluationRunId(run_id),
                runtime,
                reference,
            )
        observers = self._controllers.setdefault(str(experiment_id), {})
        observers[str(attempt_id)] = asyncio.create_task(
            observe, name=f"xaytune-controller-{experiment_id}-{attempt_id}"
        )

    def _settle_refused(
        self,
        kind: str,
        experiment_id: ExperimentId,
        run_id: RunId | EvaluationRunId,
        attempt_id: RunAttemptId | EvaluationAttemptId,
    ) -> None:
        """Settle an attempt whose submission the runtime definitively refused."""
        if kind == "training-attempt":
            self._settle(
                RunAttemptId(attempt_id), RunId(run_id), RunAttemptStatus.FAILED, RunStatus.FAILED
            )
            return
        self._settle_evaluation(
            EvaluationAttemptId(attempt_id),
            EvaluationRunId(run_id),
            EvaluationAttemptStatus.FAILED,
            EvaluationRunStatus.FAILED,
        )
        self._reconcile_node_of(EvaluationRunId(run_id))

    def _driving(self, experiment_id: ExperimentId) -> bool:
        """Whether this host is still observing any of the experiment's attempts."""
        observers = self._controllers.get(str(experiment_id), {})
        return any(not task.done() for task in observers.values())

    async def _reconcile(self, experiment: Experiment) -> None:
        """Adopt an experiment's unsettled work after the process driving it died.

        For every attempt that has not reached an outcome -- training or
        evaluation, the same rule -- its submit operation decides what happens,
        and a submission is issued only when the runtime positively says it
        never received it:

        ```text
        CONFIRMED, reference recorded      adopt the workload
        INTENDED/SENT, lookup finds it     confirm it, then adopt
        INTENDED/SENT, lookup: rejected    fail the operation and the attempt
        INTENDED/SENT, lookup: nothing     issue it, under the recorded id and
                                           digest -- or escalate, if the runtime
                                           cannot tell "never received" from
                                           "finished and forgotten" (ADR-013)
        ```

        Then whatever a crash between two commits left behind: a trained node
        whose evaluation cycle never began, an evaluation run with no attempt
        (so no intent, so nothing can have been started), a node whose
        evaluation finished but which never moved on. Each is carried forward
        from the record; none is guessed.

        Each implementation is resolved, and its version checked, only on a
        path that uses it: the runtime where an external effect is looked up,
        adopted or cancelled; the compiler or evaluator only where a request
        is rebuilt. An experiment with nothing unsettled needs none of them,
        so its record can be attached to after the runtime that ran it is
        gone. A cancellation still in flight is carried on: its intended
        effects are issued, which the runtime treats as idempotent under the
        same operation id.
        """
        if experiment.runtime is None:
            # Recorded before a host drove experiments: nothing to adopt with.
            return

        # Normally a no-op: every settlement commits with its transition.
        self.repository.settle_budget(experiment.id, actor=_ACTOR)
        aggregates = self.repository.aggregates
        for node in aggregates.nodes_for_experiment(str(experiment.id)):
            for run in aggregates.runs_for_node(str(node.id)):
                attempts = aggregates.attempts_for_run(str(run.id))
                for attempt in attempts:
                    if not attempt.is_terminal:
                        await self._reconcile_submission(
                            "training-attempt", experiment, run.id, attempt.id
                        )
                latest = max(attempts, key=lambda item: item.attempt_number) if attempts else None
                if (
                    run.status is RunStatus.ACTIVE
                    and latest is not None
                    and latest.status in (RunAttemptStatus.FAILED, RunAttemptStatus.PREEMPTED)
                ):
                    await self._drive_oom_recovery(experiment.id, run.id, latest.id)
                    run = aggregates.load_run(str(run.id))
                    if run.status is RunStatus.ACTIVE and not self._oom_incident(latest.id):
                        await self._drive_numerical_recovery(experiment.id, run.id, latest.id)
            await self._reconcile_evaluation(experiment, node.id)

        for action in self.repository.actions.for_target("experiment", str(experiment.id)):
            if action.type == "cancel-experiment" and not action.is_terminal:
                await self._cancel(experiment.id, reason=action.reason)

    async def _reconcile_evaluation(
        self, experiment: Experiment, node_id: ExperimentNodeId
    ) -> None:
        """Carry one node's evaluation forward from wherever a crash left it."""
        aggregates = self.repository.aggregates
        node = aggregates.load_node(str(node_id))
        if node.status is ExperimentNodeStatus.ACTIVE:
            # Trained, and the cycle never began: begin it now.
            await self._continue_to_evaluation(experiment.id, node.id)
            return
        if node.status is ExperimentNodeStatus.DECIDING:
            # Evaluated, and the crash came before -- or after -- the decision.
            # Deciding again is safe: a cycle already decided is not decided
            # twice.
            self._decide(node.id)
            return
        if node.status is not ExperimentNodeStatus.EVALUATING:
            return

        for evaluation_run in aggregates.evaluation_runs_for_node(
            str(node.id), cycle=node.evaluation_cycle
        ):
            if evaluation_run.is_terminal:
                continue
            attempts = aggregates.evaluation_attempts_for_run(str(evaluation_run.id))
            live = [attempt for attempt in attempts if not attempt.is_terminal]
            for attempt in live:
                await self._reconcile_submission(
                    "evaluation-attempt", experiment, evaluation_run.id, attempt.id
                )
            if not attempts:
                # No attempt means no intent was ever recorded, so nothing can
                # have been started: issuing now is the first issue, not a
                # re-issue.
                await self._start_evaluation_run(experiment, evaluation_run)
            elif not live:
                # Every attempt ended and the run was never settled: the crash
                # fell between the two. Settle it from the attempt's outcome.
                final = attempts[-1]
                outcome = _EVALUATION_OUTCOME.get(final.status.value)
                if outcome is not None:
                    self._settle_evaluation(final.id, evaluation_run.id, *outcome)
        self._reconcile_node(node.id)

    async def _reconcile_submission(
        self,
        kind: str,
        experiment: Experiment,
        run_id: RunId | EvaluationRunId,
        attempt_id: RunAttemptId | EvaluationAttemptId,
    ) -> None:
        """Adopt, settle, escalate or -- only on proven absence -- re-issue one attempt.

        One rule for both workloads. Settling a submission the record already
        shows failed needs nothing but the record. Rediscovering an effect
        that exists needs the runtime, and nothing else. The compiler or
        evaluator becomes a dependency only when the original request must be
        rebuilt, so that is the only branch that resolves one: a workload
        already running is adopted even if what built its request is no
        longer installed.
        """
        submissions = [
            op
            for op in self.repository.operations.for_target(kind, str(attempt_id))
            if op.type == "submit"
        ]
        if not submissions:
            return
        (submission,) = submissions

        if submission.state == "failed":
            # Crashed between recording the refusal and settling the attempt.
            self._settle_refused(kind, experiment.id, run_id, attempt_id)
            return

        # From here on the external effect is interpreted, so the runtime that
        # tracks it is needed -- and it must be the one that recorded it.
        runtime = self._recorded_runtime(experiment)
        if submission.state == "confirmed" and submission.runtime_ref is not None:
            self._adopt(kind, experiment.id, run_id, attempt_id, runtime, submission.runtime_ref)
            return

        outcome = await runtime.lookup_operation(submission.id)
        if outcome is not None and outcome.disposition == "rejected":
            self.repository.fail_operation(
                submission.id, expected_revision=submission.revision, actor=_ACTOR
            )
            self._settle_refused(kind, experiment.id, run_id, attempt_id)
            return
        if outcome is not None:
            assert outcome.runtime_ref is not None
            self.repository.confirm_operation(
                submission.id,
                expected_revision=submission.revision,
                actor=_ACTOR,
                runtime_ref=outcome.runtime_ref,
            )
            self._adopt(kind, experiment.id, run_id, attempt_id, runtime, outcome.runtime_ref)
            return

        resilience = runtime.capabilities().resilience
        if resilience is None or resilience.reports_completed_operations is not True:
            self._escalations[str(experiment.id)] = (
                f"submission {submission.id} for {kind} {attempt_id} is unconfirmed and the "
                f"runtime has no record of it, but this runtime cannot report completed "
                f"operations: it may have run and been forgotten, so it is not re-issued"
            )
            return

        # Only now is the original request rebuilt, so only now is its builder
        # needed -- and it must be the one that built the original.
        plan = self._rebuild_plan(kind, experiment, run_id, attempt_id, submission)
        if plan.request_digest("submit") != submission.request_digest:
            raise ImplementationMismatchError(
                f"re-issuing submission {submission.id} would send a different request than "
                f"the one recorded; what it describes, or how it is built, changed"
            )
        await self._issue(experiment.id, run_id, attempt_id, submission, plan, runtime)

    def _rebuild_plan(
        self,
        kind: str,
        experiment: Experiment,
        run_id: RunId | EvaluationRunId,
        attempt_id: RunAttemptId | EvaluationAttemptId,
        submission: RuntimeOperation,
    ) -> ResolvedExecutionPlan:
        """Rebuild a recorded submission's plan with the implementation that built it.

        Raises:
            ImplementationMismatchError: If the record names no builder, or
                this host provides a different version of it.
            UnknownImplementationError: If this host has no such builder.
        """
        if kind == "training-attempt":
            if experiment.compiler is None:
                raise ImplementationMismatchError(
                    f"submission {submission.id} must be re-issued, but the record names no "
                    f"compiler to rebuild its request with"
                )
            compiler = self._compiler(experiment.compiler.name)
            _require_version("compiler", experiment.compiler, compiler.descriptor.plugin_version)
            run = self.repository.aggregates.load_run(str(run_id))
            attempt = self.repository.aggregates.load_attempt(str(attempt_id))
            return self._plan(experiment, run, attempt, compiler)

        evaluation_run = self.repository.aggregates.load_evaluation_run(str(run_id))
        evaluator = self._recorded_evaluator(evaluation_run.spec)
        return self._evaluation_plan(
            experiment, evaluation_run, EvaluationAttemptId(attempt_id), evaluator
        )

    # ---- evaluation (ADR-015) ------------------------------------------------

    async def _continue_to_evaluation(
        self, experiment_id: ExperimentId, node_id: ExperimentNodeId
    ) -> None:
        """Begin the node's next evaluation cycle, if its training is done and asks for one.

        Only once every training run of the node has ended, one succeeded, and
        no cancellation is in flight: evaluating a node whose experiment is
        being cancelled would start a workload the cancellation must then
        chase. The trained model -- the successful run's ``model`` artifact --
        is the subject. A successful run with no model is still recorded as a
        cycle, with no run in it, so reconciliation reports it stalled rather
        than the node sitting ``ACTIVE`` with nobody saying why.

        The run's seed is the training run's, and its replicate 1: the
        embedded controller's default for the first evaluation sample, not a
        coupling. An evaluation seed means nothing about training, stays
        outside ``EvaluationFingerprint``, and a planner may schedule further
        replicates with seeds of its own.
        """
        aggregates = self.repository.aggregates
        experiment = aggregates.load_experiment(str(experiment_id))
        if experiment.evaluation is None or experiment.is_terminal:
            return
        if self._cancelling(experiment_id):
            return
        node = aggregates.load_node(str(node_id))
        if node.status is not ExperimentNodeStatus.ACTIVE:
            return
        runs = aggregates.runs_for_node(str(node.id))
        if not runs or any(not run.is_terminal for run in runs):
            return
        trained = [run for run in runs if run.status is RunStatus.SUCCEEDED]
        if not trained:
            return
        run = trained[-1]
        subject = _trained_model(aggregates.attempts_for_run(str(run.id)))

        evaluation_runs: tuple[EvaluationRun, ...] = ()
        if subject is not None:
            evaluation_runs = (
                EvaluationRun(
                    id=EvaluationRunId.generate(),
                    experiment_id=experiment.id,
                    node_id=node.id,
                    evaluation_cycle=node.evaluation_cycle + 1,
                    spec=experiment.evaluation,
                    subject=subject,
                    evaluation_fingerprint=experiment.evaluation.evaluation_fingerprint(),
                    seed=run.seed,
                    replicate=1,
                ),
            )
        try:
            node, _ = self.repository.begin_evaluation_cycle(
                node.id, expected_revision=node.revision, runs=evaluation_runs, actor=_ACTOR
            )
        except BudgetExhaustedError as exhausted:
            # The evaluation is the next effect, and the budget has nothing
            # left for it: nothing starts, the node stays trained, and the
            # experiment ends once nothing else is running.
            self.repository.exhaust_budget(experiment.id, reasons=exhausted.reasons, actor=_ACTOR)
            return
        for evaluation_run in evaluation_runs:
            await self._start_evaluation_run(experiment, evaluation_run)
        if not evaluation_runs:
            self._reconcile_node(node.id)

    async def _start_evaluation_run(self, experiment: Experiment, run: EvaluationRun) -> None:
        """Record an evaluation attempt with its intent, then issue it.

        The evaluator that prepares the request is the one the record names,
        at the recorded version: a request this host would build differently
        is refused rather than run under the old one's name.
        """
        aggregates = self.repository.aggregates
        run = aggregates.load_evaluation_run(str(run.id))
        if run.status is EvaluationRunStatus.CREATED:
            run = self.repository.transition_evaluation_run(
                run.id,
                expected_revision=run.revision,
                new_status=EvaluationRunStatus.ACTIVE,
                actor=_ACTOR,
            )
        evaluator = self._recorded_evaluator(run.spec)
        attempt = EvaluationAttempt(
            id=EvaluationAttemptId.generate(),
            evaluation_run_id=run.id,
            attempt_number=len(aggregates.evaluation_attempts_for_run(str(run.id))) + 1,
        )
        try:
            plan = self._evaluation_plan(experiment, run, attempt.id, evaluator)
        except UnsupportedEvaluationError as refusal:
            # Definitive, and before any effect: the spec passed at submission,
            # so what is refused now is the subject or the run. Nothing was
            # issued, so there is no attempt to record -- the run fails with
            # the evaluator's reasons, and the node's cycle is reconciled,
            # which reports it stalled rather than leaving it waiting forever.
            self.repository.transition_evaluation_run(
                run.id,
                expected_revision=run.revision,
                new_status=EvaluationRunStatus.FAILED,
                actor=_ACTOR,
                reason=f"evaluator refused: {'; '.join(refusal.reasons)}",
            )
            self._reconcile_node(run.node_id)
            return
        try:
            attempt, operation = self.repository.create_evaluation_attempt_with_submit_intent(
                attempt, request_digest=plan.request_digest("submit"), actor=_ACTOR
            )
        except BudgetExhaustedError as exhausted:
            # Before any effect, as a refusal is: the run fails with the
            # reason, the cycle is reconciled, and the experiment ends.
            self.repository.transition_evaluation_run(
                run.id,
                expected_revision=run.revision,
                new_status=EvaluationRunStatus.FAILED,
                actor=_ACTOR,
                reason="budget exhausted: " + "; ".join(exhausted.reasons),
            )
            self._reconcile_node(run.node_id)
            self.repository.exhaust_budget(experiment.id, reasons=exhausted.reasons, actor=_ACTOR)
            return
        runtime = self._recorded_runtime(experiment)
        await self._issue(experiment.id, run.id, attempt.id, operation, plan, runtime)

    def _evaluation_plan(
        self,
        experiment: Experiment,
        run: EvaluationRun,
        attempt_id: EvaluationAttemptId,
        evaluator: Evaluator,
    ) -> ResolvedExecutionPlan:
        """The evaluation attempt's plan, built from the durable record alone.

        Like :meth:`_plan`, so a re-issued evaluation is the same request by
        construction, and its digest is still checked. On the experiment's
        runtime: choosing another for evaluation is a later decision.

        Raises:
            ValueError: If the evaluator prepared a spec for another evaluation
                or another subject than the run names.
        """
        assert experiment.runtime is not None and experiment.artifact_root is not None
        spec = evaluator.prepare(
            run.subject,
            run.spec,
            EvaluationContext(
                experiment_id=str(experiment.id),
                node_id=str(run.node_id),
                evaluation_run_id=str(run.id),
                seed=run.seed,
                replicate=run.replicate,
                output_uri=str(Path(experiment.artifact_root) / "evaluations" / str(run.id)),
            ),
        )
        if spec.evaluation_fingerprint != run.evaluation_fingerprint or spec.subject != run.subject:
            raise ValueError(
                f"evaluator {evaluator.descriptor.name!r} prepared a request for another "
                f"evaluation or subject than run {run.id} names"
            )
        return ResolvedExecutionPlan(
            spec=spec,
            runtime=experiment.runtime.kind,
            target=RuntimeOperationTarget(kind="evaluation-attempt", id=str(attempt_id)),
        )

    async def _observe_evaluation(
        self,
        experiment_id: ExperimentId,
        attempt_id: EvaluationAttemptId,
        run_id: EvaluationRunId,
        runtime: RuntimeBackend,
        reference: RuntimeRef,
    ) -> None:
        """Turn one evaluation's telemetry into durable transitions, then settle it.

        Success takes **two** facts: an ``EvaluationCompleted`` carrying the
        metrics, and the runtime reporting that the workload succeeded. An exit
        of 0 with no completion is not a success: the evaluation produced no
        result, so it failed.

        The completion is held durably, with its telemetry position, in the
        same commit that advances the cursor to it
        (``hold_evaluation_completion``). It is not yet an ``EvaluationResult``
        -- the workload may still fail on its way out, and runtime success is
        still required. Because it is durable, a change of stream generation
        or a controller restart cannot lose it: whichever controller sees the
        workload end reads the completion from the record, not from a stream.
        """
        aggregates = self.repository.aggregates
        generation, sequence = aggregates.telemetry_position(
            str(attempt_id), kind="evaluation-attempt"
        )
        async for envelope in runtime.watch(
            reference, StreamCursor(generation=generation, sequence=sequence)
        ):
            observation = envelope.payload.data
            position = (envelope.stream_generation, envelope.sequence)
            if isinstance(observation, WorkerReadyPayload):
                self._advance_evaluation_attempt(
                    attempt_id, EvaluationAttemptStatus.STARTING, position
                )
            elif isinstance(observation, EvaluationStartedPayload):
                self._advance_evaluation_attempt(
                    attempt_id, EvaluationAttemptStatus.RUNNING, position
                )
            elif isinstance(observation, EvaluationCompletedPayload):
                # Held durably, not in memory: the stream it arrived on may
                # die before the workload ends, and this controller with it.
                self.repository.hold_evaluation_completion(
                    attempt_id, observation, telemetry_position=position, actor=_ACTOR
                )
            else:
                self._record_incident(
                    RuntimeOperationTarget(kind="evaluation-attempt", id=str(attempt_id)), envelope
                )

        status = await runtime.get_status(reference)
        if status.state in _LIVE_STATES:
            self.repository.record_telemetry_degraded(
                attempt_id,
                reason=status.detail or "the telemetry stream ended while the workload ran",
                actor=_ACTOR,
                kind="evaluation-attempt",
            )
            while status.state in _LIVE_STATES:
                await asyncio.sleep(_OUTCOME_POLL_SECONDS)
                status = await runtime.get_status(reference)

        if status.state == "succeeded":
            # Read from the record, not from this stream: the completion may
            # have arrived on an earlier generation, or to an earlier host.
            completion = aggregates.pending_completion(str(attempt_id))
            if completion is not None and completion[0].metrics:
                self._record_evaluation_result(attempt_id, run_id, *completion)
            else:
                # Exit 0 is an operating-system fact. Without the completion
                # carrying its metrics, the evaluation measured nothing that
                # was recorded, and inventing a result is the one thing this
                # must not do.
                self._settle_evaluation(
                    attempt_id,
                    run_id,
                    EvaluationAttemptStatus.FAILED,
                    EvaluationRunStatus.FAILED,
                    reason="the workload exited successfully without reporting its result",
                )
        elif status.state in _EVALUATION_OUTCOME:
            self._settle_evaluation(attempt_id, run_id, *_EVALUATION_OUTCOME[status.state])
        else:
            self._escalations[str(experiment_id)] = (
                f"evaluation attempt {attempt_id} ended with no recorded outcome "
                f"({status.detail or status.state}); it is left unsettled rather than guessed"
            )
            return
        self._reconcile_node_of(run_id)
        self._reconcile_cancellations(experiment_id)

    def _record_evaluation_result(
        self,
        attempt_id: EvaluationAttemptId,
        run_id: EvaluationRunId,
        completion: EvaluationCompletedPayload,
        position: tuple[int, int],
    ) -> None:
        """Record the result a completed evaluation carried, with its success.

        Unless it cannot be attributed to this run. What the worker reported
        is checked against the run first (``result_provenance_problems``): a
        metric from another evaluator, version or seed, or a report naming its
        own producer -- which the worker cannot know, since the result's id
        is assigned here -- fails the evaluation, with the reason recorded.
        A result the record cannot attribute is not a result.
        """
        aggregates = self.repository.aggregates
        run = aggregates.load_evaluation_run(str(run_id))
        result_id = EvaluationId.generate()
        problems: list[str] = []
        artifacts: tuple[ArtifactRef, ...] = ()
        if completion.result_ref is not None:
            report = completion.result_ref
            if report.producer_evaluation_id is not None or report.producer_attempt_id is not None:
                problems.append(
                    f"the worker named a producer for its report {report.id}; only the "
                    f"controller can, once the result exists"
                )
            artifacts = (
                report.model_copy(
                    update={"producer_evaluation_id": result_id, "producer_attempt_id": None}
                ),
            )
        assert completion.metrics is not None
        result = EvaluationResult(
            id=result_id,
            evaluation_run_id=run.id,
            node_id=run.node_id,
            subject=run.subject,
            evaluation_fingerprint=run.evaluation_fingerprint,
            metrics=completion.metrics,
            artifacts=artifacts,
        )
        problems.extend(result_provenance_problems(result, run))
        if problems:
            self._settle_evaluation(
                attempt_id,
                run_id,
                EvaluationAttemptStatus.FAILED,
                EvaluationRunStatus.FAILED,
                reason="the reported result cannot be attributed to this run: "
                + "; ".join(problems),
            )
            return
        # A worker that reported completion without first reporting its start
        # still ran: the machine requires RUNNING before SUCCEEDED.
        attempt = self._advance_evaluation_attempt(attempt_id, EvaluationAttemptStatus.RUNNING)
        self.repository.record_evaluation_result(
            attempt.id,
            result,
            expected_revision=attempt.revision,
            actor=_ACTOR,
            telemetry_position=position,
        )

    def _advance_evaluation_attempt(
        self,
        attempt_id: EvaluationAttemptId,
        target: EvaluationAttemptStatus,
        position: tuple[int, int] | None = None,
    ) -> EvaluationAttempt:
        """Move an evaluation attempt forward to *target*, through each state between.

        Only ever forward, as for training: a late observation is a no-op.
        """
        attempt = self.repository.aggregates.load_evaluation_attempt(str(attempt_id))
        if attempt.is_terminal:
            return attempt
        path = _EVALUATION_PATH
        index = path.index(attempt.status) if attempt.status in path else -1
        steps = path[index + 1 : path.index(target) + 1]
        for number, status in enumerate(steps, start=1):
            attempt = self.repository.transition_evaluation_attempt(
                attempt.id,
                expected_revision=attempt.revision,
                new_status=status,
                actor=_ACTOR,
                telemetry_position=position if number == len(steps) else None,
            )
        return attempt

    def _settle_evaluation(
        self,
        attempt_id: EvaluationAttemptId,
        run_id: EvaluationRunId,
        attempt_status: EvaluationAttemptStatus,
        run_status: EvaluationRunStatus,
        *,
        reason: str | None = None,
    ) -> None:
        """Record how an evaluation that did not succeed ended, and why if known."""
        aggregates = self.repository.aggregates
        attempt = aggregates.load_evaluation_attempt(str(attempt_id))
        if not attempt.is_terminal and EVALUATION_ATTEMPT_MACHINE.can(
            attempt.status, attempt_status
        ):
            self.repository.transition_evaluation_attempt(
                attempt.id,
                expected_revision=attempt.revision,
                new_status=attempt_status,
                actor=_ACTOR,
                reason=reason,
            )
        run = aggregates.load_evaluation_run(str(run_id))
        if not run.is_terminal:
            self.repository.transition_evaluation_run(
                run.id,
                expected_revision=run.revision,
                new_status=run_status,
                actor=_ACTOR,
                reason=reason,
            )

    def _reconcile_node_of(self, run_id: EvaluationRunId) -> None:
        run = self.repository.aggregates.load_evaluation_run(str(run_id))
        self._reconcile_node(run.node_id)

    def _reconcile_node(self, node_id: ExperimentNodeId) -> EvaluationReconciliation | None:
        """Wait, decide or stall a node in ``EVALUATING`` (ADR-015 §5).

        Skipped while a cancellation is in flight: the evaluations it stopped
        ended without results by design, and calling that a stall would
        record an incident the operator caused on purpose.
        """
        node = self.repository.aggregates.load_node(str(node_id))
        if node.status is not ExperimentNodeStatus.EVALUATING:
            return None
        if self._cancelling(node.experiment_id):
            return None
        reconciled = self.repository.reconcile_evaluating_node(node.id, actor=_ACTOR)
        if reconciled is EvaluationReconciliation.DECIDING:
            self._decide(node.id)
        return reconciled

    def _decide(self, node_id: ExperimentNodeId) -> None:
        """Decide a node in ``DECIDING`` from the record, and apply it (PR-015).

        The context is assembled from durable state alone -- the experiment's
        objective and the results of the node's current evaluation cycle --
        and the engine sees nothing else. What it decides is recorded and
        applied in one commit. A cycle decided before, by a controller that
        then died, is recognised by the repository and not decided twice.

        An engine that cannot decide leaves the node ``DECIDING``, with a
        ``DecisionDeferred`` event saying why. Skipped while a cancellation
        is in flight: the experiment is ending, and a decision would race it.
        """
        aggregates = self.repository.aggregates
        node = aggregates.load_node(str(node_id))
        if node.status is not ExperimentNodeStatus.DECIDING:
            return
        experiment = aggregates.load_experiment(str(node.experiment_id))
        if experiment.is_terminal or self._cancelling(experiment.id):
            return
        results = tuple(
            result
            for run in aggregates.evaluation_runs_for_node(
                str(node.id), cycle=node.evaluation_cycle
            )
            if (result := aggregates.evaluation_result_for_run(str(run.id))) is not None
        )
        context = DecisionContext(
            experiment_id=experiment.id,
            node_id=node.id,
            evaluation_cycle=node.evaluation_cycle,
            objective=experiment.objective,
            results=results,
        )
        engine = self._decision_engine
        try:
            proposal = engine.decide(context)
        except UndecidableError as undecidable:
            self.repository.defer_decision(
                node.id,
                engine=f"{engine.name} {engine.version}",
                reasons=undecidable.reasons,
                actor=_ACTOR,
            )
            return
        self.repository.record_decision(
            proposal, expected_node_revision=node.revision, actor=_ACTOR
        )

    def _cancelling(self, experiment_id: ExperimentId) -> bool:
        return any(
            action.type == "cancel-experiment" and not action.is_terminal
            for action in self.repository.actions.for_target("experiment", str(experiment_id))
        )

    def _recorded_evaluator(self, spec: EvaluationSpec) -> Evaluator:
        """The evaluator the record names, at the version it names."""
        evaluator = self._evaluator(spec.evaluator.name)
        _require_version("evaluator", spec.evaluator, evaluator.descriptor.plugin_version)
        return evaluator

    # ---- cancellation (ADR-013 §6) -----------------------------------------

    async def _cancel(self, experiment_id: ExperimentId, *, reason: str) -> None:
        """Record the cancellation saga, then carry out its effects.

        Intent first, in one commit: the experiment's Action and a child
        Action and cancel operation per live attempt -- training or
        evaluation. Then each effect, confirmed once the runtime has accepted
        it. Nothing here moves the experiment: it reaches ``CANCELLED`` when
        reconciliation sees that no attempt is live, which for a running
        workload is after the controller has observed it stop.

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
            for child, operation in children:
                if operation is None or operation.state != "intended":
                    continue
                reference = self._submitted_reference(child.target.kind, child.target.id)
                if reference is None:
                    continue
                # Resolved per effect, so cancelling an experiment with no
                # effect left to issue needs no runtime.
                runtime = self._recorded_runtime(experiment)
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

    def _submitted_reference(self, kind: str, attempt_id: str) -> RuntimeRef | None:
        """The runtime reference the attempt's confirmed submission recorded."""
        for operation in self.repository.operations.for_target(kind, attempt_id):
            if operation.type == "submit" and operation.runtime_ref is not None:
                return operation.runtime_ref
        return None

    # ---- durable writes --------------------------------------------------

    async def _create_attempt(
        self, attempt: RunAttempt, request_digest: str
    ) -> tuple[RunAttempt, RuntimeOperation]:
        """Record a training attempt and its intent, waiting for a parallel-run slot if needed.

        A full capacity is not exhaustion: the attempt waits for a running
        one to end and release its slot, then is recorded. Nothing is written
        while it waits.

        Raises:
            BudgetExhaustedError: If a quota is used up.
        """
        while True:
            try:
                return self.repository.create_attempt_with_submit_intent(
                    attempt, request_digest=request_digest, actor=_ACTOR
                )
            except CapacityUnavailableError:
                await asyncio.sleep(_CAPACITY_POLL_SECONDS)

    def _bind_evaluation(self, spec: EvaluationSpec) -> EvaluationSpec:
        """Resolve the evaluator and the spec, and record which implementation it is (ADR-016).

        ```text
        supports(declared) → resolve() → supports(resolved) → recorded
        ```

        ``resolve()`` pins what the spec names mutably -- a benchmark task and
        its dataset. It is optional (:class:`ResolvableEvaluator`), and asked
        only here: the resolved spec is what is
        recorded and fingerprinted, and nothing after submission, restart
        included, resolves it again. It is judged twice because what a spec
        resolves to can be something the evaluator refuses though the spec
        as declared was not.

        The version and determinism are the evaluator's own declarations, read
        now, so the record says which evaluator measured -- and a restarted
        host rebuilding the request can check it has the same one.

        Raises:
            UnsupportedEvaluationError: If the evaluator cannot run the spec as
                declared or as resolved, or cannot pin it. Asked here, at
                submission, so an evaluation that would be refused is refused
                before hours of training rather than after them.
        """
        evaluator = self._evaluator(spec.evaluator.name)
        name = evaluator.descriptor.name
        support = evaluator.supports(spec)
        if not support:
            raise UnsupportedEvaluationError(name, support.reasons)
        # Optional (ResolvableEvaluator): an evaluator with nothing to pin has
        # its spec recorded as declared.
        resolved = evaluator.resolve(spec) if isinstance(evaluator, ResolvableEvaluator) else spec
        if resolved.evaluator.name != spec.evaluator.name:
            raise UnsupportedEvaluationError(
                name,
                (
                    f"resolving named evaluator {resolved.evaluator.name!r}, not "
                    f"{spec.evaluator.name!r}; resolution pins a spec, it does not reassign it",
                ),
            )
        support = evaluator.supports(resolved)
        if not support:
            raise UnsupportedEvaluationError(
                name, tuple(f"as resolved: {reason}" for reason in support.reasons)
            )
        bound = EvaluatorSpec(
            name=resolved.evaluator.name,
            version=evaluator.descriptor.plugin_version,
            determinism=evaluator.determinism,
            config=resolved.evaluator.config,
        )
        return resolved.model_copy(update={"evaluator": bound})

    def _record_experiment(
        self,
        spec: ExperimentSpec,
        compiler: TrainerCompiler,
        runtime: RuntimeBackend,
        evaluation: EvaluationSpec | None,
        planner: PlannerSpec | None = None,
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
            budget=spec.budget,
            numerical_recovery=spec.numerical_recovery,
            evaluation=evaluation,
            planner=planner,
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

    def _advance_attempt(
        self,
        attempt_id: RunAttemptId,
        target: RunAttemptStatus,
        position: tuple[int, int] | None = None,
    ) -> RunAttempt:
        """Move an attempt forward to *target*, through each state in between.

        Only ever forward: an observation that arrives late -- a
        ``WorkerReady`` read after ``TrainingStarted`` was already applied --
        is a no-op, not a regression.
        """
        attempt = self.repository.aggregates.load_attempt(str(attempt_id))
        if attempt.is_terminal:
            return attempt
        path = _ATTEMPT_PATH
        index = path.index(attempt.status) if attempt.status in path else -1
        steps = path[index + 1 : path.index(target) + 1]
        for number, status in enumerate(steps, start=1):
            attempt = self.repository.transition_attempt(
                attempt.id,
                expected_revision=attempt.revision,
                new_status=status,
                actor=_ACTOR,
                # The cursor moves with the last step the event caused.
                telemetry_position=position if number == len(steps) else None,
            )
        return attempt

    def _record_artifact(
        self,
        attempt_id: RunAttemptId,
        artifact: ArtifactRef,
        position: tuple[int, int] | None = None,
    ) -> None:
        attempt = self.repository.aggregates.load_attempt(str(attempt_id))
        if any(existing.id == artifact.id for existing in attempt.artifact_refs):
            return
        self.repository.record_artifact(
            attempt.id,
            artifact,
            expected_revision=attempt.revision,
            actor=_ACTOR,
            telemetry_position=position,
        )

    def _record_incident(
        self, target: RuntimeOperationTarget, envelope: RuntimeEventEnvelope
    ) -> None:
        """Diagnose structured evidence, record it durably, and stop there."""
        if envelope.target != target:
            raise ProvenanceError("incident envelope belongs to another attempt")
        incident = _INCIDENT_CLASSIFIER.inspect(
            envelope.payload.data,
            self.repository.incident_context(target),
            evidence=envelope.model_dump(mode="json"),
        )
        if incident is not None:
            self.repository.record_incident(incident, actor=_ACTOR)

    def _settle(
        self,
        attempt_id: RunAttemptId,
        run_id: RunId,
        attempt_status: RunAttemptStatus,
        run_status: RunStatus | None,
        *,
        status: RuntimeStatus | None = None,
    ) -> None:
        """Settle one attempt; terminalize its logical Run only when resolved.

        A successful attempt passes through ``RUNNING`` if telemetry never
        reported the start: it cannot have succeeded without running, and the
        machine requires the state. A failed or cancelled one goes straight to
        its outcome from wherever it was. ``run_status=None`` leaves a failed
        attempt under an ACTIVE Run while recovery remains unresolved. A later
        successor attempt can continue that same Run; only a definitive refusal
        or exhaustion moves it to FAILED.
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
        if run_status is not None and not RUN_MACHINE.is_terminal(run.status):
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

    def _recorded_runtime(self, experiment: Experiment) -> RuntimeBackend:
        """The runtime that recorded *experiment*'s effects, at the recorded version.

        For work that touches an effect already recorded. A different version
        may track workloads differently, so it is refused rather than trusted.
        """
        spec = experiment.runtime
        assert spec is not None, "only called for experiments that record a runtime"
        runtime = self._runtime(spec)
        _require_version("runtime", spec, runtime.descriptor.plugin_version)  # type: ignore[attr-defined]
        return runtime

    def _planner(self, spec: PlannerSpec) -> Planner:
        """Bind *spec* to a planner: resolved, its API checked, its config validated (ADR-016).

        Raises:
            UnknownImplementationError: If this host has no planner of the kind.
            PlannerConfigurationError: If the planner refuses the spec.
        """
        factory = self._planners.get(spec.kind)
        if factory is None:
            raise UnknownImplementationError(
                f"no planner of kind {spec.kind!r}; this host knows {sorted(self._planners)}"
            )
        return factory(spec)

    def _recorded_planner(self, experiment: Experiment) -> Planner | None:
        """The planner the record names, at the version it names, or ``None`` if it names none.

        Raises:
            ImplementationMismatchError: If this host's planner of that kind
                is another version.
        """
        spec = experiment.planner
        if spec is None:
            return None
        # Bound by kind and config first, so a version that moved is reported
        # as the mismatch it is rather than as a configuration error.
        planner = self._planner(spec.model_copy(update={"version": None}))
        _require_version("planner", spec, planner.descriptor.plugin_version)
        if planner.spec != spec:
            raise ImplementationMismatchError(
                f"the record's planner spec {spec} does not bind to itself on this host "
                f"({planner.spec}); continuing with a different planner is refused"
            )
        return planner

    def _evaluator(self, name: str) -> Evaluator:
        factory = self._evaluators.get(name)
        if factory is None:
            raise UnknownImplementationError(
                f"no evaluator named {name!r}; this host knows {sorted(self._evaluators)}"
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


def _trained_model(attempts: tuple[RunAttempt, ...]) -> ArtifactRef | None:
    """The model the run's successful attempt published, if it published one."""
    for attempt in reversed(attempts):
        if attempt.status is RunAttemptStatus.SUCCEEDED:
            models = [a for a in attempt.artifact_refs if a.kind == "model"]
            return models[-1] if models else None
    return None


def _is_refusal(exc: BaseException) -> bool:
    """Whether *exc* is a runtime's definitive refusal of a plan."""
    from xaytune.runtimes.local import UnsupportedPlanError

    return isinstance(exc, UnsupportedPlanError)


def _require_version(
    kind: str, spec: CompilerSpec | RuntimeSpec | EvaluatorSpec | PlannerSpec, available: str
) -> None:
    """Refuse to continue work recorded against a different implementation version."""
    if spec.version != available:
        name = spec.kind if isinstance(spec, (RuntimeSpec, PlannerSpec)) else spec.name
        raise ImplementationMismatchError(
            f"the record names {kind} {name!r} at version {spec.version}, but this host "
            f"provides {available}; continuing its work with a different version is refused"
        )
