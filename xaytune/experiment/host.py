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

The controller loop only turns observations into transitions. It does not
decide scientific outcomes. A successful run leaves the node ``ACTIVE`` when
no evaluation is configured, because the next thing that can happen to it is
evaluation, and the experiment ``ACTIVE``, because ending an experiment is a
decision (implementation plan, PR-012).

**Evaluation, when the spec asks for it** (ADR-015, PR-013):

```text
training run SUCCEEDED
   ├── node ACTIVE → EVALUATING, cycle n, EvaluationRun   one commit
   ├── EvaluationAttempt + INTENDED operation              one commit
   ├── runtime.submit_or_get()                             the same journal
   ├── observe: EvaluationCompleted(metrics) held until the runtime says
   │   the workload succeeded, then
   │   result + attempt + run SUCCEEDED + cursor           one commit
   └── node EVALUATING → DECIDING                          reconciled, not assumed
```

It reaches ``DECIDING`` and stops: deciding is PR-015's. An evaluation in
flight is adopted after a restart exactly as training is, through the same
reconciliation.
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
from xaytune.core.domain.evaluation import (
    EvaluationAttempt,
    EvaluationResult,
    EvaluationRun,
    EvaluationSpec,
    EvaluatorSpec,
)
from xaytune.core.domain.event import DomainEvent
from xaytune.core.domain.experiment import (
    CandidateSpecSnapshot,
    Experiment,
    ExperimentNode,
)
from xaytune.core.domain.operation import RuntimeOperation, RuntimeOperationTarget
from xaytune.core.domain.run import Run, RunAttempt
from xaytune.core.domain.specs import CompilerSpec, RuntimeSpec
from xaytune.core.errors import XaytuneError
from xaytune.core.execution import ResolvedExecutionPlan
from xaytune.core.ids import (
    EvaluationAttemptId,
    EvaluationId,
    EvaluationRunId,
    ExperimentId,
    ExperimentNodeId,
    RunAttemptId,
    RunId,
)
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
    EvaluationCompletedPayload,
    EvaluationStartedPayload,
    TrainingStartedPayload,
    WorkerReadyPayload,
)
from xaytune.evaluation import EvaluationContext, Evaluator
from xaytune.experiment.handle import (
    EvaluationOutcome,
    ExperimentHandle,
    ExperimentResult,
    NextStage,
    NodeOutcome,
    RunOutcome,
)
from xaytune.experiment.spec import ExperimentSpec
from xaytune.runtimes import RuntimeBackend, RuntimeStatus, StreamCursor
from xaytune.storage.control_plane import ControlPlaneRepository, EvaluationReconciliation
from xaytune.storage.migrations import migrate

__all__ = [
    "ControllerNotRunningError",
    "EmbeddedControllerHost",
    "ImplementationMismatchError",
    "ReconciliationEscalatedError",
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

_LIVE_STATES = frozenset({"pending", "queued", "starting", "running"})
_OUTCOME_POLL_SECONDS = 0.5

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
    """None yet: the evaluators wrapping Xaytune's metrics and lm-eval are PR-014's."""
    return {}


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
            ``EvaluatorSpec`` resolves through. None are built in yet.
    """

    def __init__(
        self,
        state_path: Path | str,
        *,
        compilers: Mapping[str, Callable[[], TrainerCompiler]] | None = None,
        runtimes: Mapping[str, Callable[[Mapping[str, Any]], RuntimeBackend]] | None = None,
        evaluators: Mapping[str, Callable[[], Evaluator]] | None = None,
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
            UnknownImplementationError: If the spec names a compiler, runtime
                or evaluator this host cannot resolve.
            UnsupportedCandidateError: If the compiler cannot run the candidate
                exactly as declared. Nothing is recorded in either case: a spec
                that cannot run is refused at submission, not after.
        """
        compiler = self._compiler(spec.compiler.name)
        support = compiler.supports(spec.candidate)
        if not support:
            raise UnsupportedCandidateError(compiler.descriptor.name, support.reasons)
        runtime = self._runtime(spec.runtime)
        evaluation = None if spec.evaluation is None else self._bind_evaluation(spec.evaluation)

        experiment = self._record_experiment(spec, compiler, runtime, evaluation)
        node = self._record_node(experiment, spec)
        run = self._record_run(node, spec.seed)

        attempt = RunAttempt(id=RunAttemptId.generate(), run_id=run.id, attempt_number=1)
        plan = self._plan(experiment, run, attempt.id, compiler)
        attempt, operation = self.repository.create_attempt_with_submit_intent(
            attempt, request_digest=plan.request_digest("submit"), actor=_ACTOR
        )

        await self._issue(experiment.id, run.id, attempt.id, operation, plan, runtime)
        return ExperimentHandle(experiment.id, self)

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
        elif deciding:
            next_stage = "decision"
        elif trained or evaluating:
            next_stage = "evaluation"
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
        self._settle(attempt_id, run_id, *outcome, status=status)
        self._reconcile_cancellations(experiment_id)
        if outcome[1] is RunStatus.SUCCEEDED:
            run = self.repository.aggregates.load_run(str(run_id))
            await self._continue_to_evaluation(experiment_id, run.node_id)

    # ---- issuing and reconciling submissions (ADR-013) -----------------------

    def _plan(
        self,
        experiment: Experiment,
        run: Run,
        attempt_id: RunAttemptId,
        compiler: TrainerCompiler,
    ) -> ResolvedExecutionPlan:
        """The attempt's execution plan, built from the durable record alone.

        Submission and reconciliation both use this, so a submission re-issued
        after a restart is the same request by construction: the same
        candidate snapshot, seed, output location and target, compiled by the
        same compiler version. Its digest is still checked against the
        recorded one before anything is issued.
        """
        node = self.repository.aggregates.load_node(str(run.node_id))
        assert experiment.runtime is not None and experiment.artifact_root is not None
        assert run.seed is not None
        return ResolvedExecutionPlan(
            spec=compiler.compile(
                node.candidate.candidate,
                CompilationContext(
                    run_id=str(run.id),
                    seed=run.seed,
                    output_uri=str(Path(experiment.artifact_root) / str(run.id)),
                ),
            ),
            runtime=experiment.runtime.kind,
            target=RuntimeOperationTarget(kind="training-attempt", id=str(attempt_id)),
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

        aggregates = self.repository.aggregates
        for node in aggregates.nodes_for_experiment(str(experiment.id)):
            for run in aggregates.runs_for_node(str(node.id)):
                for attempt in aggregates.attempts_for_run(str(run.id)):
                    if not attempt.is_terminal:
                        await self._reconcile_submission(
                            "training-attempt", experiment, run.id, attempt.id
                        )
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
            return self._plan(experiment, run, RunAttemptId(attempt_id), compiler)

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
                    replicate=run.replicate or 1,
                ),
            )
        node, _ = self.repository.begin_evaluation_cycle(
            node.id, expected_revision=node.revision, runs=evaluation_runs, actor=_ACTOR
        )
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
        plan = self._evaluation_plan(experiment, run, attempt.id, evaluator)
        attempt, operation = self.repository.create_evaluation_attempt_with_submit_intent(
            attempt, request_digest=plan.request_digest("submit"), actor=_ACTOR
        )
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
        metrics, and the runtime reporting that the workload succeeded. The
        completion alone is held, not recorded -- the workload may still fail
        on its way out -- and an exit of 0 with no completion is not a
        success either: the evaluation produced no result, so it failed.

        The completion's position is the cursor recorded with the result, in
        the same commit. So a controller that dies holding a completion has
        advanced nothing past it, and the next one reads it again.
        """
        aggregates = self.repository.aggregates
        generation, sequence = aggregates.telemetry_position(
            str(attempt_id), kind="evaluation-attempt"
        )
        completion: tuple[EvaluationCompletedPayload, tuple[int, int]] | None = None
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
                completion = (observation, position)

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
        """Record the result a completed evaluation carried, with its success."""
        aggregates = self.repository.aggregates
        run = aggregates.load_evaluation_run(str(run_id))
        # A worker that reported completion without first reporting its start
        # still ran: the machine requires RUNNING before SUCCEEDED.
        attempt = self._advance_evaluation_attempt(attempt_id, EvaluationAttemptStatus.RUNNING)
        result_id = EvaluationId.generate()
        artifacts: tuple[ArtifactRef, ...] = ()
        if completion.result_ref is not None:
            report = completion.result_ref
            if report.producer_evaluation_id is None:
                report = report.model_copy(update={"producer_evaluation_id": result_id})
            artifacts = (report,)
        assert completion.metrics is not None
        self.repository.record_evaluation_result(
            attempt.id,
            EvaluationResult(
                id=result_id,
                evaluation_run_id=run.id,
                node_id=run.node_id,
                subject=run.subject,
                evaluation_fingerprint=run.evaluation_fingerprint,
                metrics=completion.metrics,
                artifacts=artifacts,
            ),
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
    ) -> None:
        """Record how an evaluation that did not succeed ended."""
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
            )
        run = aggregates.load_evaluation_run(str(run_id))
        if not run.is_terminal:
            self.repository.transition_evaluation_run(
                run.id, expected_revision=run.revision, new_status=run_status, actor=_ACTOR
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
        return self.repository.reconcile_evaluating_node(node.id, actor=_ACTOR)

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

    def _bind_evaluation(self, spec: EvaluationSpec) -> EvaluationSpec:
        """Resolve the evaluator and record which implementation it is (ADR-016).

        The version and determinism are the evaluator's own declarations, read
        now, so the record says which evaluator measured -- and a restarted
        host rebuilding the request can check it has the same one.
        """
        evaluator = self._evaluator(spec.evaluator.name)
        bound = EvaluatorSpec(
            name=spec.evaluator.name,
            version=evaluator.descriptor.plugin_version,
            determinism=evaluator.determinism,
            config=spec.evaluator.config,
        )
        return spec.model_copy(update={"evaluator": bound})

    def _record_experiment(
        self,
        spec: ExperimentSpec,
        compiler: TrainerCompiler,
        runtime: RuntimeBackend,
        evaluation: EvaluationSpec | None,
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
            evaluation=evaluation,
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
    kind: str, spec: CompilerSpec | RuntimeSpec | EvaluatorSpec, available: str
) -> None:
    """Refuse to continue work recorded against a different implementation version."""
    if spec.version != available:
        name = spec.kind if isinstance(spec, RuntimeSpec) else spec.name
        raise ImplementationMismatchError(
            f"the record names {kind} {name!r} at version {spec.version}, but this host "
            f"provides {available}; continuing its work with a different version is refused"
        )
