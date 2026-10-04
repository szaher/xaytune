"""What an experiment's record says about where it stands (ExperimentResult).

Read entirely from the record, by any host: the embedded controller reports it
from ``wait()``, and a daemon-backed handle reads it without running a
controller of its own (PR-029).
"""

from __future__ import annotations

from xaytune.core.domain.planning import settled_for_planning
from xaytune.core.ids import ExperimentId
from xaytune.core.state.machines import RUN_MACHINE
from xaytune.core.state.status import ExperimentNodeStatus, RunStatus
from xaytune.experiment.handle import (
    EvaluationOutcome,
    ExperimentResult,
    NextStage,
    NodeOutcome,
    RunOutcome,
)
from xaytune.storage.control_plane import ControlPlaneRepository

__all__ = ["experiment_result"]

# A candidate whose decision settled what it is: rejected on a constraint, or
# completed short of the target. A COMPLETED node under an ACTIVE experiment
# can only be a BRANCH -- STOP_SUCCEEDED ends the experiment with it.
_SCIENTIFICALLY_SETTLED = frozenset({ExperimentNodeStatus.REJECTED, ExperimentNodeStatus.COMPLETED})


def experiment_result(
    repository: ControlPlaneRepository, experiment_id: ExperimentId
) -> ExperimentResult:
    """Where the experiment stands, read entirely from the record.

    Shared by every host: what ``wait()`` reports, and what a daemon-backed
    handle reads without a controller of its own (PR-029).
    """
    aggregates = repository.aggregates
    experiment = aggregates.load_experiment(str(experiment_id))

    nodes: list[NodeOutcome] = []
    settled = True
    trained = deciding = evaluating = planned = False
    awaiting_approval, awaiting_execution = repository.resting_actions(str(experiment_id))
    approval_targets = set()
    for action in awaiting_approval:
        binding = repository.recovery_action_bindings.for_action(
            str(action.id)
        ) or repository.numerical_recovery_bindings.for_action(str(action.id))
        if binding is not None and repository.recovery_plans.is_effective_and_fresh(
            str(binding.plan_id)
        ):
            episode = repository.recovery_episodes.get(str(binding.episode_id))
            if episode is not None:
                approval_targets.add(episode.context.target.id)
    for node in aggregates.nodes_for_experiment(str(experiment_id)):
        runs: list[RunOutcome] = []
        for run in aggregates.runs_for_node(str(node.id)):
            attempts = aggregates.attempts_for_run(str(run.id))
            final = max(attempts, key=lambda a: a.attempt_number) if attempts else None
            settled = settled and (
                RUN_MACHINE.is_terminal(run.status)
                or (final is not None and final.is_terminal and str(final.id) in approval_targets)
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
        planned = planned or node.status is ExperimentNodeStatus.PLANNED
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
    operations, actions = repository.unsettled_work(str(experiment_id))
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
    elif planned:
        # An accepted candidate -- branched from a proposal, say -- is
        # waiting for its first run. That is healthy work to do next, not
        # a reason to plan another candidate, and not a failure: a PLANNED
        # node has no run because nothing has realized it yet.
        next_stage = "training"
    elif nodes and all(node.status in _SCIENTIFICALLY_SETTLED for node in nodes):
        if all(_settled_for_planning(repository, node) for node in nodes):
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
        budget=repository.budget_status(experiment.id),
    )


def _settled_for_planning(repository: ControlPlaneRepository, node: NodeOutcome) -> bool:
    decisions = repository.aggregates.decisions_for_node(str(node.node_id))
    latest = max(decisions, key=lambda d: d.evaluation_cycle) if decisions else None
    return settled_for_planning(node.status, None if latest is None else latest.outcome)
