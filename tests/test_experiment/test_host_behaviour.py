"""EmbeddedControllerHost: what it refuses, how it fails, and what it will not claim."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.training_fixtures import sft_candidate, tiny_dataset, tiny_model
from xaytune.core.domain.event import DomainEvent
from xaytune.core.domain.objective import Objective, ObjectiveMetric
from xaytune.core.ids import EventId, ExperimentId
from xaytune.core.refs import Actor, RuntimeRef
from xaytune.core.state.status import (
    ExperimentNodeStatus,
    ExperimentStatus,
    RunAttemptStatus,
    RunStatus,
)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "1")


def _spec(tmp_path: Path, *, dataset: Path | None = None, **overrides: object):
    from xaytune.experiment import CompilerSpec, ExperimentSpec, RuntimeSpec

    fields: dict[str, object] = {
        "name": "tiny-sft",
        "objective": Objective(primary=ObjectiveMetric(name="loss", direction="minimize")),
        "candidate": sft_candidate(
            tiny_model(tmp_path / "model"),
            dataset or tiny_dataset(tmp_path / "data" / "train.jsonl", "text"),
            "text",
        ),
        "seed": 7,
        "compiler": CompilerSpec(name="native"),
        "runtime": RuntimeSpec(kind="local", config={"root": str(tmp_path / "runtime")}),
        "artifact_root": str(tmp_path / "artifacts"),
    }
    fields.update(overrides)
    return ExperimentSpec(**fields)  # type: ignore[arg-type]


def _experiments(host) -> list:
    return host.repository._connection.execute("SELECT id FROM experiments").fetchall()


# ---- refused at submission, before anything is written -------------------


def test_a_candidate_the_compiler_cannot_run_is_refused_and_nothing_is_recorded(
    tmp_path: Path,
) -> None:
    """ADR-016: refusing late is the failure to avoid."""
    from xaytune.compilation import UnsupportedCandidateError
    from xaytune.experiment import EmbeddedControllerHost

    spec = _spec(tmp_path)
    alpaca = spec.candidate.model_copy(
        update={"data": spec.candidate.data.model_copy(update={"format": "alpaca"})}
    )

    async def scenario():
        host = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            with pytest.raises(UnsupportedCandidateError, match="data.format"):
                await host.submit(
                    spec.model_copy(
                        update={
                            "candidate": alpaca,
                            "compiler": spec.compiler.model_copy(update={"name": "trl"}),
                        }
                    )
                )
            return _experiments(host)
        finally:
            await host.close()

    assert asyncio.run(scenario()) == []


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("compiler", {"name": "deepspeed-magic"}, "no compiler named"),
        ("runtime", {"kind": "kubernetes"}, "no runtime of kind"),
    ],
)
def test_an_implementation_the_host_cannot_resolve_is_refused(
    tmp_path: Path, field: str, value: dict, match: str
) -> None:
    from xaytune.experiment import (
        CompilerSpec,
        EmbeddedControllerHost,
        RuntimeSpec,
        UnknownImplementationError,
    )

    spec = _spec(tmp_path)
    replacement = CompilerSpec(**value) if field == "compiler" else RuntimeSpec(**value)

    async def scenario():
        host = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            with pytest.raises(UnknownImplementationError, match=match):
                await host.submit(spec.model_copy(update={field: replacement}))
            return _experiments(host)
        finally:
            await host.close()

    assert asyncio.run(scenario()) == []


def test_a_caller_cannot_claim_an_implementation_version(tmp_path: Path) -> None:
    """The version is the host's to resolve and record, not the caller's to assert."""
    from xaytune.experiment import CompilerSpec

    with pytest.raises(ValueError, match="resolved by the host"):
        _spec(tmp_path, compiler=CompilerSpec(name="native", version="9.9.9"))


def test_a_live_object_cannot_be_submitted(tmp_path: Path) -> None:
    """ADR-016 AC-5: refused at construction, naming the field."""
    from xaytune.experiment import RuntimeSpec

    with pytest.raises(ValueError, match="config"):
        _spec(tmp_path, runtime=RuntimeSpec(kind="local", config={"root": object()}))


def test_a_relative_artifact_root_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="artifact_root"):
        _spec(tmp_path, artifact_root="artifacts")


# ---- a failed run ----------------------------------------------------------


def test_a_failed_run_leaves_the_candidate_awaiting_failure_handling(tmp_path: Path) -> None:
    """Failure is recorded on the run; what it means for the candidate is not decided here.

    The node does not fail because one run did: retrying, recovering or giving
    up is failure handling, which does not exist yet. ``next_stage`` names that
    work, and it is not ``DECIDING`` -- the node never went through evaluation.
    """
    from xaytune.experiment import EmbeddedControllerHost

    async def scenario():
        host = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            handle = await host.submit(_spec(tmp_path, dataset=tmp_path / "missing.jsonl"))
            return await asyncio.wait_for(handle.wait(), timeout=120)
        finally:
            await host.close()

    result = asyncio.run(scenario())

    assert result.status is ExperimentStatus.ACTIVE
    assert result.quiescent is True
    assert result.next_stage == "failure-handling"
    (node,) = result.nodes
    assert node.status is ExperimentNodeStatus.ACTIVE
    (run,) = node.runs
    assert (run.status, run.attempt_status) == (RunStatus.FAILED, RunAttemptStatus.FAILED)
    assert run.artifacts == ()


# ---- what the host does not claim ------------------------------------------


def test_waiting_on_work_no_controller_is_driving_says_so(tmp_path: Path) -> None:
    """Not restart reconciliation: a new host cannot settle an orphaned attempt.

    The first host is closed while training is still running, which cancels
    its controller task. A second host on the record finds an unsettled run
    and nothing driving it -- and says that, rather than waiting forever for
    an outcome nobody will record. Adopting it is PR-012a.
    """
    from xaytune.experiment import ControllerNotRunningError, EmbeddedControllerHost

    async def orphan():
        host = EmbeddedControllerHost(tmp_path / "state.db")
        handle = await host.submit(_spec(tmp_path))
        await host.close()
        return handle.experiment_id

    async def attach(experiment_id):
        host = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            handle = await host.attach(experiment_id)
            with pytest.raises(ControllerNotRunningError, match="PR-012a"):
                await handle.wait()
        finally:
            await host.close()

    experiment_id = asyncio.run(orphan())
    asyncio.run(attach(experiment_id))


# ---- events() shares the loop ----------------------------------------------


class _LongHistory:
    """Just the part of a host ``events()`` reads: pages of recorded events."""

    def __init__(self, count: int) -> None:
        self.events = [
            DomainEvent(
                id=EventId.generate(),
                sequence=n,
                aggregate_type="Experiment",
                aggregate_id="e",
                aggregate_revision=n,
                event_type="Replayed",
                experiment_id="e",
                actor=Actor(type="system", id="test"),
            )
            for n in range(1, count + 1)
        ]

    def _events_after(self, _experiment_id, sequence: int, limit: int = 100):
        return tuple(e for e in self.events if (e.sequence or 0) > sequence)[:limit]


def test_replaying_a_long_history_does_not_starve_the_controller() -> None:
    """Yielding an event does not give the loop a turn; awaiting does.

    The controller that writes events shares the loop with the handle reading
    them. A replay that never awaited while pages remained would hold the loop
    until it ended -- and with a cursor bug, forever.
    """
    from xaytune.experiment import ExperimentHandle

    async def scenario() -> bool:
        ran = asyncio.Event()

        async def controller() -> None:
            ran.set()

        task = asyncio.ensure_future(controller())
        handle = ExperimentHandle(ExperimentId("exp_x"), _LongHistory(1000))  # type: ignore[arg-type]
        seen = 0
        ran_during_replay = False
        async for _event in handle.events():
            seen += 1
            ran_during_replay = ran_during_replay or ran.is_set()
            if seen == 1000:
                break
        await task
        return ran_during_replay

    assert asyncio.run(scenario()), "the controller never ran while history was replayed"


# ---- quiescence covers control work, not only runs -------------------------


def test_a_finished_run_with_unresolved_control_work_is_not_quiescent(tmp_path: Path) -> None:
    """A cancellation raced natural completion, and its effect has no outcome.

    ```text
    cancel Action            EXECUTING
    cancel RuntimeOperation  INTENDED
    attempt, run             SUCCEEDED
    ```

    Every run is terminal, and yet an effect was requested that nothing has
    resolved. "Nothing left to do" would be false, so the result is not
    quiescent -- and with no controller here to resolve it, ``wait()`` says so
    rather than returning.
    """
    from tests.test_storage.conftest import make_attempt, make_experiment, make_node, make_run
    from xaytune.experiment import ControllerNotRunningError, EmbeddedControllerHost

    actor = Actor(type="system", id="test")

    async def scenario():
        host = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            repo = host.repository
            experiment = repo.create_experiment(make_experiment(), actor=actor)
            experiment = repo.transition_experiment(
                experiment.id, expected_revision=0, new_status=ExperimentStatus.ACTIVE, actor=actor
            )
            node = repo.create_node(make_node(experiment), actor=actor)
            run = repo.create_run(make_run(node), actor=actor)
            run = repo.transition_run(
                run.id, expected_revision=run.revision, new_status=RunStatus.ACTIVE, actor=actor
            )
            attempt, submit = repo.create_attempt_with_submit_intent(
                make_attempt(run), request_digest="sha256:x", actor=actor
            )
            repo.confirm_operation(
                submit.id,
                expected_revision=submit.revision,
                actor=actor,
                runtime_ref=RuntimeRef(backend="local", external_id="op_x"),
            )
            for status in (
                RunAttemptStatus.QUEUED,
                RunAttemptStatus.STARTING,
                RunAttemptStatus.RUNNING,
            ):
                attempt = repo.transition_attempt(
                    attempt.id, expected_revision=attempt.revision, new_status=status, actor=actor
                )

            # The cancellation is recorded while the attempt is live...
            repo.request_experiment_cancellation(experiment.id, reason="stop", actor=actor)
            # ...and the attempt finishes on its own before the effect resolves.
            repo.transition_attempt(
                attempt.id,
                expected_revision=attempt.revision,
                new_status=RunAttemptStatus.SUCCEEDED,
                actor=actor,
            )
            repo.transition_run(
                run.id, expected_revision=run.revision, new_status=RunStatus.SUCCEEDED, actor=actor
            )

            result = host._result(experiment.id)
            handle = await host.attach(experiment.id)
            with pytest.raises(ControllerNotRunningError):
                await handle.wait()
            return result
        finally:
            await host.close()

    result = asyncio.run(scenario())

    (node,) = result.nodes
    (run,) = node.runs
    assert run.status is RunStatus.SUCCEEDED, "every run is terminal..."
    assert result.quiescent is False, "...and there is still work nobody has resolved"
