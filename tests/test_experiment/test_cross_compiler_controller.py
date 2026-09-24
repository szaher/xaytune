"""The controller does not care which trainer runs underneath it.

```text
                 ┌─ NativeCompiler ─ NativeWorker ─ Trainer.train()
ExperimentHandle ┤
                 └─ TRLCompiler ─── TRLWorker ──── SFTTrainer.train()
```

PR-011 showed the compile/execute contract is trainer-neutral. This is the
layer above it: the same spec, differing only in the compiler it names, must
produce the same durable history, the same handle answers and the same
cancellation outcome. What differs is execution identity -- the compiler the
record names and the request the runtime received -- and nothing else.

Weights and losses are not compared, for the reason PR-011 gives.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.training_fixtures import sft_candidate, tiny_dataset, tiny_model
from xaytune.core.domain.objective import Objective, ObjectiveMetric

pytestmark = pytest.mark.trl


@pytest.fixture(autouse=True)
def _trl_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("trl")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "1")


def _spec(root: Path, compiler: str, *, dataset: Path | None = None):
    from xaytune.experiment import CompilerSpec, ExperimentSpec, RuntimeSpec

    return ExperimentSpec(
        name="tiny-sft",
        objective=Objective(primary=ObjectiveMetric(name="loss", direction="minimize")),
        candidate=sft_candidate(
            tiny_model(root / "model"),
            dataset or tiny_dataset(root / "data" / "train.jsonl", "text"),
            "text",
        ),
        seed=7,
        compiler=CompilerSpec(name=compiler),
        runtime=RuntimeSpec(kind="local", config={"root": str(root / "runtime")}),
        artifact_root=str(root / "artifacts"),
    )


def _history(events) -> list[tuple[str, str, str | None]]:
    """What a controller reads: which aggregate did what, without ids or times."""
    return [
        (
            e.aggregate_type,
            e.event_type,
            e.payload.get("status") or e.payload.get("state"),
        )
        for e in events
    ]


def _shape(result) -> tuple:
    """A result with its ids and artifact locations removed."""
    return (
        result.status,
        result.quiescent,
        result.next_stage,
        tuple(
            (
                node.status,
                tuple(
                    (run.status, run.attempt_status, tuple(a.kind for a in run.artifacts))
                    for run in node.runs
                ),
            )
            for node in result.nodes
        ),
    )


def _drive(root: Path, compiler: str, *, cancel: bool = False, dataset: Path | None = None):
    from xaytune.experiment import EmbeddedControllerHost

    async def scenario():
        host = EmbeddedControllerHost(root / "state.db")
        try:
            handle = await host.submit(_spec(root, compiler, dataset=dataset))
            status = await handle.status()
            if cancel:
                await handle.cancel()
            result = await asyncio.wait_for(handle.wait(), timeout=180)
            repo = host.repository
            experiment = repo.aggregates.load_experiment(str(handle.experiment_id))
            (submit,) = [
                op
                for node in repo.aggregates.nodes_for_experiment(str(experiment.id))
                for run in repo.aggregates.runs_for_node(str(node.id))
                for attempt in repo.aggregates.attempts_for_run(str(run.id))
                for op in repo.operations.for_target("training-attempt", str(attempt.id))
                if op.type == "submit"
            ]
            actions = sorted(
                (a.type, a.status.value, a.outcome.value if a.outcome else None)
                for a in repo.actions.for_target("experiment", str(experiment.id))
            )
            return {
                "status": status,
                "result": result,
                "history": repo.events.events_for_experiment(str(experiment.id)),
                "compiler": experiment.compiler,
                "digest": submit.request_digest,
                "actions": actions,
            }
        finally:
            await host.close()

    return asyncio.run(scenario())


def test_the_same_spec_succeeds_identically_on_both_trainers(tmp_path: Path) -> None:
    native = _drive(tmp_path / "native", "native")
    trl = _drive(tmp_path / "trl", "trl")

    # Same control-plane semantics...
    assert native["status"] == trl["status"]
    assert _shape(native["result"]) == _shape(trl["result"])
    assert _history(native["history"]) == _history(trl["history"])
    assert ("RunAttempt", "ArtifactRecorded", "running") in _history(native["history"])
    assert ("Run", "RunStatusChanged", "succeeded") in _history(native["history"]), (
        "equal histories must also be the right history"
    )

    # ...and a different execution, named as such.
    assert (native["compiler"].name, trl["compiler"].name) == ("native", "trl")
    assert native["digest"] != trl["digest"]


def test_a_training_failure_is_recorded_identically_on_both_trainers(tmp_path: Path) -> None:
    runs = {
        compiler: _drive(
            tmp_path / compiler, compiler, dataset=tmp_path / compiler / "missing.jsonl"
        )
        for compiler in ("native", "trl")
    }

    assert _shape(runs["native"]["result"]) == _shape(runs["trl"]["result"])
    assert _history(runs["native"]["history"]) == _history(runs["trl"]["history"])
    assert runs["native"]["result"].next_stage == "decision"


def test_cancellation_settles_identically_on_both_trainers(tmp_path: Path) -> None:
    """The outcome, not the sequence: where the cancel lands in startup is a race.

    Whether the worker had reported ready before the cancel arrived depends on
    scheduling, so the event sequence can differ by a transition. What must
    not differ is how it settles.
    """
    runs = {
        compiler: _drive(tmp_path / compiler, compiler, cancel=True)
        for compiler in ("native", "trl")
    }

    assert _shape(runs["native"]["result"]) == _shape(runs["trl"]["result"])
    assert (
        runs["native"]["actions"]
        == runs["trl"]["actions"]
        == [("cancel-experiment", "succeeded", "applied")]
    )
