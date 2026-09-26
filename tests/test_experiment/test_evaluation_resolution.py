"""An evaluation is resolved once, at submission, and judged before and after (PR-014b).

```text
submit:   supports(declared) → resolve() → supports(resolved) → recorded + fingerprinted
attach:   the record → prepare()                      resolve() is never asked again
```

With the scripted evaluator, whose ``resolve()`` pins a ``resolve_to`` value
as ``pin`` -- a stand-in for pinning a task and its dataset -- so the resolved
spec can be told from the one submitted. What the lm-eval evaluator resolves
is ``tests/test_evaluation/test_lmeval_evaluator.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

import pytest

from tests.evaluation_fixtures import ScriptedEvaluator, evaluation
from tests.test_experiment.test_evaluation_lifecycle import _drive
from tests.test_experiment.test_evaluation_restart import (
    _adopt,
    _decided_once,
    _evaluator_workloads,
)
from tests.test_experiment.test_restart_reconciliation import _CountingSubmissions, _crash, _spec
from xaytune.compilation import SupportResult
from xaytune.core.capabilities import CapabilityDocument
from xaytune.core.domain.evaluation import EvaluationSpec
from xaytune.core.immutable import thaw
from xaytune.evaluation import Evaluator, ResolvableEvaluator, UnsupportedEvaluationError


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "1")


def _pinning(tmp_path: Path, **config: object) -> Any:
    spec = evaluation(value=0.8, resolve_to="pinned-at-submission", **config)
    return _spec(tmp_path).model_copy(update={"evaluation": spec})


class _Counting(ScriptedEvaluator):
    resolved: ClassVar[int] = 0

    def resolve(self, spec: EvaluationSpec) -> EvaluationSpec:
        type(self).resolved += 1
        return super().resolve(spec)


class _NeverAgain(ScriptedEvaluator):
    """What a restarted host has: an evaluator that must not be asked to resolve."""

    def resolve(self, spec: EvaluationSpec) -> EvaluationSpec:
        raise AssertionError("an evaluation was resolved again after submission")


class _ResolvesIntoTheUnsupported(ScriptedEvaluator):
    """Accepts any declared spec; what every spec resolves to, it refuses."""

    def supports(self, spec: EvaluationSpec) -> SupportResult:
        if "generation" in spec.evaluator.config:
            return SupportResult(supported=False, reasons=("it generates",))
        return SupportResult(supported=True)

    def resolve(self, spec: EvaluationSpec) -> EvaluationSpec:
        config = {**thaw(spec.evaluator.config), "generation": "greedy"}
        return spec.model_copy(
            update={"evaluator": spec.evaluator.model_copy(update={"config": config})}
        )


class _Reassigns(ScriptedEvaluator):
    def resolve(self, spec: EvaluationSpec) -> EvaluationSpec:
        return spec.model_copy(
            update={"evaluator": spec.evaluator.model_copy(update={"name": "another"})}
        )


class _BeforeResolution:
    """The evaluator contract as it was before resolution existed: no ``resolve()``."""

    descriptor = ScriptedEvaluator.descriptor
    determinism = ScriptedEvaluator.determinism

    def capabilities(self) -> CapabilityDocument:
        return CapabilityDocument()

    def supports(self, spec: EvaluationSpec) -> SupportResult:
        return SupportResult(supported=True)

    def prepare(self, subject: Any, spec: EvaluationSpec, context: Any) -> Any:
        return ScriptedEvaluator().prepare(subject, spec, context)


def _refused(tmp_path: Path, evaluator: type) -> tuple[UnsupportedEvaluationError, tuple]:
    from xaytune.experiment import EmbeddedControllerHost

    async def scenario() -> tuple:
        host = EmbeddedControllerHost(tmp_path / "state.db", evaluators={"scripted": evaluator})
        try:
            with pytest.raises(UnsupportedEvaluationError) as refused:
                await host.submit(_pinning(tmp_path))
            rows = tuple(host._connection.execute("SELECT id FROM experiments").fetchall())
            return refused.value, rows
        finally:
            await host.close()

    return asyncio.run(scenario())


# ---- at submission -------------------------------------------------------------------


def test_what_is_recorded_is_the_resolved_spec(tmp_path: Path) -> None:
    submitted = _pinning(tmp_path)
    result, experiment, _ = _drive(tmp_path, submitted, evaluators={"scripted": _Counting})

    recorded = experiment.evaluation
    assert recorded.evaluator.config["pin"] == "pinned-at-submission"
    assert "resolve_to" not in recorded.evaluator.config
    (node,) = result.nodes
    (evaluated,) = node.evaluations
    assert evaluated.result is not None
    fingerprint = evaluated.result.evaluation_fingerprint
    assert fingerprint == recorded.evaluation_fingerprint()
    assert fingerprint != submitted.evaluation.evaluation_fingerprint(), (
        "the fingerprint names what was resolved, not the name that was submitted"
    )


def test_resolution_is_asked_once_per_submission(tmp_path: Path) -> None:
    _Counting.resolved = 0
    _drive(tmp_path, _pinning(tmp_path), evaluators={"scripted": _Counting})
    assert _Counting.resolved == 1


def test_a_spec_that_resolves_into_one_the_evaluator_refuses_is_refused(tmp_path: Path) -> None:
    """The second supports(): accepted as declared, refused as resolved, nothing recorded."""
    refused, experiments = _refused(tmp_path, _ResolvesIntoTheUnsupported)

    assert refused.reasons == ("as resolved: it generates",)
    assert experiments == ()


def test_resolution_cannot_reassign_the_evaluator(tmp_path: Path) -> None:
    refused, experiments = _refused(tmp_path, _Reassigns)

    assert "does not reassign it" in refused.reasons[0]
    assert experiments == ()


def test_an_evaluator_without_resolve_is_recorded_as_declared(tmp_path: Path) -> None:
    """Resolution is optional: an evaluator written before it existed still evaluates."""
    evaluator = _BeforeResolution()
    assert isinstance(evaluator, Evaluator)
    assert not isinstance(evaluator, ResolvableEvaluator)
    submitted = _pinning(tmp_path)

    result, experiment, _ = _drive(tmp_path, submitted, evaluators={"scripted": _BeforeResolution})

    assert result.next_stage == "decision", "trained, evaluated, and waiting to be decided"
    recorded = experiment.evaluation.evaluator.config
    assert recorded["resolve_to"] == "pinned-at-submission", "nothing resolved it"
    assert "pin" not in recorded
    (node,) = result.nodes
    (evaluated,) = node.evaluations
    assert evaluated.result is not None
    assert evaluated.result.evaluation_fingerprint == experiment.evaluation.evaluation_fingerprint()


# ---- after a restart -----------------------------------------------------------------


def test_a_restarted_host_rebuilds_the_request_without_resolving(tmp_path: Path) -> None:
    """The crash leaves the evaluation unsent, so the new host must prepare and issue it.

    It does so from the record: the rebuilt request has the recorded digest --
    the host checks that before issuing -- and resolve() is never asked.
    """
    experiment_id = _crash(tmp_path, "eval-never-sent", _pinning(tmp_path))
    assert _evaluator_workloads(tmp_path) == []

    result, run, attempts, operations = _adopt(
        tmp_path, experiment_id, release=False, evaluators={"scripted": _NeverAgain}
    )

    _decided_once(result, run, attempts)
    (submit,) = operations
    assert _CountingSubmissions.issued == [submit.id], "issued once, under the recorded id"
    assert run.spec.evaluator.config["pin"] == "pinned-at-submission"
