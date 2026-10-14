"""``/compare`` separates lineage from measurement comparability (ADR-023 §S6).

Two candidates can be distinct in the graph and still have measurements that
cannot be compared: same metric name, different evaluation. Those pairs are
marked not comparable, with a reason, and carry no difference.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.test_api.conftest import Recorded
from tests.test_api.test_real_data import _get, _parent_child
from xaytune.api.v1.models import MetricOut, MetricSide
from xaytune.api.v1.routes import FINGERPRINTS_DIFFER, ONE_SIDED, UNITS_DIFFER, _pair
from xaytune.core.immutable import FrozenDict
from xaytune.storage.repository import AggregateStore

OTHER_EVALUATION = "sha256:" + "e" * 64


def _restamp(monkeypatch: pytest.MonkeyPatch, node_id: str, **changes: Any) -> None:
    """Read *node_id*'s recorded results back as if measured under another evaluation.

    The real results, with only the named fields changed: the recorded scenario
    evaluates every candidate under the experiment's one evaluation spec, and
    results are immutable in the store, so a second evaluation is substituted
    at read time.
    """
    original = AggregateStore.evaluation_results_for_node

    def results(self: AggregateStore, wanted: str) -> Any:
        found = original(self, wanted)
        if wanted != node_id:
            return found
        restamped = []
        for result in found:
            update = dict(changes)
            if "unit" in update:
                unit = update.pop("unit")
                update["metrics"] = tuple(
                    metric.model_copy(update={"metadata": FrozenDict({"unit": unit})})
                    for metric in result.metrics
                )
            restamped.append(result.model_copy(update=update))
        return tuple(restamped)

    monkeypatch.setattr(AggregateStore, "evaluation_results_for_node", results)


def test_same_metric_under_different_evaluations_is_not_comparable(
    api: TestClient, recorded: Recorded, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, child = _parent_child(api, recorded)
    _restamp(monkeypatch, parent["id"], evaluation_fingerprint=OTHER_EVALUATION)
    comparison = _get(api, "/compare", left=child["id"], right=parent["id"])
    assert comparison["lineage"]["distinct_candidates"] is True  # lineage is unaffected
    (pair,) = comparison["metrics"]
    assert pair["name"] == "task_success"
    assert pair["comparable"] is False
    assert pair["reasons"] == [FINGERPRINTS_DIFFER]
    assert pair["difference"] is None
    assert pair["right"]["evaluation_fingerprint"] == OTHER_EVALUATION
    assert pair["left"]["evaluation_fingerprint"] != OTHER_EVALUATION
    # Both values are still shown, with what each was measured under.
    assert (pair["left"]["metric"]["value"], pair["right"]["metric"]["value"]) == (0.83, 0.79)


def test_same_evaluation_in_different_units_is_not_comparable(
    api: TestClient, recorded: Recorded, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, child = _parent_child(api, recorded)
    _restamp(monkeypatch, parent["id"], unit="percent")
    (pair,) = _get(api, "/compare", left=child["id"], right=parent["id"])["metrics"]
    assert pair["comparable"] is False
    assert pair["reasons"] == [UNITS_DIFFER]
    assert (pair["left"]["unit"], pair["right"]["unit"]) == (None, "percent")
    assert pair["difference"] is None


def _side(value: float, fingerprint: str, result: str, unit: str | None = None) -> MetricSide:
    return MetricSide(
        metric=MetricOut(
            name="acc",
            value=value,
            sample_count=None,
            dataset_ref=None,
            slice=None,
            evaluator_name="native",
            evaluator_version=None,
            seed=None,
            confidence_interval=None,
            standard_error=None,
            metadata={},
        ),
        result_id=result,
        evaluation_fingerprint=fingerprint,
        unit=unit,
    )


def test_a_measurement_on_one_side_only_has_no_difference() -> None:
    from datetime import datetime, timezone

    moment = datetime(2026, 10, 14, tzinfo=timezone.utc)
    pair = _pair(("native", "acc", None), [(moment, _side(0.5, "sha256:a", "r1"))], [])
    assert (pair.comparable, pair.reasons, pair.difference, pair.right) == (
        False,
        [ONE_SIDED],
        None,
        None,
    )


def test_a_shared_evaluation_is_preferred_over_a_newer_unshared_one() -> None:
    """The latest results differ in evaluation, but an earlier pair shares one: show that."""
    from datetime import datetime, timedelta, timezone

    start = datetime(2026, 10, 14, tzinfo=timezone.utc)
    left = [
        (start, _side(0.6, "sha256:shared", "l1")),
        (start + timedelta(hours=2), _side(0.9, "sha256:left-only", "l2")),
    ]
    right = [(start + timedelta(hours=1), _side(0.5, "sha256:shared", "r1"))]
    pair = _pair(("native", "acc", None), left, right)
    assert pair.comparable is True
    assert (pair.left.result_id, pair.right.result_id) == ("l1", "r1")  # type: ignore[union-attr]
    assert pair.difference == pytest.approx(0.1)
