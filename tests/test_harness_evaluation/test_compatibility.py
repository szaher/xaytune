"""ADR-022 V12: H06 moves nothing already recorded, and the decision contracts are untouched.

``PINNED_NATIVE_EVALUATION`` was computed on ``main`` at ``3c8add6``, before
H06, from the spec below.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.test_harness.fixtures import compile_spec
from tests.test_harness.test_compiler import PINNED_EXECUTION_V1
from tests.test_trajectory.test_compatibility import PINNED_H03_REQUEST
from xaytune.core.agent_execution import agent_execution_fingerprint, agent_run_request_digest
from xaytune.core.domain.decision import DecisionContext
from xaytune.core.domain.evaluation import (
    EvaluationAttempt,
    EvaluationResult,
    EvaluationRun,
    EvaluationSpec,
    EvaluatorSpec,
    MetricResult,
    evaluation_identity_v1,
)
from xaytune.core.domain.objective import Objective, ObjectiveMetric
from xaytune.core.ids import (
    EvaluationId,
    EvaluationRunId,
    ExperimentId,
    ExperimentNodeId,
)
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import ArtifactRef, DatasetRef
from xaytune.core.trajectory import TRAJECTORY_EVENT_KINDS
from xaytune.decision import ThresholdDecisionEngine, UndecidableError

PINNED_NATIVE_EVALUATION = "sha256:6ef0946212ec9f2325444c2480d3aea099f6ac3bd099a72c26a7e6250d73b3ed"


def test_the_evaluation_identity_is_unchanged() -> None:
    native = EvaluationSpec(
        evaluator=EvaluatorSpec(
            name="native",
            version="0.1.0",
            config=FrozenDict(
                {
                    "format": "text",
                    "max_seq_length": 128,
                    "batch_size": 2,
                    "metrics": ["loss"],
                    "precision": "fp32",
                }
            ),
        ),
        dataset=DatasetRef(uri="/data/held-out.jsonl", content_digest="sha256:" + "0" * 64),
    )
    assert native.evaluation_fingerprint() == PINNED_NATIVE_EVALUATION
    assert set(evaluation_identity_v1(native)) == {
        "kind", "identity_version", "api_version", "evaluator", "dataset", "slices",
    }  # fmt: skip


def test_the_evaluation_and_decision_records_are_unchanged() -> None:
    assert set(MetricResult.model_fields) == {
        "name", "value", "sample_count", "dataset_ref", "slice", "evaluator_name",
        "evaluator_version", "seed", "confidence_interval", "standard_error", "metadata",
    }  # fmt: skip
    assert set(EvaluationResult.model_fields) == {
        "id", "evaluation_run_id", "node_id", "subject", "evaluation_fingerprint",
        "metrics", "artifacts", "created_at",
    }  # fmt: skip
    assert set(EvaluationRun.model_fields) == {
        "id", "experiment_id", "node_id", "evaluation_cycle", "spec", "subject",
        "evaluation_fingerprint", "seed", "replicate", "status", "created_at", "updated_at",
        "revision",
    }  # fmt: skip
    assert set(EvaluationAttempt.model_fields) == {
        "id", "evaluation_run_id", "attempt_number", "status", "runtime_ref", "started_at",
        "ended_at", "revision",
    }  # fmt: skip
    assert set(Objective.model_fields) == {"primary", "target", "constraints"}
    assert set(DecisionContext.model_fields) >= {"objective", "results"}


def test_an_unbound_spec_keeps_its_h03_request_digest() -> None:
    spec = compile_spec("/runs/r1")
    assert agent_execution_fingerprint(spec) == PINNED_EXECUTION_V1
    assert agent_run_request_digest(spec) == PINNED_H03_REQUEST


def test_the_trajectory_schema_is_unchanged() -> None:
    assert frozenset({
        "run-started", "model-call-started", "model-call-finished", "tool-call-started",
        "tool-call-finished", "context-changed", "error-observed", "run-terminated",
    }) == TRAJECTORY_EVENT_KINDS  # fmt: skip


def test_g16_r_results_of_one_cycle_stay_undecidable() -> None:
    """With R > 1, a harness cycle reports each metric R times: undecidable, by design."""
    node = ExperimentNodeId.generate()
    subject = ArtifactRef(
        id="artifact_" + "0" * 26,  # type: ignore[arg-type]
        kind="agent_run_evidence",
        uri="/runs/run-evidence.json",
        digest="sha256:" + "1" * 64,
    )
    results = tuple(
        EvaluationResult(
            id=EvaluationId.generate(),
            evaluation_run_id=EvaluationRunId.generate(),
            node_id=node,
            subject=subject,
            evaluation_fingerprint="sha256:" + "e" * 64,
            metrics=(
                MetricResult(
                    name="task_success",
                    value=value,
                    evaluator_name="harness-outcome",
                    evaluator_version="0.1.0",
                    seed=seed,
                ),
            ),
        )
        for seed, value in ((1, 0.6), (2, 0.7), (3, 0.8))
    )
    context = DecisionContext(
        experiment_id=ExperimentId.generate(),
        node_id=node,
        evaluation_cycle=1,
        objective=Objective(primary=ObjectiveMetric(name="task_success", direction="maximize")),
        results=results,
    )
    with pytest.raises(UndecidableError):
        ThresholdDecisionEngine().decide(context)


def test_no_migration_is_added() -> None:
    """H06 evaluates locally and records no table: the durable path is ADR-019 M1-M6's.

    Checked by what the migrations create, not by which one is last, so a
    later migration for something else (020's metric points) does not break it.
    """
    migrations = Path(__file__).parents[2] / "xaytune" / "storage" / "migrations"
    tables = {
        name.lower()
        for path in migrations.glob("*.sql")
        for name in re.findall(r"CREATE TABLE\s+(\w+)", path.read_text(), re.IGNORECASE)
    }
    assert not {table for table in tables if "harness" in table}
