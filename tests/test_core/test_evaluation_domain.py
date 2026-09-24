"""The evaluation contract: identity without seeds, results tied to their run (ADR-015)."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from xaytune.core import (
    Actor,
    ArtifactId,
    EvaluationId,
    EvaluationResult,
    EvaluationRun,
    EvaluationRunId,
    EvaluationSpec,
    EvaluatorDeterminism,
    EvaluatorSpec,
    ExperimentId,
    ExperimentNodeId,
    MetricResult,
)
from xaytune.core.domain.candidate import (
    CandidateSpec,
    DataSpec,
    ModelSpec,
    TrainingKind,
    TrainingSpec,
)
from xaytune.core.domain.experiment import CandidateSpecSnapshot, ExperimentNode
from xaytune.core.refs import ArtifactRef, DatasetRef, ModelRef
from xaytune.core.state.status import EvaluationRunStatus, ExperimentNodeStatus


def _spec(**overrides: object) -> EvaluationSpec:
    fields: dict[str, object] = {
        "evaluators": (EvaluatorSpec(name="exact-match", version="1.0.0", config={"k": 1}),),
        "dataset": DatasetRef(uri="./data/held-out.jsonl"),
        "slices": ("all",),
    }
    fields.update(overrides)
    return EvaluationSpec(**fields)  # type: ignore[arg-type]


def _subject() -> ArtifactRef:
    return ArtifactRef(id=ArtifactId.generate(), kind="model", uri="/models/m", digest="sha256:m")


def _run(spec: EvaluationSpec, *, seed: int | None = 7, replicate: int = 1) -> EvaluationRun:
    return EvaluationRun(
        id=EvaluationRunId.generate(),
        experiment_id=ExperimentId.generate(),
        node_id=ExperimentNodeId.generate(),
        evaluation_cycle=1,
        spec=spec,
        subject=_subject(),
        evaluation_fingerprint=spec.evaluation_fingerprint(),
        seed=seed,
        replicate=replicate,
    )


def _metric(**overrides: object) -> MetricResult:
    fields: dict[str, object] = {"name": "accuracy", "value": 0.8, "evaluator_name": "exact-match"}
    fields.update(overrides)
    return MetricResult(**fields)  # type: ignore[arg-type]


# ---- seeds belong to the run, not to the evaluation ---------------------------------


def test_an_evaluation_spec_has_no_seed() -> None:
    with pytest.raises(ValidationError, match="seed"):
        EvaluationSpec(evaluators=(EvaluatorSpec(name="exact-match"),), seed=1)  # type: ignore[call-arg]


def test_two_seeds_are_two_samples_of_one_evaluation() -> None:
    """ADR-015 AC-4b: the seed is not part of the fingerprint."""
    spec = _spec()
    first, second = _run(spec, seed=1, replicate=1), _run(spec, seed=2, replicate=2)

    assert first.evaluation_fingerprint == second.evaluation_fingerprint
    assert (first.seed, first.replicate) != (second.seed, second.replicate)


# ---- what the fingerprint covers --------------------------------------------------


@pytest.mark.parametrize(
    "changed",
    [
        {"evaluators": (EvaluatorSpec(name="exact-match", version="1.0.1", config={"k": 1}),)},
        {"evaluators": (EvaluatorSpec(name="exact-match", version="1.0.0", config={"k": 5}),)},
        {"evaluators": (EvaluatorSpec(name="bleu", version="1.0.0", config={"k": 1}),)},
        {"dataset": DatasetRef(uri="./data/other.jsonl")},
        {"slices": ("hard",)},
    ],
    ids=["evaluator-version", "evaluator-config", "evaluator", "dataset", "slices"],
)
def test_what_is_measured_changes_the_fingerprint(changed: dict) -> None:
    assert _spec(**changed).evaluation_fingerprint() != _spec().evaluation_fingerprint()


@pytest.mark.parametrize(
    "changed",
    [
        {"metadata": {"note": "rerun"}},
        {
            "evaluators": (
                EvaluatorSpec(
                    name="exact-match",
                    version="1.0.0",
                    config={"k": 1},
                    determinism=EvaluatorDeterminism.STOCHASTIC,
                ),
            )
        },
    ],
    ids=["metadata", "determinism"],
)
def test_what_describes_it_does_not(changed: dict) -> None:
    assert _spec(**changed).evaluation_fingerprint() == _spec().evaluation_fingerprint()


def test_a_run_cannot_carry_a_fingerprint_its_spec_does_not_have() -> None:
    with pytest.raises(ValidationError, match="does not describe"):
        EvaluationRun(
            id=EvaluationRunId.generate(),
            experiment_id=ExperimentId.generate(),
            node_id=ExperimentNodeId.generate(),
            evaluation_cycle=1,
            spec=_spec(),
            subject=_subject(),
            evaluation_fingerprint=_spec(slices=("hard",)).evaluation_fingerprint(),
        )


# ---- results are attributable ------------------------------------------------------


def test_a_result_names_the_run_that_produced_it() -> None:
    """ADR-015 AC-4c: without the run, a sample cannot be traced to its execution."""
    run = _run(_spec())
    fields = {
        "id": EvaluationId.generate(),
        "node_id": run.node_id,
        "subject": run.subject,
        "evaluation_fingerprint": run.evaluation_fingerprint,
        "metrics": (_metric(),),
    }
    with pytest.raises(ValidationError, match="evaluation_run_id"):
        EvaluationResult(**fields)  # type: ignore[arg-type]

    result = EvaluationResult(evaluation_run_id=run.id, **fields)  # type: ignore[arg-type]
    assert result.evaluation_run_id == run.id


def test_a_result_measures_something() -> None:
    run = _run(_spec())
    with pytest.raises(ValidationError, match="metrics"):
        EvaluationResult(
            id=EvaluationId.generate(),
            evaluation_run_id=run.id,
            node_id=run.node_id,
            subject=run.subject,
            evaluation_fingerprint=run.evaluation_fingerprint,
            metrics=(),
        )


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_a_metric_is_finite(value: float) -> None:
    with pytest.raises(ValidationError):
        _metric(value=value)


def test_a_confidence_interval_is_ordered() -> None:
    with pytest.raises(ValidationError, match="lower bound"):
        _metric(confidence_interval=(0.9, 0.1))


# ---- lifecycle -----------------------------------------------------------------------


def test_an_evaluation_run_follows_its_own_machine() -> None:
    run = _run(_spec()).with_status(EvaluationRunStatus.ACTIVE)
    assert run.with_status(EvaluationRunStatus.SUCCEEDED).is_terminal


def _node() -> ExperimentNode:
    return ExperimentNode(
        id=ExperimentNodeId.generate(),
        experiment_id=ExperimentId.generate(),
        candidate=CandidateSpecSnapshot(
            candidate=CandidateSpec(
                model=ModelSpec(model=ModelRef(uri="m")),
                data=DataSpec(dataset=DatasetRef(uri="d")),
                training=TrainingSpec(kind=TrainingKind.SFT),
            )
        ),
        candidate_fingerprint=CandidateSpec(
            model=ModelSpec(model=ModelRef(uri="m")),
            data=DataSpec(dataset=DatasetRef(uri="d")),
            training=TrainingSpec(kind=TrainingKind.SFT),
        ).candidate_fingerprint(),
        created_by=Actor(type="system", id="test"),
        status=ExperimentNodeStatus.ACTIVE,
    )


def test_each_entry_into_evaluating_starts_a_new_cycle() -> None:
    """ADR-015 §5: a second round of evaluation is not the first one again."""
    node = _node()
    assert node.evaluation_cycle == 0

    first = node.with_status(ExperimentNodeStatus.EVALUATING)
    assert first.evaluation_cycle == 1
    deciding = first.with_status(ExperimentNodeStatus.DECIDING)
    assert deciding.evaluation_cycle == 1, "only entering EVALUATING advances it"

    second = deciding.with_status(ExperimentNodeStatus.ACTIVE).with_status(
        ExperimentNodeStatus.EVALUATING
    )
    assert second.evaluation_cycle == 2
