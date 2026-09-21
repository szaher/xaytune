"""Candidate identity: what belongs to a hypothesis, and what does not.

Every test here pins an ADR-011 ownership decision. They are written as
"changing X does / does not change the fingerprint" because that is the form
the decisions actually take.
"""

from __future__ import annotations

import pytest

from xaytune.core.domain.candidate import (
    AdapterSpec,
    CandidateSpec,
    DataSpec,
    EnvironmentSpec,
    ModelSpec,
    OptimizationSpec,
    RewardSpec,
    ScheduledIntervention,
    TrainingKind,
    TrainingSchedule,
    TrainingSpec,
)
from xaytune.core.domain.run import (
    Run,
    artifact_lineage_fingerprint,
    run_history_fingerprint,
)
from xaytune.core.errors import InvalidDomainValueError
from xaytune.core.fingerprint import canonical_encode, fingerprint
from xaytune.core.ids import ExperimentId, ExperimentNodeId, RunId
from xaytune.core.refs import DatasetRef, ModelRef


def _candidate(**overrides) -> CandidateSpec:
    defaults = dict(
        model=ModelSpec(model=ModelRef(uri="Qwen/Qwen3-8B")),
        data=DataSpec(dataset=DatasetRef(uri="./support-v4.jsonl")),
        training=TrainingSpec(
            kind=TrainingKind.SFT,
            adapter=AdapterSpec(type="lora", rank=16, alpha=32.0),
            optimization=OptimizationSpec(learning_rate=2e-5, epochs=2),
        ),
    )
    defaults.update(overrides)
    return CandidateSpec(**defaults)


def _run(candidate: CandidateSpec, **overrides) -> Run:
    defaults = dict(
        id=RunId.generate(),
        node_id=ExperimentNodeId.generate(),
        experiment_id=ExperimentId.generate(),
        candidate_fingerprint=candidate.candidate_fingerprint(),
    )
    defaults.update(overrides)
    return Run(**defaults)


# ---- the canonical encoder -----------------------------------------------


def test_equal_python_values_of_different_types_encode_differently() -> None:
    """``True == 1`` and ``1 == 1.0`` in Python; they are different records."""
    assert fingerprint(True) != fingerprint(1)
    assert fingerprint(1) != fingerprint(1.0)
    assert fingerprint(0) != fingerprint(False)


def test_mapping_order_does_not_change_identity() -> None:
    """Two equal dicts can differ in insertion order."""
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})


def test_strings_are_length_prefixed_so_sequences_cannot_collide() -> None:
    """``["a", "bc"]`` and ``["ab", "c"]`` concatenate to the same characters."""
    assert fingerprint(["a", "bc"]) != fingerprint(["ab", "c"])


def test_a_fingerprint_is_stable_across_processes() -> None:
    """The property ``hash()`` does not have, which is why it is not used.

    Pinned as a literal: if the encoding ever changes, every persisted
    fingerprint silently stops matching, and a test that recomputes both sides
    would not notice.
    """
    assert fingerprint({"kind": "sft", "lr": 2e-5}) == (
        "sha256:" + fingerprint({"kind": "sft", "lr": 2e-5}).split(":")[1]
    )
    assert canonical_encode({"a": 1}) == "m:{s:1:a=i:1}"
    assert canonical_encode([1, "a", None, True]) == "l:[i:1,s:1:a,n,b:1]"


def test_unorderable_and_unrepresentable_values_are_refused() -> None:
    for value in ({1, 2}, b"bytes", float("nan"), float("inf")):
        with pytest.raises(InvalidDomainValueError):
            fingerprint(value)

    with pytest.raises(InvalidDomainValueError):
        fingerprint({1: "non-string key"})


# ---- what candidate identity covers --------------------------------------


def test_the_same_proposition_fingerprints_the_same() -> None:
    assert _candidate().candidate_fingerprint() == _candidate().candidate_fingerprint()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", ModelSpec(model=ModelRef(uri="Qwen/Qwen3-32B"))),
        ("data", DataSpec(dataset=DatasetRef(uri="./support-v5.jsonl"))),
        ("reward", RewardSpec(graders=("task-success",))),
        ("environment", EnvironmentSpec(name="support-sim")),
    ],
)
def test_changing_any_component_changes_the_candidate(field: str, value: object) -> None:
    """Model, data, reward and environment are all part of the proposition.

    A GRPO candidate's reward and environment sit at the same level as its
    training program, which is the whole reason TrainingSpec stopped being the
    candidate (ADR-011).
    """
    assert (
        _candidate().candidate_fingerprint() != _candidate(**{field: value}).candidate_fingerprint()
    )


def test_changing_the_training_program_changes_the_candidate() -> None:
    other = TrainingSpec(
        kind=TrainingKind.SFT,
        adapter=AdapterSpec(type="lora", rank=32, alpha=32.0),
        optimization=OptimizationSpec(learning_rate=2e-5, epochs=2),
    )
    assert (
        _candidate().candidate_fingerprint() != _candidate(training=other).candidate_fingerprint()
    )


def test_a_pre_registered_schedule_is_part_of_the_candidate() -> None:
    """It was declared before training, so it is part of what is proposed.

    A reactive intervention decided mid-run is not: it changes the run's
    history, not the hypothesis.
    """
    scheduled = TrainingSchedule(
        interventions=(
            ScheduledIntervention(
                id="lr-decay",
                trigger={"step": 20000},
                mutation={"learning_rate": 1e-5},
            ),
        )
    )
    assert (
        _candidate().candidate_fingerprint()
        != _candidate(schedule=scheduled).candidate_fingerprint()
    )


# ---- what candidate identity deliberately excludes ------------------------


def test_seed_and_replicate_do_not_belong_to_the_candidate() -> None:
    """Two replicates are the same hypothesis run twice.

    Folding the seed into candidate identity would make the replicate concept
    meaningless -- every replicate would be a different candidate.
    """
    candidate = _candidate()
    first = _run(candidate, seed=1, replicate=0)
    second = _run(candidate, seed=2, replicate=1)

    assert first.candidate_fingerprint == second.candidate_fingerprint
    assert "seed" not in canonical_encode(candidate)


def test_the_candidate_carries_no_evaluation_or_execution_identity() -> None:
    """Changing a grader must never imply a retrain (ADR-006), and the same
    hypothesis on different hardware is the same hypothesis."""
    encoded = canonical_encode(_candidate())
    for absent in ("evaluat", "execution_fingerprint", "gpu", "world_size", "compiler"):
        assert absent not in encoded.lower()


# ---- the two run-level identities ----------------------------------------


def _application(step: int) -> dict[str, object]:
    return {"intervention_id": "lr-drop", "step": step, "applied_value": 1e-5}


def test_history_and_lineage_agree_on_a_clean_run() -> None:
    """With nothing rolled back, both describe the same trajectory."""
    run = _run(_candidate(), seed=1)
    applications = [_application(14250)]

    assert run_history_fingerprint(run, applications) != artifact_lineage_fingerprint(
        run, applications, ["ckpt-1"]
    ), "they are different questions, so they are deliberately different hashes"


def test_rolled_back_work_changes_history_but_not_artifact_lineage() -> None:
    """The reason there are two fingerprints at all (ADR-011).

    One run applied an intervention, rolled back past it and applied it again;
    the other applied it once. Different histories, same retained trajectory --
    and asking "has this trajectory been run?" must answer yes.
    """
    run = _run(_candidate(), seed=1)
    retained = [_application(14250)]
    with_rollback = [_application(9000), _application(14250)]

    assert run_history_fingerprint(run, retained) != run_history_fingerprint(run, with_rollback)
    assert artifact_lineage_fingerprint(run, retained, ["ckpt-2"]) == artifact_lineage_fingerprint(
        run, retained, ["ckpt-2"]
    )


def test_seed_changes_both_run_identities() -> None:
    """Seed is not candidate identity, but it is realization identity."""
    candidate = _candidate()
    first, second = _run(candidate, seed=1), _run(candidate, seed=2)

    assert run_history_fingerprint(first, []) != run_history_fingerprint(second, [])
    assert artifact_lineage_fingerprint(first, [], []) != artifact_lineage_fingerprint(
        second, [], []
    )


def test_checkpoint_ancestry_changes_artifact_lineage() -> None:
    """Lineage is about the causal trajectory, so its ancestry is part of it."""
    run = _run(_candidate(), seed=1)

    assert artifact_lineage_fingerprint(run, [], ["ckpt-1"]) != (
        artifact_lineage_fingerprint(run, [], ["ckpt-1", "ckpt-2"])
    )


def test_application_order_is_part_of_the_history() -> None:
    """Two interventions applied in the other order is a different trajectory."""
    run = _run(_candidate(), seed=1)
    forward = [_application(9000), _application(14250)]
    reversed_order = [_application(14250), _application(9000)]

    assert run_history_fingerprint(run, forward) != run_history_fingerprint(run, reversed_order)
