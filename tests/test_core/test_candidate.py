"""Candidate identity: what belongs to a hypothesis, and what does not.

Every test here pins an ADR-011 ownership decision. They are written as
"changing X does / does not change the fingerprint" because that is the form
the decisions actually take.
"""

from __future__ import annotations

import pytest

from xaytune.core.domain.candidate import (
    AdapterSpec,
    AlgorithmSpec,
    CandidateSpec,
    CheckpointIntent,
    DataSpec,
    EnvironmentSpec,
    LRScheduleSpec,
    ModelSpec,
    OptimizationSpec,
    OptimizerSpec,
    PrecisionSpec,
    RewardSpec,
    ScheduledIntervention,
    TrainingKind,
    TrainingSchedule,
    TrainingSpec,
    candidate_identity_v1,
    candidate_identity_v2,
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


URI = "Qwen/Qwen3-8B"
DATA = "./support-v4.jsonl"


def _plain_candidate() -> CandidateSpec:
    """The candidate whose fingerprint is pinned literally below."""
    return CandidateSpec(
        model=ModelSpec(model=ModelRef(uri="Qwen/Qwen3-8B")),
        data=DataSpec(dataset=DatasetRef(uri="./support-v4.jsonl")),
        training=TrainingSpec(kind=TrainingKind.SFT),
    )


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
    assert canonical_encode({"a": 1}) == "m:{s:1:a=i:1}"
    assert canonical_encode([1, "a", None, True]) == "l:[i:1,s:1:a,n,b:1]"
    assert fingerprint({"kind": "sft", "lr": 2e-5}) == (
        "sha256:46543e5f2f877d09d43d9add1d067b294b91eed8652872c063b83b39fcb67d74"
    )


def test_a_candidate_fingerprint_is_pinned() -> None:
    """One whole CandidateFingerprint, as a literal.

    This is the key experiment comparison, reuse, execution plans and
    checkpoint compatibility are built on. A test that recomputed both sides
    would pass through any change to it while everything already recorded
    stopped matching, so the brittleness is deliberate.
    """
    assert _plain_candidate().candidate_fingerprint() == (
        "sha256:730b6faed16f611f7be3ed499a7e72e4703883d6cfff772bfbef7d0351b13fe4"
    )


def test_the_v1_fingerprint_is_frozen() -> None:
    """Every fingerprint recorded before v2 must still reproduce.

    This is the old pinned literal, **unchanged**. v2 added identity fields to
    ``DataSpec`` and ``OptimizationSpec``; v1 enumerates its fields explicitly
    and so cannot see them. If this ever fails, a v1 helper was edited rather
    than forked -- and since v2 reuses the unchanged v1 helpers, that edit
    would have moved both versions at once, silently.
    """
    assert fingerprint(candidate_identity_v1(_plain_candidate())) == (
        "sha256:3c174a88119ffa7ed47693e5854398f10277ec46c2f972de6adfbfc5a73ac4a9"
    )


def test_v1_and_v2_cannot_collide() -> None:
    """``version`` separates the domains, even when v2's new fields are unset."""
    plain = _plain_candidate()
    assert plain.data.format is None and plain.training.optimization.max_grad_norm is None

    assert fingerprint(candidate_identity_v1(plain)) != fingerprint(candidate_identity_v2(plain))


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


def test_the_two_run_identities_are_different_questions() -> None:
    """Domain-separated, so they cannot collide even on identical contents."""
    run = _run(_candidate(), seed=1)
    applications = [_application(14250)]

    assert run_history_fingerprint(run, applications) != artifact_lineage_fingerprint(
        run, applications, ["ckpt-1"]
    )


def test_rolled_back_work_changes_history_but_not_artifact_lineage() -> None:
    """The reason there are two fingerprints at all (ADR-011).

    Two genuinely different runs:

    ```text
    A: applied at 9000, rolled back, applied again at 14250
    B: applied once at 14250
    ```

    Different histories -- A did something B did not. The same artifact
    lineage, because A's rolled-back work never reached the artifact. Asking
    "has this trajectory been run?" must answer yes.
    """
    candidate = _candidate()
    run_a = _run(candidate, seed=1)
    run_b = _run(candidate, seed=1, id=run_a.id)

    a_history = [_application(9000), _application(14250)]
    b_history = [_application(14250)]
    retained = [_application(14250)]
    ancestry = ["ckpt-2"]

    assert run_history_fingerprint(run_a, a_history) != run_history_fingerprint(run_b, b_history)
    assert artifact_lineage_fingerprint(run_a, retained, ancestry) == artifact_lineage_fingerprint(
        run_b, retained, ancestry
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


# ---- identity is versioned, not whatever the schema happens to be --------


@pytest.mark.parametrize("projection", [candidate_identity_v1, candidate_identity_v2])
def test_adding_a_field_later_does_not_change_existing_identity(projection) -> None:
    """The reason identity hashes a projection rather than the model.

    Hashing ``model_dump()`` makes the current schema define identity: add a
    field with a default six months from now and every candidate already on
    disk fingerprints differently, though nobody changed anything -- and
    reuse, comparison and lineage all silently stop matching.

    Checked for each projection, comparing like with like. It previously
    compared ``candidate_fingerprint()`` with a v1 projection, which quietly
    assumed the default *was* v1 and broke the moment it stopped being.
    """
    candidate = _plain_candidate()
    before = fingerprint(projection(candidate))

    class FutureTrainingSpec(TrainingSpec):
        gradient_checkpointing: bool | None = None

    class FutureCandidateSpec(CandidateSpec):
        training: FutureTrainingSpec  # type: ignore[assignment]
        owner: str | None = None

    evolved = FutureCandidateSpec.model_validate(candidate.model_dump(mode="python"))

    assert fingerprint(projection(evolved)) == before


def test_metadata_is_not_scientific_identity() -> None:
    """A ticket number is not a hypothesis.

    Anything that genuinely bears identity gets a typed field; that is what
    ``AlgorithmSpec.params`` and its siblings are for.
    """
    plain = _plain_candidate()
    annotated = CandidateSpec(
        model=plain.model,
        data=plain.data,
        training=plain.training,
        metadata={"ticket": "RHOAI-1234", "owner": "someone"},
    )

    assert annotated.candidate_fingerprint() == plain.candidate_fingerprint()


def test_identity_is_domain_separated() -> None:
    """A candidate and a run history cannot collide, whatever they contain."""
    projection = candidate_identity_v1(_plain_candidate())

    assert projection["type"] == "candidate"
    assert projection["version"] == 1


@pytest.mark.parametrize(
    ("component", "value"),
    [
        ("algorithm", AlgorithmSpec(name="dpo", params={"beta": 0.1})),
        ("precision", PrecisionSpec(dtype="bf16")),
        ("checkpoint", CheckpointIntent(every_optimizer_steps=500)),
    ],
)
def test_the_newly_typed_components_bear_identity(component: str, value: object) -> None:
    """Each was previously expressible only by hiding it in metadata.

    ``AdamW`` versus ``SGD``, ``bf16`` versus ``fp32``, DPO's beta -- all
    change what the experiment is, so all must change its fingerprint.
    """
    plain = _plain_candidate()
    changed = CandidateSpec(
        model=plain.model,
        data=plain.data,
        training=TrainingSpec(kind=TrainingKind.SFT, **{component: value}),
    )

    assert changed.candidate_fingerprint() != plain.candidate_fingerprint()


def test_the_optimizer_and_schedule_are_distinguishable() -> None:
    """There was no typed way to tell these apart before."""
    plain = _plain_candidate()

    def with_optimization(optimization: OptimizationSpec) -> str:
        return CandidateSpec(
            model=plain.model,
            data=plain.data,
            training=TrainingSpec(kind=TrainingKind.SFT, optimization=optimization),
        ).candidate_fingerprint()

    adamw = with_optimization(
        OptimizationSpec(optimizer=OptimizerSpec(name="adamw"), learning_rate=2e-5)
    )
    sgd = with_optimization(
        OptimizationSpec(optimizer=OptimizerSpec(name="sgd"), learning_rate=2e-5)
    )
    cosine = with_optimization(
        OptimizationSpec(
            optimizer=OptimizerSpec(name="adamw"),
            learning_rate=2e-5,
            lr_schedule=LRScheduleSpec(name="cosine", warmup_steps=100),
        )
    )

    assert len({adamw, sgd, cosine}) == 3


# ---- the projection is enumerated all the way down ----------------------


@pytest.mark.parametrize(
    ("label", "build"),
    [
        (
            "ModelRef",
            lambda: {"model": ModelSpec(model=ModelRef(uri=URI, metadata={"n": "x"}))},
        ),
        (
            "ModelSpec",
            lambda: {"model": ModelSpec(model=ModelRef(uri=URI), metadata={"n": "x"})},
        ),
        (
            "DatasetRef",
            lambda: {"data": DataSpec(dataset=DatasetRef(uri=DATA, metadata={"n": "x"}))},
        ),
        (
            "DataSpec",
            lambda: {"data": DataSpec(dataset=DatasetRef(uri=DATA), metadata={"n": "x"})},
        ),
    ],
)
def test_nested_metadata_is_not_identity(label: str, build) -> None:
    """The metadata rule holds at every level, not only the top one.

    An earlier projection named the top-level fields and dumped the nested
    models, so ``ModelRef.metadata`` still changed identity while
    ``CandidateSpec.metadata`` did not -- and the test only checked the top.
    """
    plain = _plain_candidate()
    annotated = CandidateSpec(
        **{"model": plain.model, "data": plain.data, "training": plain.training, **build()}
    )

    assert annotated.candidate_fingerprint() == plain.candidate_fingerprint(), label


def test_nested_metadata_on_optional_components_is_not_identity() -> None:
    plain = _plain_candidate()

    def with_reward(**kwargs) -> str:
        return CandidateSpec(
            model=plain.model,
            data=plain.data,
            training=plain.training,
            reward=RewardSpec(graders=("task-success",), **kwargs),
        ).candidate_fingerprint()

    assert with_reward() == with_reward(metadata={"owner": "someone"})


@pytest.mark.parametrize("projection", [candidate_identity_v1, candidate_identity_v2])
def test_a_new_field_on_a_nested_model_does_not_change_identity(projection) -> None:
    """The guarantee a projection's version makes, checked at depth.

    The previous evolution test added fields only to the two models the
    projection named, so it confirmed the implementation rather than the
    claim.
    """
    plain = _plain_candidate()
    before = fingerprint(projection(plain))

    class FutureModelRef(ModelRef):
        future: str | None = None

    class FutureModelSpec(ModelSpec):
        model: FutureModelRef  # type: ignore[assignment]

    class FutureDatasetRef(DatasetRef):
        future: int | None = None

    class FutureDataSpec(DataSpec):
        dataset: FutureDatasetRef  # type: ignore[assignment]

    class FutureOptimization(OptimizationSpec):
        future: bool | None = None

    class FutureTrainingSpec(TrainingSpec):
        optimization: FutureOptimization  # type: ignore[assignment]

    class FutureCandidate(CandidateSpec):
        model: FutureModelSpec  # type: ignore[assignment]
        data: FutureDataSpec  # type: ignore[assignment]
        training: FutureTrainingSpec  # type: ignore[assignment]

    evolved = FutureCandidate.model_validate(plain.model_dump(mode="python"))

    assert fingerprint(projection(evolved)) == before


@pytest.mark.parametrize(
    ("label", "changed"),
    [
        ("dataset revision", {"data": DataSpec(dataset=DatasetRef(uri=DATA, revision="v2"))}),
        (
            "tokenizer",
            {"data": DataSpec(dataset=DatasetRef(uri=DATA, tokenizer_fingerprint="sha256:t"))},
        ),
        ("model revision", {"model": ModelSpec(model=ModelRef(uri=URI, revision="main"))}),
    ],
)
def test_nested_identity_bearing_fields_still_count(label: str, changed: dict) -> None:
    """Excluding metadata must not have excluded the things that matter.

    A tokenizer change produces different training data from the same source,
    so it is identity even though the URI is unchanged.
    """
    plain = _plain_candidate()
    other = CandidateSpec(
        **{"model": plain.model, "data": plain.data, "training": plain.training, **changed}
    )

    assert other.candidate_fingerprint() != plain.candidate_fingerprint(), label


def test_the_learning_rate_has_one_authority() -> None:
    """Two places to put it means nothing says which a compiler should read."""
    assert "learning_rate" not in OptimizerSpec.model_fields
    assert "learning_rate" in OptimizationSpec.model_fields
