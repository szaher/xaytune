"""ADR-024 §P12 criteria 4-8: determinism and replay, provenance, selection, the local
validation run, and compatibility."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, get_args

import pytest

from tests.test_harness_search import fixtures as fx
from tests.test_storage.conftest import make_experiment
from xaytune.core.domain.harness import harness_fingerprint
from xaytune.core.domain.harness_search import (
    FeedbackRef,
    HarnessCandidateProposal,
    HarnessFeedback,
    HarnessNodeView,
    HarnessSearchContext,
    NoChoice,
    ObjectiveView,
    ProviderComparison,
    ProviderMetricComparison,
    SearchChoice,
    SetStoppingLimits,
)
from xaytune.core.domain.planning import ActionProposal, CandidateProposal, Proposal
from xaytune.core.harness_evaluation import ProviderFeedback, RunValue
from xaytune.core.refs import Actor
from xaytune.harness.search import (
    BestFirstSearchProvider,
    LocalSearchReport,
    ReplayMismatchError,
    explain_local_search,
    replay_local_harness_search,
)
from xaytune.storage import write_transaction
from xaytune.storage.control_plane import BranchRefusedError, ControlPlaneRepository
from xaytune.storage.database import connect
from xaytune.storage.migrations import migrate

EVAL = "sha256:" + "a" * 64
OTHER = "sha256:" + "b" * 64


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> LocalSearchReport:
    return fx.search(tmp_path_factory.mktemp("search"))


def _proposals(report: LocalSearchReport) -> list[HarnessCandidateProposal]:
    return [o.proposal for r in report.rounds for _, o in r.plans if o.proposal is not None]


# ---- criterion 7: the local validation run --------------------------------------------------


def test_three_rounds_produce_two_evaluated_verified_proposals(report: LocalSearchReport) -> None:
    assert [r.round for r in report.rounds] == [1, 2, 3]
    assert report.stop_reason == "max-rounds"
    first, second = _proposals(report)
    assert [op.op for op in first.patch.operations] == ["set-system-instructions"]
    assert second.patch.operations == (SetStoppingLimits(max_turns=1),)
    # Each proposal became the next round's candidate, and every candidate was evaluated.
    assert [r.created_by for r in report.rounds] == [
        None,
        first.proposal_fingerprint(),
        second.proposal_fingerprint(),
    ]
    for record in report.rounds:
        assert [r.seed for r in record.replicates] == list(fx.SEEDS)
        assert all(r.evaluation_result_id and r.feedback for r in record.replicates)
    # max_turns = 1 stops before the answer: the provider-safe comparison favors its parent.
    (vs_root, vs_first) = report.rounds[2].comparisons
    assert (vs_root.b, vs_root.relation) == ("node-000", "favors-b")
    assert vs_first.relation == "favors-b"


def test_the_report_explains_every_decision_and_what_it_spent(report: LocalSearchReport) -> None:
    text = explain_local_search(report)
    for expected in (
        "local exploration: no control plane",
        "Round 1: node-000 (root)",
        "search chose node-000: best task_success",
        "verified (§P7",
        "provider-safe comparison node-002 vs node-000: favors-b",
        "providers were shown",
        "spent [attempt] node-000 seed 11 support/refund-001: settled",
        "Stopped: max-rounds",
        "Total spent: 3 rounds, 9 evaluations, 27 attempts",
    ):
        assert expected in text, expected


def test_the_report_is_written_beside_the_search(tmp_path: Path) -> None:
    report = fx.search(tmp_path)
    written = LocalSearchReport.model_validate_json((tmp_path / "search-report.json").read_text())
    assert written == report
    assert written.claim == "local-exploration"


# ---- criterion 4: determinism and restart ---------------------------------------------------


def test_the_same_record_and_bound_specs_give_the_same_proposal(report: LocalSearchReport) -> None:
    context, recorded = report.rounds[1].plans[0]
    harnesses = {r.node_id: r.harness for r in report.rounds}
    known = {node.node_id for node in context.nodes}
    again = fx.planner().plan(
        context, {k: v for k, v in harnesses.items() if k in known}, seed=0, round=2
    )
    assert again.proposal is not None and recorded.proposal is not None
    assert again.proposal.proposal_fingerprint() == recorded.proposal.proposal_fingerprint()
    assert again == recorded


def test_a_rerun_from_the_report_reproduces_every_round(tmp_path: Path) -> None:
    fx.search(tmp_path)
    written = LocalSearchReport.model_validate_json((tmp_path / "search-report.json").read_text())
    assert replay_local_harness_search(written, fx.planner()) == 2


def test_a_differently_bound_provider_or_policy_refuses_to_replay(
    report: LocalSearchReport,
) -> None:
    with pytest.raises(ReplayMismatchError, match="mutation provider is not bound as recorded"):
        replay_local_harness_search(
            report, fx.planner(alternatives=(SetStoppingLimits(max_turns=2),))
        )
    with pytest.raises(ReplayMismatchError, match="policy"):
        replay_local_harness_search(report, fx.planner(fx.policy(max_retries=1)))


def test_a_changed_provider_version_refuses_to_replay(report: LocalSearchReport) -> None:
    class Upgraded(BestFirstSearchProvider):
        descriptor = BestFirstSearchProvider.descriptor.model_copy(
            update={"plugin_version": "1.1.0"}
        )

    with pytest.raises(ReplayMismatchError, match="search provider is not bound as recorded"):
        replay_local_harness_search(report, fx.planner(search=Upgraded()))


def test_an_edited_recorded_context_refuses_to_replay(report: LocalSearchReport) -> None:
    record = report.rounds[0]
    context, outcome = record.plans[0]
    edited = context.model_copy(update={"history": ("sha256:" + "c" * 64,)})
    tampered = report.model_copy(
        update={"rounds": (record.model_copy(update={"plans": ((edited, outcome),)}),)}
    )
    with pytest.raises(ReplayMismatchError, match="the context changed"):
        replay_local_harness_search(tampered, fx.planner())


# ---- criterion 5: provenance ----------------------------------------------------------------


def test_every_proposal_names_both_providers_the_patch_the_base_and_its_feedback(
    report: LocalSearchReport,
) -> None:
    first, second = _proposals(report)
    planner = fx.planner()
    for proposal in (first, second):
        assert proposal.search == planner.search_provenance
        assert proposal.mutation == planner.mutation_provenance
        assert proposal.search.name == "best-first-search"
        assert proposal.mutation.name == "enumerated-mutation"
        assert proposal.parent_ids == ("node-000",)
        assert proposal.patch.base_fingerprint == harness_fingerprint(fx.root_harness())
        assert proposal.provenance.planner_name == "harness-search"
    assert len(first.feedback) == len(fx.SEEDS)
    assert len(second.feedback) == 2 * len(fx.SEEDS), "the search saw both evaluated nodes"
    refs = {(f.evaluation_result_id) for f in second.feedback}
    recorded = {r.evaluation_result_id for record in report.rounds[:2] for r in record.replicates}
    assert refs == recorded


@pytest.mark.parametrize(
    "change",
    [
        lambda p: {"parent_ids": ("node-009",)},
        lambda p: {
            "patch": p.patch.model_copy(update={"operations": (SetStoppingLimits(max_turns=3),)})
        },
        lambda p: {"search": p.search.model_copy(update={"seed": 99})},
        lambda p: {"mutation": p.mutation.model_copy(update={"plugin_version": "9.9.9"})},
        lambda p: {"feedback": p.feedback[:-1]},
        lambda p: {"provenance": p.provenance.model_copy(update={"context_fingerprint": OTHER})},
    ],
    ids=["base", "patch", "search", "mutation", "feedback", "context"],
)
def test_the_proposal_identity_changes_when_any_of_it_does(
    report: LocalSearchReport, change
) -> None:
    proposal = _proposals(report)[1]
    changed = proposal.model_copy(update=change(proposal))
    assert changed.proposal_fingerprint() != proposal.proposal_fingerprint()


# ---- criterion 6: selection -----------------------------------------------------------------


def _node(
    node_id: str, value: float | None, evaluation: str = EVAL, **fields: Any
) -> HarnessNodeView:
    run_values = (
        ()
        if value is None
        else (RunValue(metric="task_success", value=value, sample_count=3, unit="fraction"),)
    )
    feedback = HarnessFeedback(
        ref=FeedbackRef(
            evaluation_fingerprint=evaluation,
            evidence_digest="sha256:" + node_id[-1] * 64,
            evaluation_result_id=f"eval_{node_id}",
        ),
        feedback=ProviderFeedback(
            evaluation_fingerprint=evaluation,
            evidence_digest="sha256:" + node_id[-1] * 64,
            run_values=run_values,
            tasks=(),
        ),
    )
    return HarnessNodeView(
        node_id=node_id,
        harness_fingerprint="sha256:" + node_id[-1] * 64,
        status="evaluated",
        feedback=(feedback,),
        **fields,
    )


def _context(*nodes: HarnessNodeView, **fields: Any) -> HarnessSearchContext:
    fields.setdefault("objectives", fx.OBJECTIVE)
    return HarnessSearchContext(evaluation_fingerprint=EVAL, nodes=nodes, **fields)


def test_best_first_never_compares_across_evaluation_fingerprints() -> None:
    choice = BestFirstSearchProvider().select(
        _context(_node("node-1", 0.5), _node("node-2", 0.9, evaluation=OTHER))
    )
    assert isinstance(choice, SearchChoice) and choice.base_node_id == "node-1"


def test_best_first_breaks_ties_by_node_id_and_never_ranks_an_absent_value() -> None:
    choice = BestFirstSearchProvider().select(
        _context(_node("node-3", 0.5), _node("node-2", 0.5), _node("node-1", None))
    )
    assert isinstance(choice, SearchChoice) and choice.base_node_id == "node-2"
    assert "1 not ranked" in choice.rationale
    nothing = BestFirstSearchProvider().select(_context(_node("node-1", None)))
    assert isinstance(nothing, NoChoice) and nothing.reason == "nothing-comparable"


def _comparison(a: str, b: str, relation: str) -> ProviderComparison:
    return ProviderComparison(
        a=a,
        b=b,
        evaluation_fingerprint=EVAL,
        comparable=relation != "not-comparable",
        metrics=(
            ProviderMetricComparison(
                metric="task_success", direction="maximize", verdict="inconclusive"
            ),
        ),
        relation=relation,
    )


VECTOR = (
    ObjectiveView(metric="task_success", direction="maximize"),
    ObjectiveView(metric="total_tokens", direction="minimize"),
)


@pytest.mark.parametrize(
    ("relations", "front"),
    [
        ([("node-1", "node-2", "favors-b")], ["node-2", "node-3"]),
        ([("node-1", "node-2", "trade-off")], ["node-1", "node-2", "node-3"]),
        ([("node-1", "node-2", "not-comparable")], ["node-1", "node-2", "node-3"]),
        ([("node-1", "node-2", "inconclusive")], ["node-1", "node-2", "node-3"]),
        ([("node-1", "node-2", "withheld-from-provider")], ["node-1", "node-2", "node-3"]),
        ([("node-3", "node-1", "favors-a"), ("node-3", "node-2", "favors-a")], ["node-3"]),
    ],
    ids=["dominated", "trade-off", "not-comparable", "inconclusive", "withheld", "one-front"],
)
def test_pareto_selection_follows_the_v10_relations(relations, front) -> None:
    context = _context(
        _node("node-1", 0.5),
        _node("node-2", 0.6),
        _node("node-3", 0.7),
        objectives=VECTOR,
        comparisons=tuple(_comparison(a, b, r) for a, b, r in relations),
    )
    choice = BestFirstSearchProvider().select(context)
    assert isinstance(choice, SearchChoice)
    assert list(choice.objective_view["front"]) == front
    assert choice.base_node_id == front[0]


def test_an_exhausted_base_is_not_chosen_again() -> None:
    choice = BestFirstSearchProvider().select(
        _context(_node("node-1", 0.9), _node("node-2", 0.5), exhausted=("node-1",))
    )
    assert isinstance(choice, SearchChoice) and choice.base_node_id == "node-2"
    done = BestFirstSearchProvider().select(_context(_node("node-1", 0.9), exhausted=("node-1",)))
    assert isinstance(done, NoChoice) and done.reason == "exhausted"


def test_a_provider_that_chooses_an_exhausted_base_ends_the_search_with_its_report(
    tmp_path: Path,
) -> None:
    """A search provider that ignores ``exhausted`` would be asked forever; it is refused once."""
    from xaytune.harness.search import HarnessSearchError

    class Stubborn(BestFirstSearchProvider):
        def select(self, context: Any) -> SearchChoice:
            return SearchChoice(base_node_id="node-000", rationale="always the root")

    with pytest.raises(HarnessSearchError, match="lists as exhausted"):
        fx.search(
            tmp_path,
            the_planner=fx.planner(search=Stubborn(), alternatives=fx.ALTERNATIVES[:1]),
            the_benchmark=fx.benchmark(seeds=(11,)),
        )
    written = json.loads((tmp_path / "search-report.json").read_text())
    assert written["stop_reason"] == "provider-error"
    (last,) = [r for r in written["rounds"] if r["round"] == 2]
    assert [outcome["exhausted"] for _, outcome in last["plans"]] == ["node-000"]


def test_a_provider_that_raises_ends_the_search_with_its_report(tmp_path: Path) -> None:
    class Broken(BestFirstSearchProvider):
        def select(self, context: Any) -> SearchChoice:
            raise KeyError("provider internals")

    with pytest.raises(KeyError):
        fx.search(tmp_path, the_planner=fx.planner(search=Broken()))
    written = json.loads((tmp_path / "search-report.json").read_text())
    assert written["stop_reason"] == "provider-error"
    assert written["stop_detail"] == "a provider raised KeyError while planning"
    assert len(written["rounds"]) == 1, "the evaluated round is kept"


def test_a_comparison_that_raises_in_round_two_keeps_the_report_and_its_spending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round 1 has nothing to compare; round 2 compares node-001 with node-000, and fails."""
    import xaytune.harness.search as search

    def raising(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("a message that must not reach the report")

    monkeypatch.setattr(search, "provider_comparison", raising)
    with pytest.raises(RuntimeError, match="must not reach") as raised:
        fx.search(tmp_path)
    assert type(raised.value) is RuntimeError, "the original exception, re-raised"
    text = (tmp_path / "search-report.json").read_text()
    written = search.LocalSearchReport.model_validate_json(text)
    assert written.stop_reason == "error"
    assert written.stop_detail == "RuntimeError while comparing candidates in round 2"
    assert [r.round for r in written.rounds] == [1, 2]
    # Both rounds' replicates ran and stay charged: 2 rounds x 3 seeds x 3 tasks.
    assert (written.spent.rounds, written.spent.evaluations, written.spent.attempts) == (2, 6, 18)
    assert written.rounds[0].plans and written.rounds[0].spending
    assert len(written.rounds[1].replicates) == 3 and written.rounds[1].spending
    assert written.rounds[1].plans == ()
    assert "must not reach" not in text


def test_a_planning_context_that_cannot_be_built_ends_the_search_with_its_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import xaytune.harness.search as search

    def raising(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("context internals")

    monkeypatch.setattr(search, "_context", raising)
    with pytest.raises(ValueError, match="context internals"):
        fx.search(tmp_path)
    written = json.loads((tmp_path / "search-report.json").read_text())
    assert written["stop_reason"] == "error"
    assert written["stop_detail"] == "ValueError while building the planning context in round 1"
    assert written["spent"]["attempts"] == 9
    assert "context internals" not in json.dumps(written)


def test_a_task_attempt_that_raises_is_charged_in_full_and_named_in_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import xaytune.harness.search as search

    class Crashing(search.ScriptedHarnessRunner):
        def run(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("a message that must not reach the report")

    monkeypatch.setattr(search, "ScriptedHarnessRunner", Crashing)
    report = fx.search(
        tmp_path, the_planner=fx.planner(fx.policy(envelope=fx.envelope(max_rounds=1)))
    )
    (round_one,) = report.rounds
    assert all(r.failure for r in round_one.replicates), "no task output, so no evidence"
    errors = [e for r in round_one.replicates for e in r.attempt_errors]
    assert len(errors) == 9 and all(e.endswith(": RuntimeError") for e in errors)
    unaccounted = [e for e in round_one.spending if e["status"] == "unaccounted"]
    assert len(unaccounted) == 9
    assert all(e["note"] == "RuntimeError" and e["charged"] == e["reserved"] for e in unaccounted)
    assert report.spent.tokens[fx.AGENT_TOKEN_UNIT] == 9 * fx.ATTEMPT_WORST_CASE
    text = (tmp_path / "search-report.json").read_text()
    assert "must not reach" not in text
    assert "attempt raised support" in search.explain_local_search(report)


# ---- criterion 8: compatibility -------------------------------------------------------------


def test_the_training_proposal_union_and_identity_are_untouched() -> None:
    assert get_args(Proposal) == (CandidateProposal, ActionProposal)
    assert CandidateProposal.model_fields["kind"].default == "candidate"
    assert "harness" not in CandidateProposal.model_fields


def test_branching_a_harness_proposal_is_refused_and_writes_nothing(
    tmp_path: Path, report: LocalSearchReport
) -> None:
    connection = connect(tmp_path / "state.db")
    migrate(connection)
    repo = ControlPlaneRepository(connection)
    experiment = make_experiment()
    with write_transaction(connection):
        repo.aggregates._insert_experiment(experiment)

    def counts() -> dict[str, int]:
        return {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
            for table in ("experiment_nodes", "events", "outbox")
        }

    before = counts()
    with pytest.raises(BranchRefusedError, match="not durable yet"):
        repo.materialize_candidate_proposal(
            experiment.id, _proposals(report)[0], actor=Actor(type="system", id="test")
        )
    assert counts() == before


def test_no_migration_creates_a_harness_table() -> None:
    migrations = Path(__file__).parents[2] / "xaytune" / "storage" / "migrations"
    text = "\n".join(path.read_text().lower() for path in migrations.glob("*.sql"))
    assert "create table harness" not in text and "harness_search" not in text


def test_the_report_carries_no_released_or_restricted_report(report: LocalSearchReport) -> None:
    dumped = json.dumps(report.model_dump(mode="json"))
    assert "xaytune.harness-evaluation-restricted" not in dumped
    assert "judge" not in dumped.lower() or "judge_prompt_tokens" in dumped
