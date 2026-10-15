"""ADR-024 §P12 criterion 9 (§P12a): the spending envelope is reserved before work starts.

A search stops before it can exceed any declared limit, charges unaccounted
work in full, treats a measurement above its reservation as an error, and
refuses to start when a limit cannot be enforced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.test_harness_evaluation import fixtures as h06
from tests.test_harness_search import fixtures as fx
from xaytune.core.domain.harness_search import (
    Reservation,
    Spend,
    SpendingExceededError,
    SpendingLedger,
    SpendingLimitError,
    SpendingUnenforceableError,
    attempt_reservation,
)
from xaytune.core.immutable import FrozenDict
from xaytune.evaluation.harness import HarnessJudgeEvaluator, bind_evaluation
from xaytune.harness.search import (
    AGENT_TOKEN_UNIT,
    HarnessPlanner,
    HarnessProviderConfigurationError,
    LocalEvaluation,
    LocalSearchReport,
)


def _within_limits(report: LocalSearchReport) -> None:
    assert not report.spent.exceeds(report.limits), report.spent


def _not_started(report: LocalSearchReport) -> dict[str, Any]:
    entries = [e for r in report.rounds for e in r.spending if e["status"] == "not-started"]
    assert len(entries) == 1
    return dict(entries[0])


@pytest.mark.parametrize("limit", [fx.ATTEMPT_WORST_CASE, 1500, 2000])
def test_a_search_stops_before_the_attempt_that_could_cross_the_token_limit(
    tmp_path: Path, limit: int
) -> None:
    """Each attempt reserves its worst case first; the one that might not fit never starts."""
    the_policy = fx.policy(envelope=fx.envelope(token_limits={AGENT_TOKEN_UNIT: limit}))
    report = fx.search(tmp_path, the_planner=fx.planner(the_policy))
    assert report.stop_reason == "spending-limit"
    entry = _not_started(report)
    assert entry["unit"] == "attempt"
    assert list(entry["limits"]) == [f"tokens[{AGENT_TOKEN_UNIT}]"]
    spent = report.spent.tokens[AGENT_TOKEN_UNIT]
    assert spent <= limit < spent + fx.ATTEMPT_WORST_CASE
    _within_limits(report)
    if limit == fx.ATTEMPT_WORST_CASE:
        assert report.spent.attempts == 1, "exactly one worst case fits"


@pytest.mark.parametrize("k", [1, 4, 7])
def test_exactly_k_worst_case_attempts_fit_k_reservations(k: int) -> None:
    """A model that spends its whole reservation every time: k attempts, never k + 1."""
    ledger = _ledger(token_limits={AGENT_TOKEN_UNIT: k * fx.ATTEMPT_WORST_CASE})
    started = 0
    with pytest.raises(SpendingLimitError):
        while True:
            reservation = ledger.reserve(_attempt())
            ledger.settle(reservation, reservation.worst_case)
            started += 1
    assert started == k
    assert ledger.spent.tokens[AGENT_TOKEN_UNIT] == k * fx.ATTEMPT_WORST_CASE


def test_a_search_stops_before_an_evaluation_past_its_limit(tmp_path: Path) -> None:
    the_policy = fx.policy(envelope=fx.envelope(max_candidate_evaluations=4))
    report = fx.search(tmp_path, the_planner=fx.planner(the_policy))
    assert (report.stop_reason, report.spent.evaluations) == ("spending-limit", 4)
    assert _not_started(report)["unit"] == "evaluation"
    # Four replicates of three tasks each ran; the fifth started none.
    assert report.spent.attempts == 4 * 3
    _within_limits(report)


def test_no_task_runs_for_a_replicate_whose_evaluation_cannot_start(tmp_path: Path) -> None:
    """Admission before execution: the second replicate's evaluation is refused before
    any of its task attempts is reserved, so nothing is spent on it."""
    the_policy = fx.policy(envelope=fx.envelope(max_candidate_evaluations=1))
    report = fx.search(tmp_path, the_planner=fx.planner(the_policy))
    assert (report.stop_reason, report.spent.evaluations, report.spent.attempts) == (
        "spending-limit",
        1,
        3,
    )
    entries = [e for r in report.rounds for e in r.spending]
    assert _not_started(report)["label"] == "node-000 seed 12"
    assert not [e for e in entries if e["unit"] == "attempt" and "seed 12" in e["label"]]
    assert not (tmp_path / "node-000" / "seed-12").exists(), "not even a task directory"
    (round_one,) = report.rounds
    assert [r.seed for r in round_one.replicates] == [11]


def test_a_search_stops_before_an_attempt_past_the_attempt_or_call_limit(tmp_path: Path) -> None:
    report = fx.search(
        tmp_path / "attempts",
        the_planner=fx.planner(fx.policy(envelope=fx.envelope(max_task_attempts=5))),
    )
    assert (report.stop_reason, report.spent.attempts) == ("spending-limit", 5)
    calls = fx.search(
        tmp_path / "calls",
        the_planner=fx.planner(fx.policy(envelope=fx.envelope(max_model_calls={"agent": 9}))),
    )
    # Each attempt reserves max_turns (4) agent calls; spent calls never exceed 9.
    assert calls.stop_reason == "spending-limit"
    assert calls.spent.model_calls["agent"] <= 9
    _within_limits(calls)


def test_a_search_runs_no_round_past_its_round_limit(tmp_path: Path) -> None:
    report = fx.search(
        tmp_path, the_planner=fx.planner(fx.policy(envelope=fx.envelope(max_rounds=1)))
    )
    assert (report.stop_reason, len(report.rounds), report.spent.rounds) == ("max-rounds", 1, 1)
    assert report.rounds[0].plans == (), "no proposal is planned for a round that cannot run"


def test_judge_calls_are_reserved_and_stop_the_search_at_their_limit(tmp_path: Path) -> None:
    judge = HarnessJudgeEvaluator(h06.source())
    spec = bind_evaluation(judge, h06.evaluation_spec(h06.judge_config()))
    unit = f"provider-tokens({h06.JUDGE.provider}/{h06.JUDGE.name})"
    per_evaluation = 3 * (1000 + 16)  # three tasks, prompt bound + max_output_tokens
    the_policy = fx.policy(
        envelope=fx.envelope(
            max_model_calls={"agent": 1000, "judge": 100},
            token_limits={AGENT_TOKEN_UNIT: 100 * fx.ATTEMPT_WORST_CASE, unit: 2 * per_evaluation},
            judge_prompt_tokens=1000,
        )
    )
    report = fx.search(
        tmp_path,
        the_planner=fx.planner(the_policy),
        evaluation=LocalEvaluation(
            evaluator=judge, spec=spec, judge_model=h06.judge_model(*[{"score": 4}] * 100)
        ),
        objectives=(fx.ObjectiveView(metric="quality", direction="maximize"),),
    )
    assert (report.stop_reason, report.spent.evaluations) == ("spending-limit", 2)
    # Judge spend is not measured in v1: each evaluation is charged its full reservation.
    assert report.spent.tokens[unit] == 2 * per_evaluation
    assert report.spent.model_calls["judge"] == 6
    entry = _not_started(report)
    assert entry["unit"] == "evaluation" and list(entry["limits"]) == [f"tokens[{unit}]"]


def _judge_reporting(input_tokens: int, output_tokens: int) -> Any:
    from xaytune.agent import AgentModelResponse
    from xaytune.agent.model import AgentModelUsage

    reply = AgentModelResponse(
        content=FrozenDict({"score": 4}),
        model=h06.JUDGE.name,
        model_revision=h06.JUDGE.revision,
        finish_reason="stop",
        usage=AgentModelUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )
    return h06.judge_model(*[reply] * 100)


def _judge_search(tmp_path: Path, judge_model: Any, *, prompt_bound: int) -> LocalSearchReport:
    judge = HarnessJudgeEvaluator(h06.source())
    spec = bind_evaluation(judge, h06.evaluation_spec(h06.judge_config()))
    unit = f"provider-tokens({h06.JUDGE.provider}/{h06.JUDGE.name})"
    the_policy = fx.policy(
        envelope=fx.envelope(
            max_rounds=1,
            max_model_calls={"agent": 1000, "judge": 100},
            token_limits={AGENT_TOKEN_UNIT: 100 * fx.ATTEMPT_WORST_CASE, unit: 100_000},
            judge_prompt_tokens=prompt_bound,
        )
    )
    return fx.search(
        tmp_path,
        the_planner=fx.planner(the_policy),
        evaluation=LocalEvaluation(evaluator=judge, spec=spec, judge_model=judge_model),
        objectives=(fx.ObjectiveView(metric="quality", direction="maximize"),),
    )


def test_a_judge_reporting_its_tokens_is_charged_what_it_reported(tmp_path: Path) -> None:
    report = _judge_search(tmp_path, _judge_reporting(40, 2), prompt_bound=1000)
    unit = f"provider-tokens({h06.JUDGE.provider}/{h06.JUDGE.name})"
    # One round: three seeds, three judged tasks each, 42 reported tokens per call.
    assert report.spent.model_calls["judge"] == 9
    assert report.spent.tokens[unit] == 9 * 42
    statuses = {e["status"] for r in report.rounds for e in r.spending if e["unit"] == "evaluation"}
    assert statuses == {"settled"}


def test_a_judge_prompt_above_its_declared_bound_stops_the_search(tmp_path: Path) -> None:
    """The declared prompt bound is checked against what the judge's provider reported."""
    with pytest.raises(SpendingExceededError, match="the bound is wrong"):
        _judge_search(tmp_path, _judge_reporting(2000, 2), prompt_bound=1000)
    written = LocalSearchReport.model_validate_json((tmp_path / "search-report.json").read_text())
    assert written.stop_reason == "spending-exceeded"
    (exceeded,) = [e for r in written.rounds for e in r.spending if e["status"] == "exceeded"]
    assert exceeded["unit"] == "evaluation"


def test_an_evaluator_that_raises_is_charged_in_full_and_the_report_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import xaytune.harness.search as search

    def raising(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("a message that must not reach the report")

    monkeypatch.setattr(search, "run_harness_evaluation", raising)
    with pytest.raises(RuntimeError):
        fx.search(tmp_path)
    text = (tmp_path / "search-report.json").read_text()
    written = LocalSearchReport.model_validate_json(text)
    assert written.stop_reason == "error"
    assert written.stop_detail == "RuntimeError during evaluation node-000 seed 11"
    (round_one,) = written.rounds
    (abandoned,) = [e for e in round_one.spending if e["unit"] == "evaluation"]
    assert (abandoned["status"], abandoned["note"]) == ("unaccounted", "RuntimeError")
    assert abandoned["charged"] == abandoned["reserved"]
    assert written.spent.evaluations == 1, "the interrupted evaluation is charged"
    assert written.spent.attempts == 3, "and the attempts it followed are kept"
    assert "must not reach" not in text


def test_a_judge_that_raises_mid_evaluation_keeps_its_whole_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import xaytune.harness.search as search

    def raising(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("judge adapter crashed")

    monkeypatch.setattr(search, "run_harness_evaluation", raising)
    with pytest.raises(RuntimeError):
        _judge_search(tmp_path, _judge_reporting(40, 2), prompt_bound=1000)
    written = LocalSearchReport.model_validate_json((tmp_path / "search-report.json").read_text())
    unit = f"provider-tokens({h06.JUDGE.provider}/{h06.JUDGE.name})"
    assert written.spent.tokens[unit] == 3 * (1000 + 16), "three tasks, each at its bound"
    assert written.spent.model_calls["judge"] == 3


# ---- refusing to start -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("envelope", "reason"),
    [
        ({"token_limits": FrozenDict()}, "no token limit for"),
        ({"cost": FrozenDict({"usd": 10})}, "cost-not-metered"),
        ({"max_model_calls": FrozenDict()}, "no agent model call limit"),
    ],
    ids=["no-token-limit", "cost", "no-call-limit"],
)
def test_an_envelope_that_cannot_be_enforced_refuses_before_anything_runs(
    tmp_path: Path, envelope: dict[str, Any], reason: str
) -> None:
    the_policy = fx.policy(envelope=fx.envelope(**envelope))
    with pytest.raises(SpendingUnenforceableError, match=reason):
        fx.search(tmp_path, the_planner=fx.planner(the_policy))
    assert not any(tmp_path.iterdir()), "nothing ran"


def test_a_root_that_leaves_its_attempts_unbounded_refuses_to_start(tmp_path: Path) -> None:
    unbounded = fx.root_harness(stopping__max_total_tokens=None)
    with pytest.raises(SpendingUnenforceableError, match="max_total_tokens is unset"):
        fx.search(tmp_path, root=unbounded)
    over = fx.root_harness(stopping__max_turns=fx.CEILINGS.max_turns + 1)
    with pytest.raises(SpendingUnenforceableError, match="exceeds the experiment's ceiling"):
        fx.search(tmp_path, root=over)


def test_a_judge_without_a_prompt_bound_refuses_to_start(tmp_path: Path) -> None:
    judge = HarnessJudgeEvaluator(h06.source())
    spec = bind_evaluation(judge, h06.evaluation_spec(h06.judge_config()))
    with pytest.raises(SpendingUnenforceableError) as refused:
        fx.search(tmp_path, evaluation=LocalEvaluation(evaluator=judge, spec=spec))
    reasons = " ".join(refused.value.reasons)
    assert "judge_prompt_tokens" in reasons and "no judge model call limit" in reasons


def test_a_provider_that_calls_a_model_cannot_be_bound_in_v1() -> None:
    class Stochastic:
        descriptor = fx.mutation().descriptor
        spec = fx.mutation().spec
        capabilities = fx.mutation().capabilities.model_copy(
            update={"determinism": "stochastic", "model": FrozenDict({"provider": "acme"})}
        )

    with pytest.raises(HarnessProviderConfigurationError, match="stochastic"):
        HarnessPlanner(fx.BestFirstSearchProvider(), Stochastic(), fx.policy())  # type: ignore[arg-type]


# ---- the ledger --------------------------------------------------------------------------------


def _ledger(**limits: Any) -> SpendingLedger:
    return SpendingLedger(fx.envelope(**limits))


def _attempt() -> Reservation:
    return attempt_reservation(fx.root_harness(), AGENT_TOKEN_UNIT, "attempt")


def test_unaccounted_work_keeps_its_whole_reservation_charged() -> None:
    ledger = _ledger()
    reservation = ledger.reserve(_attempt())
    charged = ledger.settle(reservation, None)
    assert charged == reservation.worst_case
    assert ledger.spent.tokens[AGENT_TOKEN_UNIT] == fx.ATTEMPT_WORST_CASE
    assert ledger.entries[-1]["status"] == "unaccounted"


def test_retries_are_attempts_and_count() -> None:
    ledger = _ledger(max_task_attempts=2)
    for _ in range(2):
        ledger.settle(ledger.reserve(_attempt()), Spend(attempts=1))
    with pytest.raises(SpendingLimitError, match="attempts"):
        ledger.reserve(_attempt())


def test_a_measurement_above_its_reservation_stops_the_search() -> None:
    ledger = _ledger()
    reservation = ledger.reserve(_attempt())
    too_much = Spend(attempts=1, tokens=FrozenDict({AGENT_TOKEN_UNIT: fx.ATTEMPT_WORST_CASE + 1}))
    with pytest.raises(SpendingExceededError, match="the bound is wrong"):
        ledger.settle(reservation, too_much)
    assert ledger.spent.tokens[AGENT_TOKEN_UNIT] == fx.ATTEMPT_WORST_CASE
    assert ledger.entries[-1]["status"] == "exceeded"


def test_a_unit_in_an_undeclared_unit_or_role_is_never_started() -> None:
    ledger = _ledger()
    provider_call = Reservation(
        unit="provider-call",
        label="mutation model",
        worst_case=Spend(
            model_calls=FrozenDict({"provider": 1}),
            tokens=FrozenDict({"provider-tokens(acme/m)": 10}),
        ),
    )
    with pytest.raises(SpendingLimitError) as refused:
        ledger.reserve(provider_call)
    assert set(refused.value.limits) == {
        "model_calls[provider] (no limit declared)",
        "tokens[provider-tokens(acme/m)] (no limit declared)",
    }
    allowed = _ledger(
        max_model_calls={"agent": 10, "provider": 1},
        token_limits={AGENT_TOKEN_UNIT: 1, "provider-tokens(acme/m)": 10},
    )
    allowed.settle(allowed.reserve(provider_call), provider_call.worst_case)
    with pytest.raises(SpendingLimitError, match="provider"):
        allowed.reserve(provider_call)


def test_one_unit_is_open_at_a_time() -> None:
    ledger = _ledger()
    ledger.reserve(_attempt())
    with pytest.raises(RuntimeError, match="one unit of work at a time"):
        ledger.reserve(_attempt())
