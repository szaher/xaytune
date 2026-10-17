"""ADR-024 §P12 criterion 3: providers see provider feedback and nothing else (§P4, §P6, H7-F12).

The rule covers the whole search context -- values, rankings, comparison
verdicts and rationales -- not just mutation requests, and holds under both
disclosure modes: a single-task score and a ``successful-tasks`` slice never
reach a provider in any form.
"""

from __future__ import annotations

import json
import typing
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from tests.test_harness.fixtures import SCRIPT, RecordingTools
from tests.test_harness.fixtures import source as content_source
from tests.test_harness_evaluation import fixtures as h06
from tests.test_harness_search import fixtures as fx
from xaytune.core.domain.benchmark import FirstTasks
from xaytune.core.domain.evaluation import EvaluationResult
from xaytune.core.domain.harness import harness_fingerprint
from xaytune.core.domain.harness_search import (
    FeedbackRef,
    HarnessFeedback,
    HarnessNodeView,
    HarnessSearchContext,
    MutationRequest,
    NoChoice,
    ObjectiveView,
)
from xaytune.core.harness_evaluation import (
    HarnessEvaluationReport,
    ProviderFeedback,
    RestrictedEvaluationReport,
)
from xaytune.evaluation.harness import (
    HarnessJudgeEvaluator,
    HarnessOutcomeEvaluator,
    bind_evaluation,
    provider_feedback,
    run_harness_evaluation,
)
from xaytune.harness import ScriptedHarnessModel
from xaytune.harness.search import (
    AGENT_TOKEN_UNIT,
    BestFirstSearchProvider,
    LocalEvaluation,
    run_local_harness_search,
)

FORBIDDEN = {HarnessEvaluationReport, RestrictedEvaluationReport, EvaluationResult}


def _reachable(model: type[BaseModel], seen: set[type] | None = None) -> set[type]:
    """Every model class a field of *model* can hold, transitively."""
    seen = set() if seen is None else seen
    if model in seen:
        return seen
    seen.add(model)
    hints = typing.get_type_hints(model)
    for name in model.model_fields:
        stack = [hints[name]]
        while stack:
            hint = stack.pop()
            stack.extend(typing.get_args(hint))
            if isinstance(hint, type) and issubclass(hint, BaseModel):
                _reachable(hint, seen)
    return seen


@pytest.mark.parametrize("model", [MutationRequest, HarnessSearchContext])
def test_provider_inputs_cannot_carry_a_report_a_result_or_a_score(model) -> None:
    reachable = _reachable(model)
    assert not (reachable & FORBIDDEN)
    assert ProviderFeedback in reachable


def test_the_provider_input_fields_are_pinned() -> None:
    assert set(MutationRequest.model_fields) == {
        "base", "base_fingerprint", "base_node_id", "allowed_operations", "ceilings",
        "feedback", "existing", "refused", "seed", "round", "attempt", "count",
    }  # fmt: skip
    assert set(HarnessSearchContext.model_fields) == {
        "objectives", "scalarization", "evaluation_fingerprint", "nodes", "comparisons",
        "exhausted", "remaining", "history",
    }  # fmt: skip
    assert set(HarnessFeedback.model_fields) == {"ref", "feedback"}
    assert set(HarnessNodeView.model_fields) == {
        "node_id", "harness_fingerprint", "parent_ids", "status", "feedback",
    }  # fmt: skip


class _Spy:
    """Wraps a provider and keeps every input it was given."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.descriptor, self.spec, self.capabilities = (
            inner.descriptor,
            inner.spec,
            inner.capabilities,
        )
        self.inputs: list[Any] = []

    def select(self, context: HarnessSearchContext) -> Any:
        self.inputs.append(context)
        return self.inner.select(context)

    def mutate(self, request: MutationRequest, count: int = 1) -> Any:
        self.inputs.append(request)
        return self.inner.mutate(request, count)


def _spied_search(
    workspace: Path, *, evaluation: LocalEvaluation | None = None, **options: Any
) -> tuple[Any, _Spy, _Spy]:
    search, mutation = _Spy(BestFirstSearchProvider()), _Spy(fx.mutation())
    the_policy = options.pop("the_policy", fx.policy())
    report = fx.search(
        workspace,
        the_planner=fx.planner(the_policy, search=search, mutation=mutation),
        evaluation=evaluation,
        **options,
    )
    return report, search, mutation


def _texts(inputs: list[Any]) -> str:
    return "\n".join(json.dumps(item.model_dump(mode="json"), sort_keys=True) for item in inputs)


def test_a_spy_sees_only_provider_feedback_and_never_a_reference_or_judge_exchange(
    tmp_path: Path,
) -> None:
    judge = HarnessJudgeEvaluator(h06.source())
    spec = bind_evaluation(judge, h06.evaluation_spec(h06.judge_config()))
    unit = f"provider-tokens({h06.JUDGE.provider}/{h06.JUDGE.name})"
    the_policy = fx.policy(
        envelope=fx.envelope(
            max_model_calls={"agent": 1000, "judge": 100},
            token_limits={AGENT_TOKEN_UNIT: 100 * fx.ATTEMPT_WORST_CASE, unit: 100_000},
            judge_prompt_tokens=2000,
        )
    )
    report, search, mutation = _spied_search(
        tmp_path,
        the_policy=the_policy,
        evaluation=LocalEvaluation(
            evaluator=judge, spec=spec, judge_model=h06.judge_model(*[{"score": 4}] * 100)
        ),
        objectives=(ObjectiveView(metric="quality", direction="maximize"),),
    )
    assert search.inputs and mutation.inputs
    for request in (i for i in mutation.inputs if isinstance(i, MutationRequest)):
        assert all(isinstance(f.feedback, ProviderFeedback) for f in request.feedback)
    seen = _texts(search.inputs + mutation.inputs)
    for secret in (*h06.REFERENCES.values(), h06.PROMPT, h06.RUBRIC, "Orders ship within two days"):
        assert secret.strip() not in seen
    assert "quality" in seen, "the judge's aggregate quality does reach providers"
    assert "evaluation_report" not in seen and "restricted" not in seen


@pytest.mark.parametrize("disclosure", ["aggregate", "per-task"])
def test_a_single_task_score_never_reaches_a_provider_in_any_form(
    tmp_path: Path, disclosure: str
) -> None:
    report, search, mutation = _spied_search(
        tmp_path,
        evaluation=fx.outcome(single_task=True, disclosure=disclosure),
        the_benchmark=fx.benchmark(single_task=True),
    )
    contexts = [i for i in search.inputs if isinstance(i, HarnessSearchContext)]
    assert contexts
    for context in contexts:
        for node in context.nodes:
            for feedback in node.feedback:
                # Withheld by the release (aggregate) or by the provider filter (per-task).
                assert "task_success" not in {v.metric for v in feedback.feedback.run_values}
        for comparison in context.comparisons:
            (verdict,) = comparison.metrics
            assert verdict.verdict == "withheld-from-provider"
            assert (
                verdict.mean_difference is None and comparison.relation == "withheld-from-provider"
            )
    # Nothing ranks a node on it: best-first has nothing comparable.
    first = report.rounds[0].plans[0][1]
    assert isinstance(first.choice, NoChoice) and first.choice.reason == "nothing-comparable"
    assert report.stop_reason == "no-proposal"


@pytest.mark.parametrize("disclosure", ["aggregate", "per-task"])
def test_a_successful_tasks_slice_never_reaches_a_provider(tmp_path: Path, disclosure: str) -> None:
    sliced = ObjectiveView(metric="turns", slice="successful-tasks", direction="minimize")
    search, mutation = BestFirstSearchProvider(), fx.mutation()
    spy = _Spy(search)
    report = run_local_harness_search(
        root=fx.root_harness(),
        planner=fx.planner(search=spy, mutation=mutation),
        objectives=(fx.OBJECTIVE[0], sliced),
        benchmark=fx.benchmark(),
        evaluation=fx.outcome(disclosure=disclosure),
        content=content_source(),
        model=lambda harness: ScriptedHarnessModel(SCRIPT),
        toolbox=lambda: RecordingTools().toolbox(),
        recording=h06.recording(),
        workspace=tmp_path,
    )
    contexts = [i for i in spy.inputs if isinstance(i, HarnessSearchContext)]
    assert contexts and report.rounds
    for context in contexts:
        values = [v for n in context.nodes for f in n.feedback for v in f.feedback.run_values]
        assert values and all(v.slice is None for v in values)
        for comparison in context.comparisons:
            verdicts = {(m.metric, m.slice): m for m in comparison.metrics}
            withheld = verdicts[("turns", "successful-tasks")]
            assert withheld.verdict == "withheld-from-provider"
            assert withheld.mean_difference is None
            assert comparison.relation == "withheld-from-provider"
    # A withheld slice is named (so a reviewer sees it was withheld), never valued.
    for context in contexts:
        for node in context.nodes:
            for feedback in node.feedback:
                for name in feedback.feedback.withheld:
                    assert name.endswith("/successful-tasks") or "/" not in name


def test_contexts_differing_only_in_a_withheld_value_are_the_same_context(tmp_path: Path) -> None:
    """The withheld value never enters: same feedback, same fingerprint, same choice."""
    replicate = h06.Replicate(tmp_path / "run", sampling=FirstTasks(count=1))
    evaluator = HarnessOutcomeEvaluator(replicate.source)
    spec = bind_evaluation(
        evaluator,
        h06.evaluation_spec(
            h06.outcome_config(sampling={"kind": "first", "count": 1}), disclosure="per-task"
        ),
    )
    evaluated = run_harness_evaluation(
        evaluator, spec, replicate.subject, source=replicate.source,
        output_uri=str(tmp_path / "eval"), seed=0, replicate=0,
    )  # fmt: skip
    report = evaluated.report
    assert any(v.metric == "task_success" for v in report.run_values)
    changed = report.model_copy(
        update={
            "run_values": tuple(
                v.model_copy(update={"value": 1.0 - v.value}) if v.metric == "task_success" else v
                for v in report.run_values
            )
        }
    )
    assert provider_feedback(report) == provider_feedback(changed)

    def context(rep: HarnessEvaluationReport) -> HarnessSearchContext:
        feedback = HarnessFeedback(
            ref=FeedbackRef(
                evaluation_fingerprint=rep.evaluation_fingerprint,
                evidence_digest=rep.evidence_digest,
                evaluation_result_id=str(evaluated.result.id),
            ),
            feedback=provider_feedback(rep),
        )
        node = HarnessNodeView(
            node_id="node-000",
            harness_fingerprint=harness_fingerprint(fx.root_harness()),
            status="evaluated",
            feedback=(feedback,),
        )
        return HarnessSearchContext(
            objectives=fx.OBJECTIVE,
            evaluation_fingerprint=rep.evaluation_fingerprint,
            nodes=(node,),
        )

    a, b = context(report), context(changed)
    assert a.context_fingerprint() == b.context_fingerprint()
    assert BestFirstSearchProvider().select(a) == BestFirstSearchProvider().select(b)
