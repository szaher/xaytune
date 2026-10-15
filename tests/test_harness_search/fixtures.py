"""A bounded harness search over the pinned support suite (H04) with the scripted model.

The root harness enables ``kb.search`` and defines ``tickets.close`` without
enabling it: the regression case for re-enabling a disabled tool. Its limits
are small, so every attempt's reservation is small and exact.

The scripted model answers every task the same way (a ``kb.search`` call,
then a final answer), so the held-out references decide success: two of the
three tasks succeed. ``max_turns = 1`` stops a run before its answer, so it
scores lower: the search has something real to learn.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tests.test_harness.fixtures import SCRIPT, RecordingTools, harness_record
from tests.test_harness.fixtures import source as content_source
from tests.test_harness_evaluation import fixtures as h06
from xaytune.core.domain.benchmark import AllTasks, FirstTasks
from xaytune.core.domain.harness import AgentHarnessSpec, InlineContent
from xaytune.core.domain.harness_search import (
    HarnessCeilings,
    HarnessSearchPolicy,
    LocalSearchEnvelope,
    ObjectiveView,
    SetStoppingLimits,
    SetSystemInstructions,
)
from xaytune.core.immutable import FrozenDict
from xaytune.evaluation.harness import HarnessOutcomeEvaluator, bind_evaluation
from xaytune.harness import ScriptedHarnessModel
from xaytune.harness.search import (
    AGENT_TOKEN_UNIT,
    BestFirstSearchProvider,
    EnumeratedMutationConfig,
    EnumeratedMutationProvider,
    HarnessPlanner,
    LocalBenchmark,
    LocalEvaluation,
    LocalSearchReport,
    run_local_harness_search,
)

AUTHORIZED = ("kb.search", "tickets.close")
SEEDS = (11, 12, 13)
OBJECTIVE = (ObjectiveView(metric="task_success", direction="maximize"),)
CONCISE = "You are a concise support agent. Answer from the knowledge base and cite it."
ALTERNATIVES = (
    SetSystemInstructions(system=InlineContent(text=CONCISE)),
    SetStoppingLimits(max_turns=1),
)


def root_harness(**limits: Any) -> AgentHarnessSpec:
    record = harness_record()
    record["stopping"].update(max_turns=4, max_total_tokens=400, max_tool_calls=4)
    record["context"].update(max_context_tokens=300, max_tool_result_tokens=100)
    record["model"]["generation"]["max_output_tokens"] = 64
    for path, value in limits.items():
        section, field = path.split("__")
        record[section][field] = value
    return AgentHarnessSpec.model_validate(record)


CEILINGS = HarnessCeilings(
    max_context_tokens=300,
    max_tool_result_tokens=100,
    max_turns=4,
    max_total_tokens=400,
    max_tool_calls=4,
    max_output_tokens=64,
)
ATTEMPT_WORST_CASE = 400 + 300 + 64
"""One attempt's reservation under the ceilings: max_total + max_context + max_output."""


def envelope(**changes: Any) -> LocalSearchEnvelope:
    fields: dict[str, Any] = {
        "max_rounds": 3,
        "max_candidate_evaluations": 20,
        "max_task_attempts": 60,
        "max_model_calls": FrozenDict({"agent": 1000}),
        "token_limits": FrozenDict({AGENT_TOKEN_UNIT: 100 * ATTEMPT_WORST_CASE}),
    }
    fields.update(changes)
    return LocalSearchEnvelope(**fields)


def policy(**changes: Any) -> HarnessSearchPolicy:
    fields: dict[str, Any] = {
        "autonomous_search": "authorized",
        "authorized_tools": AUTHORIZED,
        "allowed_operations": ("set-stopping-limits", "set-system-instructions"),
        "ceilings": CEILINGS,
        "max_retries": 2,
        "envelope": envelope(),
    }
    fields.update(changes)
    return HarnessSearchPolicy(**fields)


def mutation(alternatives: Sequence[Any] = ALTERNATIVES) -> EnumeratedMutationProvider:
    return EnumeratedMutationProvider(EnumeratedMutationConfig(alternatives=tuple(alternatives)))


def planner(the_policy: HarnessSearchPolicy | None = None, **options: Any) -> HarnessPlanner:
    return HarnessPlanner(
        options.get("search") or BestFirstSearchProvider(),
        options.get("mutation") or mutation(options.get("alternatives", ALTERNATIVES)),
        the_policy or policy(),
    )


def outcome(*, single_task: bool = False, disclosure: str | None = None) -> LocalEvaluation:
    evaluator = HarnessOutcomeEvaluator(h06.source())
    config = h06.outcome_config(
        sampling={"kind": "first", "count": 1} if single_task else {"kind": "all"}
    )
    metadata = {"disclosure": disclosure} if disclosure else {}
    return LocalEvaluation(
        evaluator=evaluator,
        spec=bind_evaluation(evaluator, h06.evaluation_spec(config, **metadata)),
    )


def benchmark(*, single_task: bool = False, seeds: tuple[int, ...] = SEEDS) -> LocalBenchmark:
    return LocalBenchmark(
        suite=h06.suite(),
        source=h06.source(),
        sampling=FirstTasks(count=1) if single_task else AllTasks(),
        seeds=seeds,
    )


def search(
    workspace: Path,
    *,
    the_planner: HarnessPlanner | None = None,
    root: AgentHarnessSpec | None = None,
    evaluation: LocalEvaluation | None = None,
    the_benchmark: LocalBenchmark | None = None,
    objectives: tuple[ObjectiveView, ...] = OBJECTIVE,
    model: Any = None,
) -> LocalSearchReport:
    return run_local_harness_search(
        root=root or root_harness(),
        planner=the_planner or planner(),
        objectives=objectives,
        benchmark=the_benchmark or benchmark(),
        evaluation=evaluation or outcome(),
        content=content_source(),
        model=model or (lambda harness: ScriptedHarnessModel(SCRIPT)),
        toolbox=lambda: RecordingTools().toolbox(),
        recording=h06.recording(),
        workspace=workspace,
    )
