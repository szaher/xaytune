# Harness Evaluation

!!! warning "H06: local, in-process evaluation"
    Harness runs on a pinned benchmark can be measured, summarized and
    compared. Nothing is stored in the database or submitted to a runtime:
    the evaluation lifecycle runs locally until the durable harness path
    exists. The decisions are in
    [ADR-022](https://github.com/szaher/xaytune/blob/main/xaytune-training-harness-spec/adrs/ADR-022-harness-evaluation-scoring-and-comparison.md),
    accepted on 2026-10-12.

**A score is a measurement of evidence under a declared rule. It is not a
fact about the harness.** Two scores say something about two harnesses only
when the comparability conditions C1–C4 all hold, and even then a difference
is a provisional direction for choosing what to try next, never a
statistical claim.

```text
verified suite + seed ─record_assignment→ TaskAssignment
  each task: bind_task ─compile→ task-bound, recording AgentExecutionSpec ─run→ result + trajectory
assemble_evidence ─verify_run_evidence→ HarnessRunEvidence ─publish_evidence→ subject
bind_evaluation(harness-outcome | harness-judge) → run_harness_evaluation → EvaluationResult + reports
summarize (replicate-mean/v1) · comparability (C1–C4) · compare (paired-difference/v1)
```

## Running one replicate

A task is wired into execution through the spec's `config["task"]`; a spec
without it is exactly H03's. A bound spec must record its trajectory, and a
task with attachments is refused.

```python
from xaytune.core.domain.benchmark import SeededSample, record_assignment
from xaytune.core.harness_evaluation import bind_task
from xaytune.harness.evidence import HarnessEvidenceStore, assemble_evidence

store = HarnessEvidenceStore(Path("/runs/candidate-a/seed-101"))
sampling = SeededSample(count=20)
assignment = record_assignment(suite, sampling, seed=101)
specs = []
for task_id in assignment.task_ids:
    task, binding = bind_task(suite, assignment, task_id)
    spec = compiler.compile(harness, HarnessCompilationContext(
        task=task, seed=101, replicate=0, authorized_tools=tools, content=content,
        output_uri=str(store.task_output(binding.position)),
        trajectory=TrajectoryRecordingSpec(writer=WRITER),
        task_binding=binding, suite=suite,
    ))
    runner.run(spec, model=model, toolbox=toolbox, authorized_tools=tools)
    specs.append(spec)

evidence = assemble_evidence(specs, suite, sampling, assignment, source, store)
```

`HarnessRunEvidence` records every attempt instance of each task, read from
the store. An entry is `attested` (one attempt is sealed `complete` and
verifies) or `no-attested-result` (every attempt is sealed `partial` and
verifies: the outcome is absent and the spend is kept). A false claim of a
complete outcome is refused, never downgraded.

`verify_run_evidence(evidence, suite, source, store)` is the one
verification, used by assembly and by every evaluator run. It rebuilds every
fact from the suite and the store the caller names, and refuses an omitted
or added attempt, an open or quarantined one, a mismatched attribution, a
trajectory that does not verify, an entry with no attempts, and a digest that
is not its contents'.

## Measuring it

```python
from xaytune.evaluation.harness import (
    HarnessOutcomeEvaluator, bind_evaluation, publish_evidence, run_harness_evaluation,
)

subject = publish_evidence(evidence, store)          # kind "agent_run_evidence"
evaluator = HarnessOutcomeEvaluator(source)
spec = bind_evaluation(evaluator, EvaluationSpec(evaluator=EvaluatorSpec(
    name="harness-outcome", config=outcome_config)))  # supports → resolve → supports
evaluation = run_harness_evaluation(
    evaluator, spec, subject, source=source, output_uri="/evals/a-101", seed=101,
)
evaluation.result      # EvaluationResult: run-level MetricResults, or None
evaluation.report      # HarnessEvaluationReport: released per disclosure
evaluation.restricted  # RestrictedEvaluationReport: per-task scores, judge exchanges
```

The config (`xaytune.harness-evaluation/v1alpha1`) must be written in its
canonical form, every field explicit. Two evaluators exist:

| Evaluator | Measures | Determinism |
|---|---|---|
| `harness-outcome` | `task_success` by an `exact-match` rule, and every execution metric | `DETERMINISTIC` |
| `harness-judge` | `quality`, one integer on a declared scale, from a pinned judge model and rubric | `STOCHASTIC` |

### Metrics and absences

Catalog `xaytune.harness-metrics/v1`: outcome (`task_success`, `quality`),
tokens, cost, time, interaction (`turns`, tool calls, `retries`), context
and spend. Each per-task observation is a value or an absence with a reason
(`no-attested-result`, `unscored-cancelled`, `judge-failed`, `not-recorded`,
`not-applicable`, `spend-unaccounted`). **Zero never stands in for an
absence.** A metric absent on any task has no run value. Cost, cached and
reasoning tokens and end-to-end time are always `not-recorded` in v1.

A budget stop, a context overflow or a model error scores `task_success` 0,
and leaves `quality` absent (`not-applicable`): there is no answer to judge,
and no score is imputed. A cancellation is unscored.

### Disclosure and references

Held-out references are read only by the evaluator. Judge requests and
replies, and per-task reference-derived scores, go to the restricted report.
`metadata["disclosure"]` is `aggregate` (default) or `per-task`, and stays
outside the fingerprint. Under `aggregate`, a reference-derived run value over
a single task is withheld, and so is every `successful-tasks` slice, at any
sample count: next to the released per-task execution measurements, a mean
over the tasks that succeeded can tell which ones did. Under `per-task`, the
slices are released. `provider_feedback(report)` is what a search or mutation provider
may see, whatever the disclosure: never a per-task reference-derived score,
never a reference-derived run value over a single task, and never a
`successful-tasks` slice, whose mean next to the per-task execution
measurements can tell which tasks succeeded.

A judge with `reference_access="reference"` needs a `ReferenceEgressGrant`
for the suite and the pinned judge model, or `prepare()` refuses and nothing
is sent. Execution checks the prepared grant again, against the grants the
host supplied (`HarnessJudgeEvaluator(..., grants=...)`), before any
reference is read: a missing, altered or unissued grant is refused, and so
is a grant on an evaluation that sends no reference.

## Comparing two candidates

```python
from xaytune.core.harness_comparison import (
    CandidateRuns, ComparisonSpec, ObjectiveMetric, ReplicateRuns, compare,
)

report = compare(a, b, ComparisonSpec(objectives=(
    ObjectiveMetric(metric="task_success", direction="maximize", min_effect=0.05),
    ObjectiveMetric(metric="total_tokens", direction="minimize", min_effect=50),
)), suite)
report.comparability   # C1–C4: every failed condition, and the comparison key
report.metrics         # per metric: both summaries, paired differences, verdict
report.relation        # favors-a, favors-b, trade-off, inconclusive, not-comparable
report.claim           # always "exploratory"
```

- **C1** the same benchmark cohort; **C2** matched assignments, one replicate
  per seed on both sides (unpaired designs are refused); **C3** equal
  evaluation fingerprints; **C4** equal declared execution facets. Time
  metrics are compared only under one declared timing environment.
- `paired-difference/v1`: with R ≥ 3, A is favored when the mean paired
  difference exceeds `max(min_effect, k·se)` (`k` defaults to 2); fewer
  replicates are always `inconclusive`.
- `summarize` (`replicate-mean/v1`) reports the execution standard error and
  the judge's spread separately. With one replicate the uncertainty is
  unknown (`None`), not zero.

The vector is never collapsed into one score, and `Objective` and the
threshold decision engines are unchanged.
