# Benchmarks

!!! warning "H04: contracts and resolution only"
    Tasks and benchmark suites can be pinned, resolved, verified, sampled and
    keyed into benchmark cohorts. Nothing yet runs a harness on a suite,
    scores it or stores it. The decisions are in
    [ADR-020](https://github.com/szaher/xaytune/blob/main/xaytune-training-harness-spec/adrs/ADR-020-benchmark-and-task-contracts.md),
    accepted with H04.

A benchmark is **evaluation and comparison identity**. The same harness on a
different suite keeps its `HarnessFingerprint`. Only the cohort its results
belong to changes.

## Tasks

```python
from xaytune.core.domain.benchmark import TaskSpec, agent_visible

task = TaskSpec.model_validate({
    "schema_version": "xaytune.task/v1alpha1",
    "id": "support/refund-001",
    "revision": "2026.10.1",                 # pinned; "latest", "main", ... are refused
    "inputs": {
        "variables": {"ticket": "Where is my refund?"},
        "attachments": [{"name": "policy", "content": {"kind": "ref", "uri": "...",
                                                       "digest": "sha256:...",
                                                       "media_type": "text/plain"}}],
    },
    "output": {"kind": "text-answer", "media_type": "text/plain"},
    "reference": {"kind": "ref", "uri": "...", "digest": "sha256:...",
                  "media_type": "text/plain"},  # held out: a reference, never inline
})
task.task_fingerprint()     # id, revision and every input byte, by digest
agent_visible(task)         # inputs only: never the reference
```

## Suites and resolution

```python
from xaytune.benchmarks import MappingBenchmarkSource, resolve_benchmark_suite
from xaytune.core.domain.benchmark import BenchmarkSuiteRequest

resolved = resolve_benchmark_suite(
    BenchmarkSuiteRequest(suite_id="support/core", revision="stable"),   # may be an alias
    MappingBenchmarkSource(manifests=..., tasks=..., blobs=..., aliases=...),
)
resolved.request    # what was asked for
resolved.ref        # BenchmarkSuiteRef(suite_id, pinned revision, suite_fingerprint)
resolve_benchmark_suite(resolved.ref, source)   # exactly the same suite, or refused
```

**Ordered manifest.** A `BenchmarkSuiteManifest` pins an ordered membership,
each member being a task id, revision and `task_fingerprint`. Order is
identity.

**What resolution verifies.** It reports every failure, not only the first:
- the suite id, and its revision or fingerprint;
- each member task's revision and fingerprint;
- every attachment, held-out reference and environment snapshot, by digest;
- that text content is `text/*` and valid UTF-8.

Reasons name roles, never locators.

**Pins can't move.** Once a pin has been recorded, an upstream alias that
moves cannot change it.

**Reproducibility.** It is `static` or `snapshot`. A snapshot suite pins its
environment state by digest. Dynamic services without a snapshot are refused
in v1.

**Structure on load, bytes against a source.** A `ResolvedBenchmarkSuite`
holds fingerprints and digests, not bytes:
- parsing one checks its structure;
- only a source can check its bytes.

Before executing or evaluating on a suite you loaded rather than just
resolved, call `verify_suite_content(suite, source)`.

`resolve_task_inputs(task, source)` returns the verified text of agent-visible
attachments. `resolve_reference(task, source)` returns the held-out text, and
is for evaluation only.

## Sampling and cohorts

```python
from xaytune.core.domain.benchmark import (
    SeededSample, assign_tasks, assignment_mismatches, benchmark_cohort, record_assignment,
    verify_assignment,
)

assign_tasks(resolved, SeededSample(count=20), seed=7)       # same ids on any machine, forever
a = record_assignment(resolved, SeededSample(count=20), seed=7)   # cohort, seed, task ids
verify_assignment(resolved, SeededSample(count=20), b)       # a stored record is only a claim
assignment_mismatches(a, b)                                   # () only if paired task for task
benchmark_cohort(resolved.ref, SeededSample(count=20))       # necessary, not sufficient
```

**Sampling.** `AllTasks`, `FirstTasks(n)` and `SeededSample(n)` always
return task ids in manifest order. A seeded sample ranks tasks by SHA-256,
never Python's `random`.

**Comparability needs every condition** (ADR-020 B7):

| | Condition | Provided by |
|---|---|---|
| C1 | The same benchmark cohort: suite (members, order, environment, reset) and sampling policy | `benchmark_cohort`, here |
| C2 | Matched assignments: the same replicate seed and the same tasks | `record_assignment`, `verify_assignment` and `assignment_mismatches`, here |
| C3 | The same scoring rules and units | H06 |
| C4 | Compatible, declared execution conditions | H06 / H07 |

An equal cohort means only that C1 holds. Candidates run under the same
replicate seeds get identical assignments, because assignment never depends
on the candidate, so they can be compared task by task. An unmatched pair is
reported, not silently pooled.

**Verify assignments before comparing them.** `assignment_mismatches`
compares two records and nothing else, so two fabricated records can agree.
Pass each record through `verify_assignment` first. It recomputes the cohort
and the tasks from the suite, the policy and the record's seed, and refuses
any difference.
