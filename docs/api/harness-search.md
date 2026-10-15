# Harness Search

!!! warning "H07: local exploration, no control plane"
    A harness search proposes, verifies, runs and evaluates harness
    candidates locally, under a spending envelope, and writes a report. It
    touches no experiment and claims no durable orchestration: branching a
    harness proposal into an experiment is refused until harness candidates
    are durable (ADR-019 M1–M6). The decisions are in
    [ADR-024](https://github.com/szaher/xaytune/blob/main/xaytune-training-harness-spec/adrs/ADR-024-harness-search-and-mutation.md),
    accepted on 2026-10-15.

```text
HarnessSearchProvider.select(context)      where to search next (a base node), or why not
HarnessMutationProvider.mutate(request, 1) what to change: one typed HarnessPatch, or none
HarnessPlanner.plan: select → request → mutate → apply_patch → verify_harness_proposal → dedupe
run_local_harness_search: compile → run → verify evidence → evaluate → provider feedback → plan
```

## What a search may change

A `HarnessPatch` is a closed, typed list of operations on one pinned base:
the system prompt, the task template (same variables), a defined tool's
description, the enabled tools, the context policy and the stopping limits.
`apply_patch` refuses a stale base, an operation outside the allowed
surfaces, an undefined tool, a template with other variables, a result over
the experiment's ceilings, an invalid result and a no-op, with every reason.

**The enabled set only narrows:** `child.enabled ⊆ parent.enabled ∩
authorized_tools`. A tool the parent defines but has disabled stays
disabled; enabling it is a grant, and a grant is an Action. A prompt that
claims a permission grants nothing.

## What a provider may see

Only `ProviderFeedback` (run values after the provider filter, and per-task
execution measurements) and provider-safe comparisons computed from it: H06's
`compare()` over provider-visible values with no restricted report. A
single-task reference-derived score and any `successful-tasks` slice never
reach a provider, under either disclosure mode: a comparison on one is
`withheld-from-provider`, with no number and no verdict.

## Authorization and spending

A `HarnessSearchPolicy` must say `autonomous_search: authorized`; it has no
default. It fixes the authorized tools, the allowed operations, the
ceilings and the spending envelope, and all of them are in its fingerprint.

The envelope limits rounds, candidate evaluations (one per execution
replicate: three candidates on three seeds are nine), task attempts, model
calls per role and tokens per unit (units are never added together). Every
unit of work reserves its worst case before it starts; a unit that could
cross a limit never starts. A replicate's evaluation is admitted before any
of its task attempts runs, so no task runs for an evaluation that could not
start. A cost limit, a unit without a limit, and a
harness whose attempts are unbounded all refuse the search before anything
runs.

After a unit ends, it is charged what was measured. A judge call is charged
the tokens its provider reported, so a declared `judge_prompt_tokens` that is
too small is caught: a report above the reservation stops the search with
`spending-exceeded`. A call whose provider reported no token counts, and a
task attempt that raised, keep their whole reservation charged, and the
report names the error type (never its message).

A provider that breaks its contract, for example by choosing a base the
context lists as exhausted, or raises, ends the search with `provider-error`.
Any other error (an evaluator or evidence store that raises) ends it with
`error`, and the unit it interrupted is charged its whole reservation. The
report is written whatever ends the search, then the error is raised.

## Replay is planning replay

`replay_local_harness_search` re-plans each recorded round from its
recorded context and checks it gives the same choice, requests, refusals
and proposal. It replays decisions only: no model is called, no task runs
and nothing is evaluated again. It shows that the same evidence, given to
the same bound providers under the same policy, gives the same proposals,
not that the evidence itself would be reproduced.

## A worked example

```bash
python -m tests.harness_search_demo /tmp/search
```

runs three rounds on the pinned support suite with the scripted model and
prints every decision, comparison and unit of spend, then replays the
planning (not the execution) from the report.

::: xaytune.core.domain.harness_search.apply_patch
::: xaytune.core.domain.harness_search.verify_harness_proposal
::: xaytune.core.domain.harness_search.HarnessSearchPolicy
::: xaytune.core.domain.harness_search.LocalSearchEnvelope
::: xaytune.harness.search.HarnessPlanner
::: xaytune.harness.search.run_local_harness_search
::: xaytune.harness.search.replay_local_harness_search
