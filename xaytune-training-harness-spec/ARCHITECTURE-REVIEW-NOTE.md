# Architecture Review Note — Xaytune 2.0 Control Plane

```yaml
baseline_commit: a34f240          # main @ merge of PR #10
reviewed_at: 2026-09-21
supersedes: ARCHITECTURE-REVIEW-2026-09-20-PRE-IMPLEMENTATION.md
```

Status: review only. No code changed by this note.

> Regenerated against the tree at `a34f240`. The previous note was written
> before any Phase 1 work landed and described a repository that no longer
> exists — see the archived copy for the original migration mapping, which is
> still accurate about *where code goes*.

---

## 1. What changed since the pre-implementation review

Every blocker the earlier note raised has been resolved. Listing them matters,
because the earlier note is still linked from several chapters and a reader who
finds it first will believe none of this exists.

| Earlier finding | State at `a34f240` |
|---|---|
| `xaytune/core/` does not exist | Exists — 13 modules, ~1,700 LOC |
| Lazy top-level imports not implemented | Landed. `xaytune/__init__.py` resolves lazily (PEP 562) |
| `main` CI is red, 27 tests failing | Green on Python 3.10/3.11/3.12, lint and mypy clean |
| Deep immutability unspecified | Landed — `FrozenDict`, `deep_freeze`, `FrozenDomainModel`, `AggregateModel` |
| State-machine failure edges missing | Landed — every non-terminal Node/Run/Attempt state reaches `FAILED`/`CANCELLED` |
| `constant` + warmup semantics undecided | Resolved by BUG-029; warmup is honoured |
| `global_step` units undecided | Resolved by BUG-004; counts optimizer steps |
| v0.6 remediation plan stale | Reconciled (PR #10); 6 live tasks, all others closed |

### What `xaytune/core/` contains now

```text
core/ids.py        ULID-shaped sortable typed IDs (Crockford base32, monotonic)
core/errors.py     XaytuneError hierarchy
core/refs.py       ArtifactRef / DatasetRef / ModelRef / Actor
core/clock.py      injectable time source
core/immutable.py  FrozenDict, deep_freeze, FrozenDomainModel, AggregateModel
core/state/        status enums + normative transition tables
core/domain/       Experiment, ExperimentNode, Run, Objective
```

`core` imports pydantic and the standard library only, per ADR-010. This is
verified, not assumed: it imports on a bare interpreter with only `pydantic`
and `pyyaml` installed, with no ML stack present.

---

## 2. The one gate that is still open

**ADR-001 through ADR-010 are all `Status: Proposed`.** ADR-011, 012 and 013 are
`Accepted`.

`15-implementation-plan.md` Phase 0 says no feature implementation begins until
ADR-001…006 are accepted. That gate has been passed in practice, not by
decision: the implementations of ADR-002 (state machines) and ADR-010 (core
dependency boundary) are merged and under test on `main` right now.

This is a genuine inconsistency in the package and it needs a human decision,
not a documentation edit. An agent reading the plan literally will stop before
Phase 1, having been told a gate is closed that the repository has already
walked through.

The options are to accept the ADRs whose implementations have landed, or to
change the Phase 0 gate to describe what is actually required. **This note does
not make that call.** It is listed in `22-open-questions.md`.

---

## 3. Compatibility risk, at current `main`

| Risk | Where | Assessment |
|---|---|---|
| Import-graph regression | `xaytune/__init__.py` | `pipeline` must stay eagerly bound — it collides with the `xaytune/pipeline.py` submodule, and once anything imports the submodule it shadows a lazy attribute. Tested, but fragile to well-meaning cleanup |
| Latent circular imports | `trainer` ↔ `eval` ↔ `recipes` | One was found and broken (`metric_registry` resolves lazily). Others may exist on paths without test coverage |
| `TrainerConfig` straddles the boundary | `config/schema.py` | LR/epochs/seed are scientific; batch size, grad accumulation and activation checkpointing are executional — and are exactly what `ExecutionOverride` mutates. Splitting it is the first real compile-boundary decision |
| DeepSpeed is not end-to-end functional | `trainer/distributed.py` | BUG-036 is PARTIAL. The engine owns no optimizer, and checkpoint save/restore is unwired. Any runtime work that assumes a working DeepSpeed path is building on sand — see TASK-029 |
| Studio is a second control plane | `studio/jobs.py` | 401 lines of informal submit/status/cancel with its own persistence. Unchanged, and still the thing `ExperimentHandle` must supersede rather than coexist with |

---

## 4. Invariants worth asserting in tests before Phase 2

These are the properties that are cheap to hold now and expensive to recover
once persistence freezes:

1. `core` imports with no ML stack installed. *(held — tested)*
2. Every non-terminal state of a long-lived aggregate reaches `FAILED` and
   `CANCELLED`. *(held — tested)*
3. No direct status assignment outside aggregate internals; `model_copy(update=)`
   is refused on aggregates. *(held — tested)*
4. Domain values are deeply immutable, and deep-freezing rejects sets, NaN/inf,
   bytes and non-string keys. *(held — tested)*
5. **The transition tables in `04-state-machines.md` equal those in
   `core/state/machines.py`.** *(held — verified when that chapter was
   regenerated; not yet automated)*
6. A `CandidateFingerprint` does not change when an intervention is applied.
   *(not yet implemented)*
7. Replaying the event log reproduces projection state exactly. *(not yet
   implemented — belongs with PR-005)*

Invariant 5 is the one this review round exists because of: the chapter and the
code had drifted, and nothing detected it. It is a good candidate for a test
that parses the chapter, rather than a convention.

---

## 5. Sequencing risk

The largest remaining architectural risk is not in any single chapter. It is
that **persistence freezes before the contracts that determine its schema.**

PR-005 writes the event and outbox schema. That schema cannot be correct
without ADR-011 (what lineage records), ADR-012 (what a resume position is),
ADR-013 (what an operation is), and now ADR-014 (what a worker event is) and
ADR-015 (what an evaluation attempt is). All six exist as of this PR, which is
what unblocks PR-005 — and is the reason they were written before it rather
than alongside it.
