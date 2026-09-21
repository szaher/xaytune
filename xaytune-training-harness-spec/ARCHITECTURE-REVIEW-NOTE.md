# Architecture Review Note — Xaytune 2.0 Control Plane

```yaml
baseline_commit: a34f240          # main @ merge of PR #10
spec_review_commit: bac8fde505edc57bc89417b5b9c1e7b346c37641
reviewed_at: 2026-09-21
supersedes: ARCHITECTURE-REVIEW-2026-09-20-PRE-IMPLEMENTATION.md
```

Status: review only. No code changed by this note.

> Implementation observations below are pinned to `a34f240`; governance and
> sequencing are refreshed against the specification at `bac8fde`. The previous note was written
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

## 2. ADR governance status

The original global Phase 0 gate has been resolved into a per-ADR gate. An ADR
must be settled before the work that depends on it, not before all implementation.

| Status | ADRs |
|---|---|
| Ratified by merged implementation | ADR-002, ADR-010 |
| Accepted | ADR-001, ADR-005 through ADR-007, ADR-011 through ADR-016 |
| Superseded | ADR-003 by ADR-011 |
| Proposed | ADR-004, ADR-008, ADR-009, ADR-017 |

Half-accepted statuses were removed: ADR-006's open reuse half is now ADR-017,
and ADR-001 and ADR-007 are accepted because accepted ADRs already depend on
them. An ADR status is a gate, so it has to be one value.

**There is no blocking governance decision left for work that is ready to
start.** ADR-005 was accepted on 2026-09-21, opening Band B including PR-004.
The remaining proposed ADRs retain their individual gates in
`15-implementation-plan.md` Phase 0 — ADR-008 for Band C, ADR-009 for Band F,
ADR-017 for Band G, ADR-004 for Band H — and they are not implicitly accepted
by this review.

---

## 3. Compatibility risk, at implementation baseline `a34f240`

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

ADR-011 through ADR-016 are accepted: candidate/intervention identity, data
cursors, operation identity, worker telemetry, durable evaluation, and persisted
specs versus live implementations. ADR-005 was accepted on 2026-09-21, so
persistence work may start.

- **Band B / Phase 1:** PR-005 implements atomic state/event/outbox writes and
  the `runtime_operations` journal in migration 002. Attempt + INTENDED submit
  operation + request digest commit before any runtime call. Repository restart
  reloads committed state, events, outbox and operation intents.
- **Band C / Phase 2:** LocalRuntime depends on that journal and uses
  `submit_or_get(operation_id, plan)` and `lookup_operation()`. PR-012a adds
  runtime-operation reconciliation and active-attempt reattachment.
- **Band D / Phase 3:** ADR-015's evaluation runs, attempts, results and
  reconciliation provide a durable evaluation lifecycle.
- **Band H / Phase 7:** daemon hosting adds leases, ownership and whole-controller
  startup reconciliation, reusing the earlier per-attempt recovery path.

ADR-014 telemetry identifies events by `(attempt_id, stream_generation, sequence)`.
Loss of replay history over a live workload advances the telemetry generation,
records a gap and degraded provenance, and preserves the execution attempt.
These are implementation contracts to verify in their scheduled PRs, not claims
that runtime or persistence implementations have already landed.
