# Implementation Plan

Original plan: 2026-06-03 14:00
Reconciled against the tree: 2026-09-20

## Scope: which plan governs what

This directory covers **remediation of xaytune v0.6** — the defects and gaps
found by the audit in `gap-analysis/`.

`xaytune-training-harness-spec/` covers a different workstream: building the
**experiment control plane** on top of v0.6, with its own phased PR plan and
ADRs. Neither supersedes the other. For control-plane work, follow that
package's `15-implementation-plan.md`, not this one.

## Current state

**26 of the 31 tasks are done.** Every bug-sourced task has landed; the five
still live are TASK-007, 008, 009, 014 and 015, all sourced from the open gaps
in `gap-analysis/missing-features.md`. See `backlog.md` for the list.

`testing-plan.md` and `observability-plan.md` are not tied to specific task IDs
and remain useful as-is.

## Index

| File | Purpose |
|------|---------|
| [foundational-review.md](foundational-review.md) | External review findings verified against source code (FOUND-001 through FOUND-007) |
| [triage-summary.md](triage-summary.md) | Dedupe decisions, confirmed/unconfirmed items, blockers |
| [epics.md](epics.md) | Epic definitions with scope, risks, exit criteria (EPIC-0 through EPIC-11) |
| [roadmap.md](roadmap.md) | Now / Next / Later milestones with sequencing rationale |
| [backlog.md](backlog.md) | Full task list (TASK-001 through TASK-031) |
| [dependencies.md](dependencies.md) | Task dependency graph and critical path |
| [testing-plan.md](testing-plan.md) | Test coverage map, test data strategy, regression plan |
| [observability-plan.md](observability-plan.md) | Logging/metrics/warnings per epic |
| [risk-register.md](risk-register.md) | Technical, migration, and security risks |

## How to Execute

1. **Read `roadmap.md`** to understand phasing.
2. **Start with NOW milestone** — 6 tasks, all independent, all P0/P1.
3. **Check `dependencies.md`** before starting any NEXT task — TASK-007 depends on TASK-006, TASK-009 depends on TASK-007.
4. **Follow each task's Definition of Done** checklist before marking complete.
5. **Run verification** per `testing-plan.md` after each task.

## Conventions

- **Task IDs:** `TASK-###` (3-digit, zero-padded).
- **Source IDs:** `BUG-###` / `GAP-###` — internal tracking from the audit. No external issue tracker.
- **Priority:** P0 = fix now (production impact), P1 = fix this milestone, P2 = fix next milestone, P3 = nice-to-have.
- **Estimates:** XS (<1h), S (1-4h), M (4-8h), L (1-2d), XL (3-5d).
- **Milestones:** Now (ship before any user trains), Next (current dev cycle), Later (batch with next release).

## Already Fixed (This Session)

These bugs were found and fixed inline during the audit. They are NOT in the backlog but are documented for completeness:

| Bug | File | Fix |
|-----|------|-----|
| OOM on GRPO alignment (deepcopy for all methods) | `recipes/align/align.py` | Gated deepcopy behind `needs_ref_model()` |
| `global_step` counts micro-batches | `trainer/loop.py` | Only increment on optimizer steps |
| Reported loss divided by gradient_accumulation | `trainer/loop.py` | Capture `loss.item()` before division |
| `token_accuracy` always returns 0.0 | `eval/evaluate.py` | Collect predictions/references from model outputs |
| `evaluate()` device mismatch crash | `eval/evaluate.py` | Move batches to model device |
| Unknown kwargs silently ignored | `finetune.py`, `pretrain.py`, `align.py` | Raise `TypeError` on unknown keys |
| `_split_dataset` doesn't shuffle | `data/loader.py` | Shuffle with `random.Random(42)` |
| Streaming + eval_split silently drops eval | `data/loader.py` | Emit `warnings.warn()` |
| `trainlib` references in example notebooks | `examples/*.ipynb` | Replaced with `xaytune` |
| 22 documentation errors in example notebooks | `examples/*.ipynb` | Fixed imports, field names, prose |

## Stats

- **Total tasks:** 31
- **NOW:** 13 tasks — 7 foundational (EPIC-0) + 6 original
  - EPIC-0: 5 P0, 2 P1
  - EPIC-1–5: 2 P0, 4 P1
- **NEXT:** 13 tasks (6 P1, 7 P2)
- **LATER:** 5 tasks (3 P2, 2 P3)
- **All BUG/GAP/FOUND items mapped:** Yes (see triage-summary.md and foundational-review.md)
