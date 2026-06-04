# Gap Analysis — xaytune v0.6.0

Last updated: 2026-06-03 15:00

## How to Read This

This gap analysis was performed on 2026-06-03 via systematic static code review of the entire xaytune codebase. No runtime execution was possible (no GPU, no torch/pydantic locally). All findings are verified against source code with file path evidence.

## Files

| # | File | Contents |
|---|------|----------|
| 1 | [executive-summary.md](executive-summary.md) | 1-page overview: biggest risks, biggest wins, immediate actions, declarations |
| 2 | [architecture.md](architecture.md) | C4-style architecture diagram, technology stack, data flows, external integrations |
| 3 | [feature-inventory.md](feature-inventory.md) | 16 feature areas mapped to entry points, modules, config, tests |
| 4 | [scorecard.md](scorecard.md) | Quality scorecard: each feature scored 0-5 on 6 dimensions with evidence |
| 5 | [bugs.md](bugs.md) | 37 bugs (BUG-001 to BUG-037): 7 Critical, 9 High, 14 Medium, 7 Low. 10 already fixed |
| 6 | [missing-features.md](missing-features.md) | 17 missing/incomplete features (GAP-001 to GAP-017) with acceptance criteria |
| 7 | [new-features.md](new-features.md) | 10 enhancement ideas (FEAT-001 to FEAT-010), MoSCoW prioritized |
| 8 | [remediation-roadmap.md](remediation-roadmap.md) | 3-phase plan: Now (0-2w), Next (2-6w), Later (6+w) with sequencing |
| 9 | [risks-and-unknowns.md](risks-and-unknowns.md) | Top 10 technical risks, top 10 product risks, 7 unknowns |

## ID Cross-Reference

All IDs are stable across files:

- **BUG-###** — Defects (bugs.md). 37 total.
- **GAP-###** — Missing/incomplete features (missing-features.md). 17 total.
- **FEAT-###** — New feature ideas (new-features.md). 10 total.
- **TASK-###** — Implementation tasks (in `implementation-plan/backlog.md`). 31 total.

## Relationship to Implementation Plan

The `implementation-plan/` directory contains the execution-ready task list derived from this analysis. Every BUG/GAP/FEAT item maps to at least one TASK-### (or is marked FIXED/deferred with rationale).

## Summary Statistics

| Category | Count |
|----------|-------|
| Bugs found | 37 |
| Bugs fixed (this session) | 10 |
| Bugs remaining | 27 |
| Missing features | 17 |
| New feature ideas | 10 |
| Critical/Blocker severity | 7 |
| High severity | 9 |
| Medium severity | 14 |
| Low severity | 7 |

## Key Findings

1. **The core SFT path produces wrong results** — prompt tokens are included in loss computation for instruction-tuning formats. This affects every user.
2. **ORPO has never worked** — crashes immediately with TypeError. An advertised feature that was never tested end-to-end.
3. **DeepSpeed is non-functional** — the training loop ignores the DeepSpeed engine and uses vanilla PyTorch.
4. **QLoRA is incomplete** — missing the critical `prepare_model_for_kbit_training()` call.
5. **The library overclaims** — 16 feature areas for a v0.6.0 library with ~11.5K lines of source. Several advertised features don't work.

## Verdict

xaytune has clean architecture and good structural test coverage (102 test files, ~16K test lines). The recipe pattern, registry system, and one-liner API are well-designed. But correctness issues in the core training paths make it unreliable for real training jobs. The recommended path: narrow scope, fix fundamentals (masking, ORPO, DeepSpeed, QLoRA), then re-expand deliberately.
