# Roadmap

Last updated: 2026-06-03 14:00

## Milestone: NOW (Critical / Blockers)

Items that produce wrong results, crash training, or corrupt outputs. Ship before any user runs a training job.

| Epic | Rationale |
|------|-----------|
| **EPIC-0: Foundational Training Correctness** | **Highest priority.** SFT trains on prompt tokens (wrong). ORPO crashes (broken). Preference alignment includes prompt log-probs (diluted). QLoRA skips preparation (unstable). DeepSpeed loop is non-functional. Studio bypasses alignment losses. These affect every default training path. |
| EPIC-1: Training Loop Correctness | Already FIXED. Needs tests to confirm. |
| EPIC-2: Alignment Numerical Stability | ORPO NaN/Inf is a silent data corruption bug. GRPO OOM already FIXED. |
| EPIC-3: Eval Pipeline Completeness | eval_callback has same dummy-metrics bug. Device mismatch crashes GPU eval. Already partially FIXED. |
| EPIC-4: Checkpoint & Device Portability | Cross-device resume crashes. Tensor serialization can lose checkpoints. |
| EPIC-5: Config Validation (reinforce only) | `method="reinforce"` is rejected despite being supported. One-line fix. |

**Entry criteria:** Codebase is on `main` branch.
**Exit criteria:** All critical/high bugs resolved. SFT masking verified on alpaca/chat/sharegpt formats. ORPO runs end-to-end. DeepSpeed uses engine.backward()/step(). QLoRA calls prepare_model_for_kbit_training(). `ast.parse` passes on all modified files. Existing tests don't regress.
**Release strategy:** Direct merge to `main`. No feature flags needed — these are correctness fixes. Document behavior changes in CHANGELOG (SFT masking, preference log-prob masking change training dynamics).

---

## Milestone: NEXT (High-impact reliability)

Items that silently fail, mislead users, or degrade experience. Ship within the current dev cycle.

| Epic | Rationale |
|------|-----------|
| EPIC-5: Config Validation (full) | Silent config typos waste GPU hours. |
| EPIC-6: Logging Robustness | MLflow crash kills training. Backend isolation prevents cascading failures. |
| EPIC-7: Export Pipeline | model_merge output is unusable. GGUF conversion never works. |
| EPIC-8: Data Pipeline Edge Cases | Silent empty datasets waste GPU hours. |
| EPIC-9: Studio & CLI Surface Bugs | Studio is unusable for chat/text format training. |
| EPIC-10: Dependency Import Safety | Raw ImportError is hostile UX. |

**Entry criteria:** NOW milestone complete.
**Exit criteria:** All medium-severity bugs resolved. Studio format dropdown correct. MLflow backend doesn't crash. Export outputs are loadable.
**Release strategy:** Direct merge. Version bump to 0.7.0 since behavior changes (validation strictness, warning messages).

---

## Milestone: LATER (Polish & hardening)

Low-severity items and defensive improvements. Ship when convenient.

| Epic | Rationale |
|------|-----------|
| EPIC-11: Trainer Utilities | Reproducibility (numpy seed), multi-platform (gloo fallback), LR finder device. |
| EPIC-8: Data (agent tokenizer BOS) | Minor quality impact on agent fine-tuning. |
| Remaining low-severity items | SimPO clamp, constant scheduler warmup, studio theme. |

**Entry criteria:** NEXT milestone complete.
**Exit criteria:** All known issues resolved or explicitly deferred with rationale.
**Release strategy:** Batch with next feature release.

---

## Sequencing Rationale

1. **NOW first** because wrong results are worse than missing features. Users who already trained with gradient accumulation got wrong step counts and wrong loss — we need to fix and document this.
2. **NEXT second** because silent failures (empty datasets, crashes on config typos, unusable exports) waste GPU time, which is expensive.
3. **LATER third** because these are edge cases or minor quality improvements that don't block normal usage.

## Rollback

All NOW/NEXT fixes are pure correctness improvements. If a fix causes regression:
- `git revert` the specific commit.
- The "wrong" behavior is the status quo, so reverting doesn't make things worse than before.
- No migrations, no data model changes, no API contract changes.
