# Remediation Roadmap

Last updated: 2026-06-03 15:00

## Phase 1: NOW (0-2 weeks) — Correctness & Crashers

**Theme:** Make the default paths produce correct results. Fix crashes. Remove false advertising.

### Already Fixed (This Session)
| Item | File | Status |
|------|------|--------|
| BUG-003: GRPO OOM deepcopy | `recipes/align/align.py` | FIXED |
| BUG-004: global_step micro-batch counting | `trainer/loop.py` | FIXED |
| BUG-005: Loss divided by gradient_accumulation | `trainer/loop.py` | FIXED |
| BUG-006: token_accuracy always 0.0 | `eval/evaluate.py` | FIXED |
| BUG-007: evaluate() device mismatch | `eval/evaluate.py` | FIXED |
| BUG-008: Silent kwargs drop | `finetune.py`, `pretrain.py`, `align.py` | FIXED |
| BUG-009: Dataset split no shuffle | `data/loader.py` | FIXED |
| BUG-010: Streaming eval_split silent | `data/loader.py` | FIXED |
| BUG-001/002: Example notebook errors | `examples/*.ipynb` | FIXED |

### Must Fix Now
| Item | Effort | Blocks |
|------|--------|--------|
| BUG-031 / GAP-014: SFT prompt masking | M (2-3 days) | Nothing — can start immediately |
| BUG-033: ORPO crash fix | S (half day) | Nothing |
| BUG-011: ORPO numerical stability | S (half day) | BUG-033 (ORPO must work first) |
| BUG-032 / GAP-015: Preference prompt masking | M (2-3 days) | BUG-031 (same masking pattern) |
| BUG-035 / GAP-016: QLoRA preparation | S (1 hour) | Nothing |
| BUG-036 / GAP-017: DeepSpeed loop fix | L (3-5 days) | Nothing |
| BUG-034: PPO rename/documentation | S (half day) | Nothing |
| BUG-037 / GAP-010: Studio alignment wiring | S (half day) | BUG-033 |
| BUG-014: reinforce validation | XS (15 min) | Nothing |

**Sequencing rationale:**
1. SFT masking first — affects every user, unlocks preference masking.
2. ORPO crash + stability — quick win, removes a broken feature.
3. Preference masking — depends on SFT masking pattern.
4. QLoRA — one function call, high impact.
5. DeepSpeed — largest effort, can parallelize with above.
6. PPO rename + Studio fix — documentation and wiring, low risk.

**Quick wins (< 1 hour each):** BUG-014 (reinforce validation), BUG-035 (QLoRA prep).

**Risks:**
- SFT masking changes training dynamics. Add `mask_prompt=True` parameter for backward compat.
- Preference masking changes alignment loss values. Document in CHANGELOG.

**Rollback:** All changes are pure code fixes. `git revert` restores previous behavior.

---

## Phase 2: NEXT (2-6 weeks) — Reliability & Robustness

**Theme:** Fix silent failures, hostile error messages, and broken integrations. Ship v0.7.0.

| Item | Effort | Category |
|------|--------|----------|
| BUG-013 / GAP-008: torch.load map_location | S | Checkpoint portability |
| BUG-026 / GAP-009: Checkpoint tensor serialization | XS | Checkpoint reliability |
| BUG-020 / GAP-007: eval_callback metrics | S | Eval correctness |
| GAP-003: validate_config from API | S | Config safety |
| GAP-005: apply_overrides key validation | S | Config safety |
| GAP-004: Pretrain validation rules | S | Config safety |
| BUG-015: MLflow nested config crash | S | Logging |
| BUG-023: Logging exception isolation | S | Logging |
| BUG-022 / GAP-006: Import guards | S | Dependency UX |
| BUG-012 / GAP-013: model_merge config.json | S | Export |
| GAP-001: GGUF conversion command | M | Export |
| GAP-002: Hub push tokenizer warning | XS | Export |
| BUG-024: format_text silent empty | XS | Data |
| BUG-025: Preference split shuffle | XS | Data |
| BUG-016 / GAP-011: Studio format dropdown | XS | Studio |
| BUG-017 / GAP-012: CLI eval default metrics | XS | CLI |

**Quick wins (< 1 hour):** GAP-002, GAP-009, GAP-011, GAP-012, BUG-024, BUG-025 — 6 items, all XS.

**Foundational work:** Checkpoint portability (affects resume reliability), config validation from API (catches errors early), logging isolation (prevents training crashes from transient errors).

**Release:** v0.7.0 with CHANGELOG documenting all behavior changes from Phase 1 + 2.

---

## Phase 3: LATER (6+ weeks) — Polish & New Features

**Theme:** Edge cases, platform support, new capabilities.

| Item | Effort | Category |
|------|--------|----------|
| BUG-018: seed_all numpy | XS | Reproducibility |
| BUG-019: Distributed gloo fallback | XS | Platform support |
| BUG-021: LR finder device | S | Utility fix |
| BUG-027: SimPO zero-length guard | XS | Defensive |
| BUG-028: Agent tokenizer BOS | S | Agent FT quality |
| BUG-029: Constant scheduler warmup | XS | Scheduler |
| BUG-030: Studio theme | XS | Cosmetic |
| FEAT-002: Full PPO trainer | XL | New feature |
| FEAT-006: Prompt-masked eval metrics | M | Enhancement |
| FEAT-007: Multi-stage pipeline | L | New feature |
| FEAT-008: Real-time Studio monitoring | L | Enhancement |
| FEAT-010: AWQ/GPTQ export | M | New feature |

**Sequencing:** Fix remaining bugs first (all XS/S), then prioritize FEAT-006 (prompt-masked eval) since it complements the Phase 1 masking work. Full PPO and multi-stage pipeline are larger initiatives for v0.8.0+.

---

## Cross-Cutting Concerns

### Testing
- Phase 1 must add tests for every fix: SFT masking tests, ORPO e2e test, DeepSpeed mock test, QLoRA preparation test.
- Phase 2 should add cross-device checkpoint test, config validation edge cases, MLflow mock test.
- Target: every new test runs without GPU (mock models/tensors).

### Documentation
- Phase 1: Update CHANGELOG with behavior changes (SFT masking, preference masking, loss reporting, step counting).
- Phase 1: Update example notebooks if API changes.
- Phase 2: Update README to remove overclaiming. Be honest about what's experimental.

### CI/CD
- Consider adding `ast.parse` validation to CI (catches syntax errors without needing torch).
- Consider adding a `--no-gpu` test marker for tests that can run on CPU.
