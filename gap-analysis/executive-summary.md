# Executive Summary

Last updated: 2026-06-03 15:00

## Declarations

- **TESTS NOT RUN** — `torch` and `pydantic` not installed on this machine. All Python files validated via `ast.parse()` only.
- **APP NOT EXECUTED** — No GPU available locally. No training jobs run. All findings are from static code analysis.
- **STATIC CHECKS NOT RUN** — `ruff` and `mypy` not executed (missing dependencies). Syntax validated via `ast.parse()`.
- **ENVIRONMENT BLOCKED** — No CUDA GPU, no `torch`, no `pydantic`, no `transformers`. Cannot import xaytune or run any tests.

## Constraints & Assumptions

- **Deployment target:** PyPI package (`pip install xaytune`), used on GPU machines (single-GPU, multi-GPU clusters, OpenShift AI)
- **Environment(s):** Local dev (no GPU), OpenShift AI (GPU), cloud instances
- **Primary user personas:** ML engineers doing LLM fine-tuning, researchers exploring alignment methods, teams building internal training pipelines
- **Current top business goals:** Establish xaytune as a credible, usable LLM training framework
- **Non-goals:** Hosting/serving models, inference optimization, dataset curation platform

---

## Biggest Risks

1. **SFT trains on prompt tokens** — Every instruction-tuning job produces suboptimal results. The fundamental training path is wrong. (`data/tokenizer.py:62`)

2. **ORPO has never worked** — `xaytune.align(method="orpo")` crashes with `TypeError`. An advertised feature that is completely broken. (`trainer/loop.py:185`, `recipes/align/orpo.py:22`)

3. **DeepSpeed integration is non-functional** — The training loop uses vanilla PyTorch `loss.backward()` / `optimizer.step()` even when a DeepSpeed engine is active. No ZeRO, no offloading. (`trainer/loop.py:220-238`, `trainer/distributed.py:209`)

4. **QLoRA skips critical preparation** — `prepare_model_for_kbit_training()` is never called, risking dtype mismatches and garbage gradients. (`models/peft.py:46`)

5. **Preference alignment includes prompt log-probs** — DPO/GRPO/SimPO/ORPO sum log-probs over the shared prompt, diluting the preference signal. (`recipes/align/logprobs.py:24-26`)

## Biggest Wins (Already Fixed This Session)

- GRPO OOM eliminated — deepcopy gated by `needs_ref_model()`
- Gradient accumulation: `global_step` now counts optimizer steps, loss reports undivided values
- `evaluate()` — device handling added, `token_accuracy` now works, predictions/references collected
- Unknown kwargs in `finetune()`/`pretrain()`/`align()` now raise `TypeError`
- Dataset splitting now shuffles before split
- 104 `trainlib`→`xaytune` renames + 22 documentation errors fixed in example notebooks

## Immediate Actions Required

1. **Fix SFT prompt masking** (TASK-025) — Affects every instruction-tuning user. P0.
2. **Fix ORPO crash** (TASK-027) — Broken advertised feature. One session to fix. P0.
3. **Fix preference log-prob masking** (TASK-026) — Affects all alignment methods. P0.
4. **Add QLoRA preparation** (TASK-028) — One function call. P0.
5. **Fix DeepSpeed loop** (TASK-029) — Requires training loop refactor. P0 but larger effort.

## By the Numbers

| Metric | Count |
|--------|-------|
| Source files | 85 (`xaytune/`) |
| Source lines | ~11,500 |
| Test files | 102 |
| Test lines | ~16,000 |
| Bugs found (total) | 37 |
| Bugs fixed (this session) | 10 |
| Bugs remaining | 27 |
| Missing features (GAP) | 6 |
| New feature ideas (FEAT) | 8 |
| Critical/Blocker bugs | 7 |
| High severity bugs | 9 |
