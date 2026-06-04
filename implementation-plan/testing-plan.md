# Testing Plan

Last updated: 2026-06-03 13:30

## Constraints

- No GPU available locally. Tests requiring CUDA must run on the OpenShift AI cluster.
- No `torch`/`pydantic` installed locally. Syntax checks via `ast.parse()` only.
- Existing tests in `tests/` require `torch` to import; integration tests require GPU.

## Coverage Map by Epic

| Epic | Tests to Add | Type | Can Run Locally |
|------|-------------|------|-----------------|
| EPIC-0 (Foundational) | SFT label masking (alpaca/chat/text), preference prompt exclusion, ORPO e2e, QLoRA preparation, DeepSpeed engine calls, Studio alignment loss | Integration + Unit | Partial (masking: yes, DeepSpeed: GPU only) |
| EPIC-2 (Alignment) | ORPO edge-case (logps=0), SimPO zero-length | Unit | Yes (pure tensor math) |
| EPIC-3 (Eval) | eval_callback with token_accuracy | Integration | No (needs model) |
| EPIC-4 (Checkpoint) | Cross-device load, tensor metric serialization | Unit | Yes (mock torch.save/load) |
| EPIC-5 (Config) | reinforce validation, override typo rejection, pretrain rules | Unit | Yes (pydantic only) |
| EPIC-6 (Logging) | MLflow nested config, backend exception isolation | Unit | Yes (mock backends) |
| EPIC-7 (Export) | model_merge loadability, GGUF error message, hub warning | Unit | Partial |
| EPIC-8 (Data) | format_text warning, preference shuffle | Unit | Yes |
| EPIC-9 (Studio/CLI) | dropdown choices, CLI eval default metrics | Unit | Yes |
| EPIC-10 (Imports) | missing peft/wandb/mlflow error message | Unit | Yes (sys.modules mock) |
| EPIC-11 (Trainer) | numpy seed, gloo fallback, lr_finder device | Unit | Partial |

## Test Data Strategy

- **Alignment losses:** Synthetic tensors with known values. Edge cases: zeros, very large negatives, mixed signs.
- **Checkpoints:** Use `tempfile.TemporaryDirectory()`. Save/load cycle with mock state dicts.
- **Config validation:** In-memory `TrainConfig` objects. No files needed.
- **Logging:** Mock backend classes that record calls or raise exceptions.
- **Data formats:** Inline sample dicts. No JSONL files needed.

## Regression Suite

Existing tests (`tests/test_*.py`) must continue passing after all changes. Run:

```bash
python3 -m pytest tests/ -x -q --tb=short
```

## New Test Files

| File | Covers |
|------|--------|
| `tests/test_sft_masking.py` | Prompt masking for alpaca, chat, sharegpt, text formats (EPIC-0) |
| `tests/test_preference_masking.py` | Prompt exclusion from preference log-probs (EPIC-0) |
| `tests/test_orpo_e2e.py` | ORPO end-to-end with toy model and preference data (EPIC-0) |
| `tests/test_qlora_preparation.py` | prepare_model_for_kbit_training called for quantized models (EPIC-0) |
| `tests/test_deepspeed_loop.py` | DeepSpeed engine backward/step delegation (EPIC-0) |
| `tests/test_alignment_edge_cases.py` | ORPO NaN, SimPO zero-length, GRPO no-ref-model |
| `tests/test_checkpoint_portability.py` | Cross-device load, tensor metric serialization |
| `tests/test_config_validation_extended.py` | reinforce, pretrain rules, override typos |
| `tests/test_logging_robustness.py` | MLflow flatten, exception isolation, import guards |
| `tests/test_eval_metrics.py` | token_accuracy with real predictions, device handling |
| `tests/test_data_edge_cases.py` | format_text warning, preference shuffle, agent BOS |
| `tests/test_cli_eval.py` | CLI eval with/without --metrics |

## Verification Sequence

1. `ast.parse` all modified `.py` files (local).
2. `python3 -m pytest tests/ -x` (on CUDA machine).
3. Manual spot-check: run `xaytune.finetune()` with `gradient_accumulation=4`, verify loss and step count.
4. Manual spot-check: run `xaytune.align(method="reinforce")`, verify no validation error.
