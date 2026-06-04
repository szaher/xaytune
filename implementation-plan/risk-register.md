# Risk Register

Last updated: 2026-06-03 14:00

## Technical Risks

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|-----------|--------|------------|
| R1 | ORPO loss change alters training dynamics | Medium | Medium | New formulation is mathematically equivalent for normal inputs. Only edge cases (logps near 0) change from Inf/NaN to valid values. Add regression test comparing outputs for normal inputs. |
| R2 | `global_step` fix breaks existing checkpoints | Low | High | Old checkpoints had inflated `global_step`. On resume, training may run more steps than expected. Document in changelog. No data loss risk. |
| R3 | Adding validation to `setup_training()` rejects configs that previously worked | Medium | Medium | Gate strictly: only reject configs that are provably wrong (invalid method names, impossible combinations). Don't reject edge cases that might work. |
| R4 | `map_location="cpu"` changes tensor device on resume | Low | Low | Tensors are immediately moved to the correct device by `model.to(device)` and optimizer setup. No functional change after initialization. |
| R5 | Logging exception isolation masks real errors | Low | Medium | Only catch backend-specific exceptions, not `KeyboardInterrupt` or `SystemExit`. Log suppressed errors at WARNING level. Auto-disable after threshold. |
| **R6** | **SFT masking changes training dynamics for ALL users** | **High** | **High** | Models trained before the fix were trained on prompt+response. After the fix, only response tokens contribute to loss. This is correct behavior, but existing trained models are not reproducible with the new code. **Mitigation:** Document prominently in CHANGELOG. Add a `mask_prompt=True` parameter (default True) so users can opt out if needed for backward compatibility. |
| **R7** | **Preference log-prob masking changes alignment loss values** | **High** | **High** | Existing DPO/SimPO/GRPO runs included prompt log-probs. Removing them changes the loss landscape. Alignment results are not reproducible. **Mitigation:** Document in CHANGELOG. This is a correctness fix — the old behavior was wrong per the DPO paper. |
| **R8** | **DeepSpeed fix increases code complexity in training loop** | Medium | Medium | Training step must branch on engine type (vanilla vs DeepSpeed). **Mitigation:** Extract DeepSpeed-specific logic into a helper method. Keep the main loop readable. |
| **R9** | **QLoRA prepare_model_for_kbit_training may break custom model architectures** | Low | Medium | Some custom models may not be compatible with the preparation function. **Mitigation:** Wrap in try/except with a warning. |

## Migration Risks

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|-----------|--------|------------|
| M1 | Users relying on divided loss values for LR tuning | Medium | Low | Loss values were wrong before. New values are correct. Document in changelog that loss reporting is fixed. |
| M2 | Users relying on micro-batch step counts for scheduling | Low | Medium | Callbacks now fire per optimizer step, not per micro-batch. Any custom callback that assumed micro-batch granularity needs adjustment. Document. |
| **M3** | **Users who tuned hyperparameters against prompt-inclusive SFT** | Medium | Medium | Their LR/batch size choices were tuned against loss computed on prompt+response. After masking, loss is higher (fewer tokens contribute). **Mitigation:** Note in CHANGELOG that learning rates may need re-tuning. |
| **M4** | **Users with existing ORPO workflows** | Low | Low | ORPO never worked, so no one has a working ORPO workflow to break. This is a pure fix. |

## Rollback Plan

All changes are pure code fixes with no data model changes, no migrations, no external dependencies. Rollback = `git revert <commit>`. Previous (buggy) behavior is restored. No data loss in any scenario.

## Security / Privacy Risks

None identified. xaytune does not handle authentication, PII, or network communication (except HuggingFace Hub push, which uses the user's existing HF token). No changes in this plan affect security boundaries.
