# Coding Agent Prompt — Compile/Execute Boundary

Implement:

- CandidateSpec, with TrainingSpec as its training component, and the SFT model
  (TrainingSpec holds the training program only -- model and data are siblings of
  it on CandidateSpec, and seed belongs to Run)
- CandidateFingerprint and RunRealizationFingerprint (ADR-011; note seed belongs to
  the realization, not the candidate)
- TrainerCompiler protocol
- TrainingExecutionSpec
- RuntimeBackend protocol, with `submit_or_get(operation_id, plan)` and
  `lookup_operation(operation_id)` -- never a plain `submit()` (ADR-013)
- ResolvedExecutionPlan
- capability skeleton
- LocalRuntime skeleton
- NativeCompiler adapter around existing local training path

Rules:

- TrainerCompiler must not execute training.
- RuntimeBackend must not interpret recipe semantics beyond execution requirements.
- TrainingExecutionSpec must JSON round-trip.
- Existing xaytune.finetune() must keep working.
- No TRL integration in this PR unless explicitly requested.

Add a single integration test:

```text
SFT CandidateSpec
→ NativeCompiler
→ TrainingExecutionSpec
→ LocalRuntime
→ successful tiny training test or mocked worker execution
```
