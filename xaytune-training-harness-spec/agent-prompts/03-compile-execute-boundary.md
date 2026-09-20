# Coding Agent Prompt — Compile/Execute Boundary

Implement:

- TrainingSpec base and SFT model
- CandidateFingerprint and RunRealizationFingerprint (ADR-011; note seed belongs to
  the realization, not the candidate)
- TrainerCompiler protocol
- TrainingExecutionSpec
- RuntimeBackend protocol
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
SFT TrainingSpec
→ NativeCompiler
→ TrainingExecutionSpec
→ LocalRuntime
→ successful tiny training test or mocked worker execution
```
