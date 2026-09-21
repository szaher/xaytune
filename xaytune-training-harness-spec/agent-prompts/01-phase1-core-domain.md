# Coding Agent Prompt — Phase 1 Core Domain

Implement only the core domain foundation.

Scope:

- typed IDs
- Actor
- ArtifactRef
- DatasetRef
- Experiment
- ExperimentNode
- Run
- RunAttempt
- status enums
- transition definitions
- unit tests

Do not implement:

- controller
- runtime
- trainer compiler
- SQLite
- agent
- resilience

Requirements:

1. Create `xaytune/core/` without importing PyTorch/Transformers.
2. All models serialize via Pydantic.
3. Add architecture import tests.
4. Add state transition unit tests.
5. Preserve all existing APIs.
6. Do not move existing trainer/recipe files.

Before completion run:
- ruff
- mypy
- pytest

Output:
- changed files
- tests
- remaining risks
