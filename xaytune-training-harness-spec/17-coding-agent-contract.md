# Coding Agent Contract

This file is intended to be handed directly to coding agents.

## 1. Mission

Implement the Xaytune experiment control-plane architecture incrementally without breaking existing training functionality.

## 2. Non-negotiable rules

### Rule 1 — Read before changing

Before coding:

- read this spec
- read applicable ADRs
- inspect current repository code
- inspect existing tests
- identify compatibility impact

### Rule 2 — No repository rewrite

Do not reorganize the whole repository in one change.

Each PR should create one architectural seam and route one path through it.

### Rule 3 — Preserve public compatibility

Until explicitly approved, preserve:

```python
xaytune.finetune(...)
xaytune.pretrain(...)
xaytune.align(...)
xaytune.evaluate(...)
```

and:

```bash
xaytune train
xaytune eval
xaytune pipeline
```

### Rule 4 — Compile, do not execute

Trainer integrations implement compilation.

Runtime integrations implement execution.

Do not add remote execution inside a TrainerCompiler.

### Rule 5 — Scientific lineage is immutable

Do not mutate an active or completed `CandidateSpecSnapshot`. Immutability is deep:
containers inside a snapshot must not be mutable either, or the record can change after
its fingerprint was taken.

A change that is an alternative candidate creates a child `ExperimentNode`.

### Rule 6 — Operational recovery stays operational

Worker retry, checkpoint restore, preemption, and approved execution overrides do not create scientific branches.

### Rule 6b — In-run scientific changes are interventions, not branches

A scientifically meaningful change applied to a run that is still going is a
`TrainingIntervention` on that run, not a new node (ADR-011). Use the comparability
rule: fork only when you would want to compare before and after as alternatives.

Every intervention is the recorded outcome of an approved `Action`. Never add a second
path that mutates training state directly.

### Rule 7 — No direct status mutation

Use transition APIs.

### Rule 8 — Persist state + event atomically

Never save state and append its event in separate transactions.

### Rule 9 — LLMs only propose typed actions

No arbitrary shell execution in core agent logic.

### Rule 10 — Heavy imports stay outside core

`xaytune.core` cannot import ML/runtime packages.

## 3. PR structure

Every PR must include:

- implementation
- tests
- docs
- changelog entry if user-visible
- migration note if compatibility affected
- clear scope statement
- explicit non-goals

## 4. Required checks

Run:

```bash
ruff check .
mypy xaytune
pytest
```

If the project has an existing CI command, run it too.

## 5. Test expectations

Every new:

- state transition → transition test
- event → serialization test
- plugin contract → compatibility test
- recovery path → fault injection test
- persistence mutation → crash/idempotency test
- public API → compatibility test

## 6. Change discipline

Prefer:

```text
add protocol
add adapter
route one code path
add tests
```

over:

```text
rename 50 files
rewrite CLI
rewrite trainer
rewrite configs
```

## 7. Error handling

Do not swallow exceptions.

Convert errors at architectural boundaries into typed failures/incidents.

## 8. Documentation

Public types require:

- docstring
- type hints
- example or reference docs where appropriate

## 9. Commit/PR size

Prefer PRs that can be reviewed independently.

If implementation starts affecting more than 3 architectural subsystems, stop and split the work.

## 10. Before marking a task complete

Verify:

- architecture invariants preserved
- existing tests pass
- new tests fail without the change
- no accidental heavy core imports
- state/event consistency
- rollback path understood
