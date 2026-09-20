# ADR-003 — Scientific lineage and execution lineage are separate

## Status
Proposed — extended by ADR-011.

The two-level split below is correct for changes made *between* runs. It has no
category for a scientifically meaningful change applied to a run that is still
going, such as a reactive learning-rate drop or a planned curriculum transition.
ADR-011 adds `TrainingIntervention` for that case and states the rule that
decides between it and a new node.

## Decision

`ExperimentNode` is a scientific candidate.

`RunAttempt` is an execution attempt.

Operational recovery remains within the same node.

Scientific mutation creates a new node.

`ExecutionOverride` represents policy-approved operational adjustments that preserve declared training intent.

## Examples

Same node:

- worker restart
- preemption
- checkpoint restore
- microbatch reduction with effective-batch preservation

New node:

- LR change
- LoRA rank change
- optimizer change
- dataset change
- reward change
- model revision change

## Consequences

Experiment graphs remain meaningful instead of becoming infrastructure logs.
