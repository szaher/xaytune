# ADR-003 — Scientific lineage and execution lineage are separate

## Status
Superseded in substance by ADR-011 — 2026-09-20.

Its two-level lineage model (node vs attempt) is replaced by the four levels in
ADR-011: `ExperimentNode`, `TrainingIntervention`, `ExecutionOverride` and
`RunAttempt`. Retained for the reasoning that led there. **Read ADR-011 for the
current model** — this is not an equally current alternative.

The two-level split below is correct for changes made *between* runs. It has no
category for a scientifically meaningful change applied to a run that is still
going, such as a reactive learning-rate drop or a planned curriculum transition.
ADR-011 adds `TrainingIntervention` for that case and states the rule that
decides between it and a new node.

## Decision

`ExperimentNode` is a scientific candidate.

`RunAttempt` is an execution attempt.

Operational recovery remains within the same node.

An alternative scientific candidate creates a new node. A scientifically meaningful
change to a run that is still going is a `TrainingIntervention` on that run, not a new
node — see ADR-011.

`ExecutionOverride` represents policy-approved operational adjustments that preserve declared training intent.

## Examples

Same node, new attempt:

- worker restart
- preemption
- checkpoint restore

Same node, execution override:

- microbatch reduction with effective-batch preservation

Same node and run, training intervention (ADR-011):

- declared LR schedule firing mid-run
- LR lowered to stabilise a destabilising run
- planned curriculum or data-mixture transition

New node:

- LoRA rank change
- optimizer change
- dataset change
- reward change
- model revision change
- LR compared as an alternative, branched from a checkpoint

Note LR appears under both intervention and new node. The kind of parameter does not
decide lineage; experimental intent does.

## Consequences

Experiment graphs remain meaningful instead of becoming infrastructure logs.
