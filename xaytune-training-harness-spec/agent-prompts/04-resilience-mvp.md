# Coding Agent Prompt — Resilience MVP

Implement:

- Incident model
- CUDA OOM detector
- NaN/Inf detector
- RecoveryPlan
- ExecutionOverride
- recovery coordinator
- local checkpoint manager integration
- OOM microbatch adaptation
- effective batch preservation
- fault injection test

Critical lineage rule (ADR-011 — read it before starting):

- microbatch reduction + compensating gradient accumulation is an ExecutionOverride on the same ExperimentNode
- LR reduction taken to stabilise a run that is still going is a `TrainingIntervention`
  on that run. It does **not** create a new ExperimentNode: the optimizer state, data
  position and weights all carry forward, so the "before" is not a candidate anyone
  would ship.
- LR compared as an alternative — branch from a checkpoint and run 2e-5 against 1e-5 —
  is two ExperimentNodes, because the point is the comparison.

Every intervention is the recorded outcome of an approved Action. Do not add a second
mutation path.

Adaptive resize has a hard precondition (ADR-012). It may resume only from a checkpoint
taken at an optimizer-step boundary carrying a batch-size-independent `DataCursor`.
Resuming on a batch index is the bug this replaces: halving the micro-batch replays half
the consumed data while reporting exact continuation. Where no cursor is available,
reject the recovery -- do not approximate it.

Do not add TorchFT/Ray yet.

Required test:

- inject OOM after checkpoint
- classify incident
- create new RunAttempt
- restore checkpoint
- apply override
- finish training
- verify same node
- verify provenance

Second required test — the distinction this prompt previously got wrong:

- inject loss instability mid-run
- propose an LR reduction through the Action path
- verify it is recorded as a TrainingIntervention with origin REACTIVE_*
- verify **no** new ExperimentNode was created
- verify the InterventionApplication carries its training position
- verify the run's realization fingerprint changed but its candidate fingerprint did not
