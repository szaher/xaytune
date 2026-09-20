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

Critical lineage rule:

- microbatch reduction + compensating gradient accumulation is an ExecutionOverride on the same ExperimentNode
- LR reduction is a scientific mutation and must create a new ExperimentNode

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
