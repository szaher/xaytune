# Resilience and Adaptive Recovery

## 1. Resilience levels

```text
L0 process failure
L1 worker failure
L2 node / driver / runtime failure
L3 training-state recovery
L4 adaptive recovery
L5 experiment recovery
```

Xaytune coordinates L0-L2 through runtime/resilience providers.

Xaytune owns the policy and semantics for L3-L5.

## 2. Incident categories

Initial categories:

```text
PROCESS_FAILURE
WORKER_FAILURE
NODE_FAILURE
DRIVER_FAILURE
PREEMPTION

CUDA_OOM
HOST_OOM
DISK_FULL

NETWORK_FAILURE
OBJECT_STORE_FAILURE

CHECKPOINT_WRITE_FAILURE
CHECKPOINT_CORRUPTION
CHECKPOINT_INCOMPATIBLE

DATA_ERROR
DATA_CORRUPTION

NUMERICAL_NAN
NUMERICAL_INF
GRADIENT_EXPLOSION

LOSS_DIVERGENCE
TRAINING_STALL
QUALITY_REGRESSION

REWARD_COLLAPSE
KL_EXPLOSION

TIMEOUT

CONFIG_ERROR
USER_ERROR

UNKNOWN
```

## 3. Detection

```python
class IncidentDetector(Protocol):
    name: str

    def inspect(
        self,
        signal: RuntimeOrTrainingSignal,
        context: AttemptContext,
    ) -> IncidentCandidate | None: ...
```

Initial detectors:

- CudaOOMDetector
- NaNInfDetector
- ProcessFailureDetector
- CheckpointFailureDetector
- LossDivergenceDetector
- TrainingStallDetector
- RewardCollapseDetector

Structured signals are preferred over log parsing.

## 4. Classification

Phase 1 is deterministic.

```python
class IncidentClassifier:
    def classify(
        self,
        candidates: list[IncidentCandidate],
        context: AttemptContext,
    ) -> Incident: ...
```

Later an LLM may assist diagnosis, but deterministic evidence wins where available.

## 5. Recoverability

```text
RECOVERABLE_SAME_ATTEMPT
RECOVERABLE_NEW_ATTEMPT
RECOVERABLE_WITH_EXECUTION_OVERRIDE
REQUIRES_NEW_NODE
REQUIRES_HUMAN
UNRECOVERABLE
UNKNOWN
```

## 6. Recovery strategies

```text
FAIL
RETRY
RESUME
ROLLBACK
RUNTIME_RECOVER
EXECUTION_OVERRIDE
NEW_EXPERIMENT_NODE
PAUSE_FOR_APPROVAL
```

## 7. RecoveryPlan

```python
class RecoveryPlan(BaseModel):
    id: str
    incident_id: str

    strategy: RecoveryStrategy

    checkpoint_ref: CheckpointRef | None

    execution_overrides: list[ExecutionOverride]
    scientific_mutations: list[TrainingSpecMutation]

    reason: str

    requires_approval: bool

    estimated_budget_impact: BudgetDelta | None
```

## 8. CUDA OOM policy

Default adaptive flow:

```text
detect OOM
  ↓
check current micro batch
  ↓
check configured lower bound
  ↓
calculate smaller micro batch
  ↓
if preserve_effective_batch:
    increase grad accumulation
  ↓
check policy + capability + budget
  ↓
restore latest committed compatible checkpoint
  ↓
create new RunAttempt
  ↓
record ExecutionOverride
  ↓
resume
```

Example:

```text
before:
micro_batch = 4
grad_accum = 8
world_size = 8
effective_batch = 256

after:
micro_batch = 2
grad_accum = 16
world_size = 8
effective_batch = 256
```

No new experiment node.

## 9. Numerical failure policy

Default:

1. stop attempt
2. find previous known-good checkpoint
3. classify likely cause
4. decide whether recovery changes training semantics

Examples:

- restore checkpoint only → same node
- precision mode fallback if declared operationally equivalent by policy → execution override
- LR reduction → new experiment node
- optimizer change → new experiment node

Do not silently classify an LR change as operational recovery.

## 10. TorchFT provider

TorchFT is a resilience provider.

It may implement:

- per-step worker fault tolerance
- replicated training resilience
- fine-grained recovery

Xaytune translates high-level policy into provider config.

```python
class TorchFTResilienceProvider(ResilienceProvider): ...
```

Xaytune still owns:

- incident semantics
- policy
- whether an in-run scientific change is needed (a `TrainingIntervention`, proposed
  through the Action path) or a genuinely alternative candidate is needed (a new
  `ExperimentNode`) — see ADR-011 for the rule that decides
- experiment lineage
- budget
- evaluation after recovery

## 11. Ray resilience provider

Ray runtime/provider may handle:

- worker process retry
- node failure
- driver recovery
- checkpoint resume

Do not reimplement Ray's low-level mechanics.

Normalize Ray events into Xaytune incidents and attempts.

## 12. Recovery limits

Policy:

```python
class RecoveryLimits(BaseModel):
    max_attempts_per_run: int = 3
    max_recoveries_per_experiment: int = 10
    max_same_incident_repeats: int = 2
```

Repeated identical adaptive failures should escalate rather than loop forever.

## 13. Recovery loop protection

Maintain incident signatures.

Example signature:

```text
category + code-location + resource-shape + spec-fingerprint
```

If the same signature repeats after the same override, reject another identical recovery plan.

## 14. Required failure injection tests

- OOM at step N
- OOM after previous OOM override
- NaN after checkpoint
- process kill
- worker kill
- corrupt checkpoint
- object store transient failure
- evaluation failure
- preemption during checkpoint
- controller crash during recovery
