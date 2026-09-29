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

Note that adaptive micro-batch recovery has a hard precondition: it may resume only from
a checkpoint taken at an optimizer-step boundary with a batch-size-independent
`DataCursor` (ADR-012). Where the dataset or runtime cannot provide one -- streaming
sources, for example -- the coordinator must reject the recovery rather than approximate
it. An approximate resume that replays data while reporting success is worse than a
failed recovery.
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

## 7. RecoveryEpisode and RecoveryPlan (PR-019)

A failed typed attempt owns one immutable `RecoveryEpisode`, identified by a typed
ID and unique `(target.kind, target.id)`. It captures authoritative AttemptContext,
attempt number, candidate fingerprint, original explicit RecoveryRequest, pinned
coordinator identity and actor/time. It has no mutable state, revision, effective
pointer, closed flag or reservation counter.

`RecoveryEpisodeIncident` is append-only, globally unique per Incident and sequenced
within the episode. Accepted evidence participates in arbitration/signatures;
late-after-closure evidence is audit-only and cannot change decisions, reservations
or repeats. Classify membership under the incident-recording write lock using
successor existence, never incident/telemetry timestamps. Incidents without an
episode remain independent audit observations until explicit first planning.

An episode closes permanently for execution decisions when a higher-numbered
attempt exists on the same typed logical run. Successor failure/cancellation does
not reopen it. A superseded target with no episode remains historical Incident
only; do not manufacture a request-less episode or post-execution recovery decision.

`RecoveryPlan` is the immutable public conclusion/revision, not an execution
proposal. It contains episode ID, contiguous sequence, immediate predecessor ID,
accepted coverage sequence/IDs/fingerprint, deduplicated signatures, execution-state
fingerprint, typed RecoveryInputsV1, checkpoint eligibility and decision fields.
The highest sequence is effective. Every accepted evidence extension appends a
revision, even when strategy is unchanged. Accepted membership admitted before
replanning makes the old plan non-fresh; future execution must fail closed.
All original Incident evidence/candidates and previous decision revisions remain
queryable. New caller policy never silently replaces the episode's stored request.

RecoveryInputsV1 is a canonical immutable versioned projection of authoritative
context, typed statuses/revisions, actual attempt count/successor existence,
candidate/execution identity, original request/coordinator bindings, accepted
requirements/provenance, predecessor/coverage, other episode reservation usage,
prior accepted-signature episode counts and relevant checkpoint reports. Do not
embed full aggregates or every experiment plan. Unrelated non-reserving revisions
must not stale planning; relevant reservations/evidence/identity/attempts/reports do.
Optimistic retries are bounded.

Pure arbitration joins recovery authority requirements across all accepted detector
candidates, independent of arrival order. Unrecoverable evidence fails; specialised,
human/unsupported evidence blocks generic recovery; incompatible specialised
families escalate; compatible generic new-attempt evidence selects resume or
explicitly permitted retry. Never implement pairwise incident-category exemptions
or detector-priority arbitration. Disagreement behind UNKNOWN is retained.

First creation requires explicit RecoveryRequest and atomically records episode,
all known memberships, initial plan, RecoveryPlanned and outbox. Later revisions
also commit plan/event/outbox atomically. Existing episode membership admitted
separately emits RecoveryEvidenceAttached (including audit-only late evidence).
No duplicate attachment events for initial membership committed with a plan.
No aggregate revision changes merely for planning. Replay preserves original ID/time;
conflicting revision/ID replay is refused.

Checkpoint bytes/provenance are validated outside the database write lock, without
decode/application. Recording authoritatively rechecks database-derived typed
inputs and exact report/reference/fingerprint bindings, then recomputes the decision
from the trusted coordinator's eligibility evidence. Recording does not establish
cryptographic/current byte truth from eligible=True. A future executor MUST
revalidate the selected checkpoint before use and atomically verify episode open,
plan highest/current and all accepted evidence covered before successor creation,
with applicable Action/policy/capability/budget checks. These guards are future work.

Reconciliation replays complete episodes without configuration/files, repairs open
evidence gaps using original stored requests, and requires explicit reconstruction
only for missing episodes. Never invent RecoveryRequest(). Stop at the first
unresolved request gap: later reservations/repeats depend on preceding history.
Reconstruct the request and rerun; no skip mechanism.

PR-020 adds an append-only RecoveryExecutionReceipt keyed to the episode, plan ID
and sequence, governed Action, outcome (EXECUTED/ABANDONED/SUPERSEDED), actor and
time. EXECUTED links the successor attempt and INTENDED submit operation, and may
bind a selected checkpoint. It means durable submit intent, not runtime confirmation
or training success. At most one EXECUTED receipt is allowed per episode. Other
outcomes have no successor, operation or checkpoint. The receipt store has no
standalone public execution writer: a future executor must revalidate governance,
decision freshness, limits and checkpoint before inserting it atomically with the
successor and operation. Episode and plan remain immutable; successor existence
remains closure authority. Receipt schema alone does not create Actions, attempts,
operations, submissions or apply checkpoint state.

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
calculate integral grad accumulation preserving effective batch
  ↓
propose one ResizeMicrobatch action carrying both values
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

The first PR-020 layer is the pure versioned `OOMRecoveryInputsV1` →
`OOMResizeProposal | OOMEscalation` contract. It requires a recorded effective
specialised CUDA-OOM plan, authoritative attempted configuration, and a promise
check against any preceding executed adaptive resize. The proposal is bound to
the plan revision and input fingerprint; it grants no execution authority. A
minimum micro-batch, nonintegral or over-limit accumulation, or an unapplied
previous resize escalates. Autonomous OOM adjustment never changes effective
batch. The one governed `ResizeMicrobatch` intent can carry both new knobs;
the successor attempt later records their separate `ExecutionOverride` lineage.

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

One episode contributes one recovery unit iff its effective decision is RETRY or
RESUME. Observations and revisions contribute none independently. Attempt admission
counts actual attempts + other open episode reservations + proposed target unit;
experiment admission counts other episode units + proposed target unit. For generic
recovery, checkpoint eligibility and retry policy determine the candidate first:
`RETRY`/`RESUME` proposes 1, while `PAUSE_FOR_APPROVAL`/`FAIL` proposes 0 and cannot
be rejected by a reservation limit. Revisions replace the target contribution, so
RESUME → PAUSE releases current reservation
without mutating old rows. Closed reserving episodes retain one historical planned
unit and no pending attempt slot; this is not proof of execution consumption.

For each normalized signature, count distinct earlier episodes on the same typed
run with that signature in accepted evidence. Count duplicates/revisions once per
episode and exclude audit-only late evidence. max_same_incident_repeats=2 permits
prior matching counts 0/1/2 and escalates at 3. Zero permits the initial occurrence
and refuses its first repeat.

## 13. Recovery loop protection

Record versioned normalized signatures (category, structured code location,
resource shape, quantity and candidate fingerprint) and execution-state fingerprints
(execution fingerprint and existing override values/preservation claims).
Generic RETRY/RESUME do not promise execution change: unchanged execution identity
alone is never proof of a failed generic recovery loop. Generic recovery is bounded
by attempt, experiment and distinct-episode repeat limits.

PR-020+ strategies that promise an override must later verify the promised change
and escalate when the same failure recurs without it. No adaptive override or
numerical algorithm is implemented in PR-019.

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
