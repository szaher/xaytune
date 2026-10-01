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
standalone public execution writer. The OOM executor validates source resolution
and checkpoint bytes outside the database lock, then records the successor,
INTENDED submit operation, Action transition and EXECUTED receipt together after
rechecking governance, decision freshness, limits, checkpoint report identity,
budget and capacity under the write lock. This does not prove bytes remain valid
at runtime submission; the worker/runtime must consume the exact bound checkpoint.
Episode and plan remain immutable; successor existence remains closure authority.

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
require declared FULL+EXACT restore capability
  ↓
validate newest eligible FULL+EXACT checkpoint bytes
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

The PR-020 attempt resolver compiles the unchanged candidate first, then applies
the successor attempt's ordered cumulative operational overrides and checkpoint binding.
The Native and TRL compilation paths share this resolver. Each resize is a
micro-batch override followed by its compensating gradient-accumulation override.
Every `from` value must match the configuration reached so far, and each pair
must preserve effective batch. The final restore override must identify the
attempt's recorded checkpoint. Resize and restore overrides carry the governing
Action ID. Inconsistent lineage fails closed. The
resolved checkpoint reference is part of the canonical runtime request, so
rebuilding a submission after restart yields the same request digest.

An attempt's infrastructure failure and the logical Run's outcome are separate
transitions. A CUDA OOM settles the attempt as FAILED and releases its live-attempt
capacity while the Run remains ACTIVE for recovery. The existing Run reservation
continues; a successor under that Run spends no new run unit. Each failed attempt
still consumes one failure unit. The Run becomes FAILED only after recovery is
definitively refused, abandoned or exhausted, and SUCCEEDED only after an attempt
succeeds. Terminal Run states are never reopened. An approval-pending recovery
leaves the Run ACTIVE and makes `wait()` return an action-approval resting state.
Restart reconciliation finds ACTIVE Runs whose latest attempt failed, reuses the
episode's stored request and bound Action, and continues the same workflow.
If the first RecoveryRequest is unavailable, it leaves that failed attempt and
ACTIVE Run intact, writes no episode/decision/Action, and escalates. A later
attach with an explicit resolver can repair the gap.

The first PR-020 layer is the pure versioned `OOMRecoveryInputsV1` →
`OOMResizeProposal | OOMEscalation` contract. It requires a recorded effective
specialised CUDA-OOM plan, authoritative attempted configuration, and a promise
check against any preceding executed adaptive resize. The proposal is bound to
structured strategy/recoverability/accepted-diagnosis authority, never to the
human-readable `RecoveryPlan.reason` wording. It is also bound to
the plan revision and input fingerprint; it grants no execution authority. A
minimum micro-batch, nonintegral or over-limit accumulation, or an unapplied
previous resize escalates. Autonomous OOM adjustment never changes effective
batch. The one governed `ResizeMicrobatch` intent can carry both new knobs;
the successor attempt later records their separate `ExecutionOverride` lineage.

The governed proposal records an immutable `RecoveryActionBinding` at Action
creation time. It identifies the exact episode and plan revision, stores the
full OOM proposal plus its input/proposal/source-execution fingerprints, and is
committed with the Action, any policy decision, events and outbox. One plan revision
cannot mint two independently governed resize Actions. A stale or uncovered
plan cannot mint a new binding; replay of an existing identical proposal returns
its original Action without consulting current policy. The binding grants no
execution authority. An `EXECUTED` recovery receipt must reference an Action
bound to the same episode, plan and revision. The OOM executor rechecks approval,
resolved source configuration, declared restore capability, checkpoint bytes,
limits and freshness before creating a successor. A definitive refusal records
an ABANDONED receipt, rejects an unexecuted Action and fails the unresolved Run.
If new accepted incident evidence revises the effective plan before execution,
an older bound Action is rejected with a SUPERSEDED receipt; the new plan gets
its own governed Action. Approving the old Action after evidence arrives cannot
execute the obsolete plan.
The built-in local subprocess runtime claims FULL+EXACT application only for
the managed Native worker entrypoint. The worker verifies and materializes the
bound bundle through `CheckpointManager`, then applies model, optimizer,
scheduler, scaler applicability, RNG and the indexed data cursor before
continuing training. TRL and unsupported Native data layouts fail closed;
accepting a checkpoint reference alone never establishes a restore.

## 9. Numerical failure policy

Default:

1. stop attempt
2. find previous known-good checkpoint
3. classify likely cause
4. decide whether recovery changes training semantics

Examples:

- restore checkpoint only → same node
- precision mode fallback if declared operationally equivalent by policy → execution override
- LR reduction to stabilise the continuing trajectory → `TrainingIntervention` on the
  same Run (ADR-011); a new node only when LR values are alternatives to compare
- optimizer change → new experiment node

Do not silently classify an LR change as operational recovery.

### 9a. Numerical recovery execution model (PR-021, v1)

Numerical recovery uses a **checkpoint-backed successor attempt**, not mutation of a
live worker:

```text
attempt N ── NaN/Inf incident ── RecoveryEpisode E (target attempt N)
  → governed ChangeLearningRate Action → TrainingIntervention
  → validated FULL+EXACT checkpoint taken before the unsafe state
  → attempt N retired; successor attempt N+1 restored from that checkpoint
  → LR applied in attempt N+1 → confirmed InterventionApplication on attempt N+1
```

Consequences, all of which keep PR-019/020 semantics unchanged:

- Episodes stay one per attempt. The successor attempt **closes** E exactly as it
  closes an OOM episode; no intervention- or application-based closure exists.
- Instability observed later on attempt N+1 belongs to a **new** episode E′, which
  may propose a further reduction once the earlier one is reflected in the trajectory.
  One episode yields at most one numerical intervention.
- The node, the `CandidateFingerprint` and the Run are unchanged. The LR change is a
  `TrainingIntervention`, never an `ExecutionOverride`, even though it rides on a new
  attempt.
- `InterventionApplication` provenance is derived, not supplied: `previous_value` is
  the base rate on the applying attempt's retained trajectory immediately before the
  application, and `checkpoint_ancestor` is that attempt's own restore checkpoint.
  The database ties the ancestor to the attempt's `checkpoint_ref`.

In-place mutation of a running attempt would need several episodes per attempt, an
episode generation identity, application-based closure, and changes to repeat
accounting, membership routing and episode uniqueness. It is out of scope for v1.

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
