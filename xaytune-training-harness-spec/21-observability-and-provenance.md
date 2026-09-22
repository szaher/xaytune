# Observability and Provenance

## 1. Structured event-first design

Do not make MLflow/W&B the source of truth.

Xaytune emits structured events.

Sinks consume them.

## 2. Event classes

Initial high-value events:

```text
ExperimentCreated
ExperimentActivated
ExperimentPaused
ExperimentCompleted

NodeCreated
NodeActivated
NodeDecisionMade
NodeCompleted

RunCreated
RunAttemptCreated
RuntimeSubmitted
RuntimeStarted
RuntimeCompleted

EventGapDetected
TelemetryStreamGenerationAdvanced
ObservabilityDegraded

MetricObserved

IncidentDetected
IncidentClassified

RecoveryPlanned
RecoveryStarted
RecoverySucceeded
RecoveryFailed

CheckpointSaveStarted
CheckpointCommitted
CheckpointRestoreStarted
CheckpointRestored

ActionProposed
ActionValidated
ActionApproved
ActionRejected
ActionExecuted

BudgetReserved
BudgetCommitted
BudgetConsumed
BudgetReleased
BudgetExceeded

ArtifactCreated
```

Evaluation has its own lifecycle events, because after ADR-015 it has durable
`EvaluationRun` and `EvaluationAttempt` entities and a single
`EvaluationCompleted` cannot express a queued, preempted or retried evaluation:

```text
EvaluationRunCreated
EvaluationAttemptCreated
EvaluationRuntimeSubmitted
EvaluationStarted
EvaluationSucceeded
EvaluationFailed
EvaluationPreempted
EvaluationCancelled
```

These mirror the `EvaluationRun`/`EvaluationAttempt` transitions in ADR-015 —
one event per transition, with no `Checkpoint*` or `Recovery*` counterparts,
because evaluation has neither. They must exist before PR-005 defines the event
schema; adding a transition event afterwards is a migration.

## 3. Log context

Every log should include where applicable:

```text
experiment_id
node_id
run_id
attempt_id
stream_generation
evaluation_run_id
evaluation_attempt_id
action_id
incident_id
runtime_backend
compiler
```

## 4. MLflow mapping

Recommended:

```text
Xaytune Experiment
  → MLflow Experiment

ExperimentNode
  → parent/nested run

Run / RunAttempt
  → nested child run or tagged child run
```

Tags:

```text
xaytune.experiment_id
xaytune.node_id
xaytune.run_id
xaytune.attempt_id
xaytune.candidate_fingerprint
xaytune.execution_fingerprint
xaytune.compiler
xaytune.runtime
xaytune.recovery_count
```

## 5. Provenance bundle

Every promoted artifact should be able to export the following. Note the three
separate identity documents: `candidate-spec.json` is what was *declared*,
`run-realization.json` is what actually *happened* — the ordered intervention
applications that ADR-011 hashes into `RunHistoryFingerprint`, with
`ArtifactLineageFingerprint` covering only the trajectory the artifact descends
from — and
`execution-plan.json` is *how* it was run. A candidate spec alone no longer
represents the scientific result, because two runs of the same candidate can
have different realizations.

```text
provenance/
  experiment.json
  graph.json
  candidate-spec.json
  run-realization.json
  execution-plan.json
  runtime.json
  evaluations.json
  incidents.json
  recoveries.json
  actions.json
  budget.json
  software.json
  artifacts.json
```

### Observability-degraded intervals

`TelemetryStreamGenerationAdvanced` and `ObservabilityDegraded` exist so that
provenance can distinguish

```text
nothing happened in this interval
we were not watching during this interval
```

which is otherwise the same absence of events. A telemetry supervisor that dies
while its workload keeps running produces the second, and the record says so on
the **same** `RunAttempt` — the attempt is not re-minted to represent a
monitoring failure (ADR-014 §1a).

## 6. Resource usage

Normalize:

```python
class ResourceUsage(BaseModel):
    wall_seconds: float

    gpu_seconds: float | None
    gpu_count: int | None
    gpu_model: str | None

    cpu_seconds: float | None

    peak_gpu_memory_bytes: int | None
    peak_host_memory_bytes: int | None

    tokens_processed: int | None
    examples_processed: int | None
```

## 7. Training memory

Phase 1:

structured search by:

- model
- dataset
- algorithm
- adapter
- candidate fingerprint — "has this hypothesis been explored?"
- run realization fingerprint — "do we have *this exact* trajectory?"
- execution fingerprint — "was it run on this software/hardware?"
- result
- incident category

Later:

semantic retrieval plugin.

Avoid a vector DB dependency in core.

## 8. Explainability

CLI/API should answer:

```text
Why was this node created?
Why was this candidate promoted?
Why was this run retried?
Why did the controller reduce microbatch?
Why did the experiment stop?
Which policy rejected an action?
```

These answers come from recorded structured rationale and provenance, not reconstructed guesses.

## 9. PR-009a: shared observation contracts

`xaytune.core.observability` defines policy; `xaytune.core.telemetry` defines
observations; `xaytune.core.resume` defines checkpoint evidence. These modules
have no trainer, collector, exporter or tracing SDK dependency.

The authority path is worker/runtime observation → `RuntimeEventEnvelope` →
validated durable controller event/state → outbox → `EventSink`. Controller
policy consumes structured evidence after validation and durable recording.
Runtime status and checkpoint reconciliation resolve stream gaps. External
observability systems never overwrite Xaytune's record.

`TrainingExecutionSpec.observability` holds `ObservabilitySpec` independently of
`TelemetryContract`. The latter identifies the transport protocol, heartbeat
and endpoint; it is not monitoring configuration. Observation intervals are
positive and finite. Capture flags default to true; missing metric values are
`None`, and zero remains a measurement. Numeric strings, booleans as counters,
and NaN/Inf numbers are rejected.

The typed observation families are:

| Body | Meaning |
|---|---|
| `TrainingMetricObserved` | Loss, LR, norms, throughput and step timings |
| `ResourceMetricObserved` | CPU/GPU utilization, memory, power, temperature, disk/network counters |
| `DataMetricObserved` | Interval consumption, padding/truncation, rejected examples and loader timings |
| `DistributedMetricObserved` | Rank/world size, synchronization, collectives and straggler timing |
| `AlignmentMetricObserved` | Reward, sampled KL, entropy, preferences and rollout/environment performance |

All utilization and fraction fields use **0–1**. Durations are seconds, byte
counters are nonnegative, and rates are per second. `straggler_ratio` means
slowest worker duration divided by mean worker duration and is at least 1.
Sampled KL, entropy, losses and reward estimates may be negative. Metrics are
optional across DPO, GRPO, PPO and environment rollouts. Provider details go in
immutable JSON metadata. None of these observations changes candidate identity.
`DataMetricObserved` measures an interval; `DataCursor` is checkpoint resume
state and must never be reconstructed from an interval metric.

`GradientOverflowObserved`, `OptimizerStepSkipped` and
`NumericalInstabilityObserved` report evidence without failing an attempt.
Nonfinite evidence is encoded symbolically (`nan`, `positive-infinity`,
`negative-infinity`), never as a non-JSON number. Incident classification retains
the taxonomy in the resilience specification; PR-009a adds no policy engine.

### Checkpoint evidence

`CheckpointCommittedPayload` requires a checkpoint reference, optimizer step,
explicit nullable cursor and three-dimensional `ResumeGuarantee`. FULL requires
`CheckpointStateManifest` references for model, optimizer, scheduler, scaler and
captured RNG state, the micro-step, and the application IDs already reflected
in the checkpoint. Empty application IDs means captured and none applied;
`None` means unknown. They identify **InterventionApplication** records, not
intervention decisions; restoration never deletes earlier applications.

`SamplerState` identifies its provider/version and captured state artifact.
`RNGState` references captured Python, NumPy and Torch CPU streams and per-worker
accelerator/dataloader streams under unique logical worker IDs. References may
point into the same committed artifact. This wire representation replaces the
ADR-012 illustrative bytes with artifact references; no tensors or pickles enter
events. Producers must enumerate every applicable stream; codecs must verify
referenced contents before restoration. A declaration cannot prove bytes exist.

EXACT requires complete captured state, a closed optimizer-step boundary,
sampler/ordering state and an indexed next-sample or provider cursor. It means
no data replay or skip, not bitwise numerical equality. Unsupported cursor
conversion or worker remapping must be rejected by future restore adapters.
PR-009a validates declarations; it does not implement capture, restore, cursor
conversion or adaptive resize. Weights-only checkpoints declare MODEL_ONLY/NONE
rather than claiming state they do not contain.

### Profiling and tracing

`ProfilerSpec` requests a bounded schedule: an enabled profiler needs a provider
and positive `active_steps`; wait/warmup are nonnegative and repeat is positive.
A disabled schedule may be retained for later activation. `output_name` is a
logical artifact name, not a path or trace body. `ProfilerStarted`,
`ProfilerCompleted`, `ProfilerFailed` and `ProfilerArtifactProduced` carry only
lifecycle/reference information; the trace is a `profile` artifact behind
`ArtifactRef`. Trace collection and PyTorch Profiler execution remain deferred.

`TracingSpec` declares enablement and sampling fraction. `TraceContext` carries
trace/span IDs and optional flags without importing OpenTelemetry. It can be
attached to envelopes and logs. `CorrelationContext` carries optional experiment,
node, run, attempt, evaluation run/attempt, action, incident, generation, runtime
and compiler identifiers. Envelope context cannot contradict the target or
stream generation. The envelope remains the sole ordering authority.

### Logs, callbacks and redaction

`RuntimeLog` is a human/debugging channel with level, logger, worker/rank,
timestamp, trace/correlation context and immutable attributes. **Never infer a
state transition solely from a log line.** Telemetry is the controller decision
channel; text matching is not a replacement for a typed observation.

`CallbackManager` remains local to the existing trainer and future NativeWorker.
No callable, hook, serialized Python code or arbitrary callback enters
`TrainingExecutionSpec`. Local callbacks may emit telemetry. Learning rate,
optimizer, data, algorithm, reward, adapter, precision semantics and stopping
policy changes cannot bypass lineage: scientific mutation follows
Action → TrainingIntervention; operational adaptation follows
Action/recovery → ExecutionOverride.

`RedactionPolicy` contains environment-key, attribute-key and pattern selectors,
not secret values. It is declared under the observation policy, applied by
producers before records cross the boundary, and does not run regexes in core.
**Secret values must never enter execution-plan payloads, telemetry, logs,
events, provenance bundles or request digests.** `SecretRef` exposes only name,
source and version. Live secret wrappers and objects are rejected recursively
by JSON metadata/config contracts. An ordinary string cannot be recognized as
a secret by a schema: producers must prevent raw credentials from entering
those fields, even if they are JSON strings. Redaction is not permission to
persist a secret and hide it later in a sink.

### Sinks and deferred implementations

`xaytune.core.sinks.EventSink` exposes `descriptor: PluginDescriptor` and
`async consume(event: DomainEvent)`. Delivery is at least once through the
existing outbox; consumers deduplicate by event ID. The delivery host isolates
sink failures and retries; a sink failure cannot fail training or undo committed
state. Console, MLflow, W&B, TensorBoard, OpenTelemetry, webhooks and custom
sinks are projections of committed facts. Existing `xaytune/logging/` backends
remain available for later adapters.

PR-009a provides contracts and validation only. NativeWorker callbacks, metric
collection, resource/GPU collectors, profiler execution, tracing/exporter SDKs,
sink adapters, redaction execution and policy decisions remain subsequent work.
