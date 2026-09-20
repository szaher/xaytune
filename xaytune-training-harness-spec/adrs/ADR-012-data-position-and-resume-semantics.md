# ADR-012 — Data position and resume semantics

## Status
Proposed

Required before PR-005 freezes persistence structures, and before any adaptive-recovery
work. The OOM recovery path described in `18-mvp-reference-scenario.md` is not sound
without it.

## Context

Xaytune's headline resilience feature is adaptive OOM recovery: shrink the micro-batch,
raise gradient accumulation to preserve the effective batch, restore the checkpoint,
continue. That claim is currently false, because the trainer resumes on a **batch index**
rather than a data position.

`xaytune/trainer/loop.py` resumes by counting batches:

```python
resumed_step = resume_state.step if resume_state is not None else -1
...
if epoch == (resume_state.epoch if resume_state else -1) and step <= resumed_step:
    continue
```

A batch index is only meaningful at one batch size. Change it and the same index means a
different amount of data:

```text
before OOM   micro-batch 4, 100 batches consumed   = 400 samples
recovery     micro-batch 2
resume       skips 100 batches × 2                 = 200 samples

200 samples are replayed, while the run reports exact continuation.
```

Raising the micro-batch instead skips unseen data. Either way the realized trajectory
silently stops matching what the provenance record claims, which defeats the point of
recording lineage at all.

The gap is wider than the cursor. `save_checkpoint()` writes model, optimizer, scheduler
and scaler state plus `{global_step, epoch, step, metrics}`. It captures **no** data
position, **no** sampler or permutation state, and **no** RNG state. On resume,
`seed_all(seed)` re-seeds from the original seed, so the random stream restarts from the
beginning rather than continuing: dropout masks, augmentation and the shuffle
permutation all replay from epoch zero. Resuming into epoch 2 can therefore reuse epoch
0's data ordering.

## Decision

### 1. Data position is a `DataCursor`, not a batch index

A cursor identifies **the next unconsumed sample**, together with the identity of the
data and of its ordering. It is independent of micro-batch size, gradient accumulation
and world size.

```python
class DataCursor(BaseModel):
    dataset_fingerprint: str          # which data
    ordering_fingerprint: str         # which permutation of it

    epoch: int | None
    next_sample_offset: int | None    # position in the deterministic sample stream

    sampler_state: SamplerState | None
    iterable_cursor: FrozenDict | None   # provider-supplied, for non-indexable sources

    examples_seen: int | None
    tokens_seen: int | None
```

`next_sample_offset` indexes the **deterministic sample stream** — the sequence the run
would consume given its dataset and ordering — not the dataloader's batch counter.

`ordering_fingerprint` covers the shuffle seed, the permutation algorithm, the epoch
transformation and any distributed sharding. Two runs with the same
`dataset_fingerprint` and different ordering consume different data in different order
and must not be treated as resumable into one another.

### 2. Checkpoints capture the whole resumable state

A checkpoint that omits any of these cannot support `EXACT` resume:

```text
model state
optimizer state
scheduler state
gradient scaler state

optimizer_step
micro-step position within the current accumulation window

DataCursor
sampler state
ordering/permutation state

RNG state, captured not re-seeded:
    Python random
    NumPy
    Torch CPU
    Torch accelerator (CUDA / MPS)

applied intervention state (ADR-011)
```

RNG **state** is captured and restored. Re-seeding from the original seed is not a
resume; it restarts the stream.

### 3. Checkpoints are taken at optimizer-step boundaries

A checkpoint taken mid-accumulation holds partial gradients that are not represented in
the optimizer state. Checkpoints are therefore taken only when the accumulation window
is closed, and a checkpoint records the boundary it was taken at.

Where a runtime cannot guarantee this, the checkpoint is marked as
`OPTIMIZER_STEP_BOUNDARY = false` and is not eligible for `EXACT` resume.

### 4. Checkpoints record applied interventions

Per ADR-011, re-application after rollback is computed by comparing an intervention's
position against the restored position. That requires the restored position to be
recoverable, so a checkpoint records the training position it was taken at and the
applications already reflected in its state.

Without this, the controller cannot tell whether an LR change is already baked into the
restored optimizer state, and will either double-apply or silently drop it.

### 5. Resume guarantees are declared, not assumed

```python
class ResumeSemantics(str, Enum):
    EXACT = "exact"
    OPTIMIZER_STEP_BOUNDARY = "optimizer-step-boundary"
    AT_LEAST_ONCE_DATA = "at-least-once-data"
    EPOCH_BOUNDARY = "epoch-boundary"
    MODEL_ONLY = "model-only"
```

Every resume records which level it achieved, and the level becomes part of the run's
provenance. A run resumed at `AT_LEAST_ONCE_DATA` has replayed some samples; that is
often acceptable, but it must be stated rather than discovered.

### 6. Adaptive batch resize requires a batch-size-independent cursor

> Batch-size-changing recovery may resume only from a checkpoint taken at an
> optimizer-step boundary with a batch-size-independent data cursor. Where the dataset
> or runtime cannot provide one, adaptive resize recovery is **unsupported** and the
> recovery coordinator must reject it rather than approximate it.

Rejection is the point. An approximate resume that silently replays 200 samples is worse
than a failed recovery, because the run continues and reports success.

For iterable and streaming sources — `xaytune/data/loader.py` supports
`streaming=True`, which yields non-indexable datasets — a cursor requires
provider-specific support. Absent it, those sources are ineligible for adaptive resize.

## Acceptance criteria

### Data position

- **AC-1.** Given a checkpoint taken after N samples, when the run resumes, then the
  first sample consumed is sample N+1 in the deterministic stream — regardless of the
  micro-batch size used before or after.
- **AC-2.** Given a checkpoint taken at micro-batch 4 after 400 samples, when the run
  resumes at micro-batch 2, then exactly 0 samples are replayed and exactly 0 unseen
  samples are skipped. This is the failing case today.
- **AC-3.** Given a cursor whose `dataset_fingerprint` differs from the dataset being
  resumed into, when resume is attempted, then it is rejected with a typed error rather
  than silently continuing.
- **AC-4.** Given a cursor whose `ordering_fingerprint` differs — a different shuffle
  seed, permutation or sharding — when resume is attempted, then it is rejected.
- **AC-5.** Given a token-budgeted run, when it resumes, then `tokens_seen` continues
  from the checkpoint value and remains consistent with `examples_seen`.
- **AC-6.** Given a dataset that cannot express a sample offset, when a cursor is
  requested, then the capability is reported as absent — never defaulted to a batch
  index.

### Sampler, ordering and RNG

- **AC-7.** Given a resumed run, when the next permutation is drawn, then it matches
  what the uninterrupted run would have drawn. Concretely: an uninterrupted 3-epoch run
  and a run interrupted and resumed mid-epoch-2 consume the same samples in the same
  order.
- **AC-8.** Given a resumed run, when RNG-dependent operations execute, then they
  continue the original stream rather than restarting it. Verified by comparing a
  dropout mask sequence across an uninterrupted and a resumed run.
- **AC-9.** Given a distributed run, when it resumes with the same world size, then each
  rank resumes its own shard at the right offset and no sample is consumed twice across
  ranks.
- **AC-10.** Given a resume with a **changed** world size, when the cursor cannot be
  re-sharded, then resume is rejected rather than silently re-partitioned.

### Optimizer-step boundaries

- **AC-11.** Given `gradient_accumulation = 8`, when a checkpoint is requested
  mid-window, then it is deferred to the next boundary, or written and marked ineligible
  for `EXACT` resume.
- **AC-12.** Given a checkpoint taken at a boundary, when it is restored, then
  `optimizer_step` and the accumulation position agree, and no partial gradients are
  implied.

### Interventions in checkpoints

- **AC-13.** Given an intervention applied at step 10,000 and a checkpoint at step
  12,000, when the run restores from that checkpoint, then the intervention is **not**
  re-applied — its effect is already in the restored state.
- **AC-14.** Given the same intervention and a checkpoint at step 8,000, when the run
  restores, then the intervention **is** re-applied on reaching step 10,000, subject to
  its replay policy, and a new `InterventionApplication` is recorded.
- **AC-15.** Given any restore, when the controller computes re-application, then the
  restored training position is read from checkpoint metadata rather than inferred.

### Resume semantics and adaptive resize

- **AC-16.** Given any resume, when it completes, then the achieved `ResumeSemantics`
  level is recorded on the attempt and visible in provenance.
- **AC-17.** Given a batch-size-changing recovery and a dataset with no
  batch-size-independent cursor, when recovery is planned, then it is **rejected** with
  a typed error naming the missing capability — not approximated.
- **AC-18.** Given a streaming dataset with no provider cursor, when adaptive resize is
  proposed, then the recovery coordinator reports the workload ineligible.
- **AC-19.** Given a resume claiming `EXACT`, when any required state from §2 is absent
  from the checkpoint, then the claim is rejected and the resume is downgraded to the
  level actually achievable.

## Consequences

### The current trainer needs real work

- `loop.py` must stop resuming on batch index and consume a cursor instead.
- `save_checkpoint()` gains cursor, sampler, ordering and RNG state, plus the training
  position and applied interventions.
- `seed_all()` stays for run start, but resume must restore captured RNG state instead
  of re-seeding.
- The dataloader must be constructed from an explicit, seeded generator so the ordering
  is reproducible and its state is capturable. Today `shuffle=True` draws from the
  global torch RNG.
- `DistributedSampler` use must record per-rank sharding in the ordering fingerprint.

### Some workloads become explicitly unsupported

Streaming and other non-indexable sources cannot support adaptive resize without
provider cursor support. Saying so is the decision; the alternative is an approximation
that corrupts the provenance record while reporting success.

### Backward compatibility

Existing checkpoints lack every field added here. They remain loadable at `MODEL_ONLY`
or `EPOCH_BOUNDARY`, and must not be claimed as `EXACT`. Checkpoint metadata gains a
version field so the distinction is machine-checkable.

## Rejected alternatives

**Keep the batch-index cursor and document the caveat.** The caveat would be that the
system's headline recovery feature silently corrupts data ordering. Documentation does
not make a provenance record true.

**Reconstruct the position by replaying the dataloader.** Only works if ordering is
reproducible, which is the thing not currently captured, and costs a full pass over
consumed data.

**Treat sample replay as acceptable and move on.** Defensible for some workloads, which
is why `AT_LEAST_ONCE_DATA` exists as a declared level. Not defensible as an undeclared
default, and not compatible with claiming exact continuation.
