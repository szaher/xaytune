# Checkpoint Architecture

## 1. Three-layer design

```text
CheckpointCodec
    understands training state format

CheckpointStore
    stores bytes/objects

CheckpointManager
    coordinates save / validate / commit / restore
```

## 2. CheckpointCodec

```python
class CheckpointCodec(Protocol):
    descriptor: PluginDescriptor

    def compatibility_key(
        self,
        context: CheckpointContext,
    ) -> CheckpointCompatibilityKey: ...

    async def encode(
        self,
        state: CheckpointState,
        destination: Path,
    ) -> CheckpointManifest: ...

    async def decode(
        self,
        source: Path,
        context: RestoreContext,
    ) -> RestoredCheckpoint: ...
```

Possible codecs:

- NativePyTorchCodec
- DistributedCheckpointCodec
- TRLCodec
- TorchFTCodec
- future FSDP/HSDP-specific codecs

## 3. CheckpointStore

```python
class CheckpointStore(Protocol):
    async def put_staging(...) -> StagingRef:
        ...

    async def commit(...) -> CheckpointRef:
        ...

    async def get(...) -> LocalizedCheckpoint:
        ...

    async def list(...) -> list[CheckpointRef]:
        ...

    async def delete(...) -> None:
        ...
```

Stores:

- LocalCheckpointStore
- S3CheckpointStore
- future PVCCheckpointStore
- future OCIArtifactStore

Store does not know optimizer/FSDP semantics.

## 4. CheckpointManager lifecycle

```text
SAVE_REQUESTED
  ↓
SERIALIZING
  ↓
STAGING
  ↓
VALIDATING
  ↓
COMMITTING
  ↓
COMMITTED
```

Incomplete states are not eligible for recovery.

## 5. Manifest

```python
class CheckpointManifest(BaseModel):
    schema_version: str

    files: list[CheckpointFile]

    candidate_fingerprint: str
    execution_fingerprint: str

    compatibility_key: str

    global_step: int | None
    epoch: float | None

    codec: str
    codec_version: str

    manifest_digest: str
```

`manifest_digest` is mandatory.

File digests are strongly recommended and may become mandatory by store policy.

## 5b. Required contents for resume

What a checkpoint must carry is set by ADR-012. Summarised:

```text
model / optimizer / scheduler / scaler state
optimizer_step + micro-step position in the accumulation window
DataCursor + sampler + ordering state
RNG state (Python, NumPy, Torch CPU, Torch accelerator) -- captured, not re-seeded
training position + applied interventions (ADR-011)
```

A checkpoint missing any of these cannot claim `state=FULL, data=EXACT`, and one taken
mid-accumulation is not eligible for it at all. Checkpoints record the boundary and the
training position they were taken at, because the recovery coordinator needs the
restored position to decide intervention re-application.

## 6. Compatibility

`CheckpointCompatibilityKey` includes what is required to determine safe resume.

Candidate fields:

- checkpoint state format version
- optimizer type/state layout
- scheduler state layout
- distributed strategy
- sharding scheme
- framework versions
- model architecture/revision
- adapter structure
- topology reshardability
- tokenizer/config identity where needed

Compatibility must be checked before submitting recovery.

## 7. Checkpoint intent vs execution contract

Scientific `TrainingSpec` contains intent:

```yaml
checkpoint:
  everySteps: 200
  keepLast: 3
```

Compiler/runtime creates execution contract:

```text
codec = torch-dcp
store = s3
async = true
reshardable = true
atomic_commit = true
```

## 8. Cross-runtime recovery

Do not promise cross-runtime recovery unless capability negotiation proves compatibility.

Example:

```text
Ray/FSDP checkpoint
  ↓
resume via Training Hub/HSDP
```

may be invalid unless codec/provider explicitly supports it.

## 9. Retention

Policy:

- keep last N
- keep best N
- keep incident checkpoints
- keep promoted-node checkpoints
- TTL for ordinary checkpoints

Deletion is evented and auditable.
