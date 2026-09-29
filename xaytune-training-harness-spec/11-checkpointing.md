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
        context: CheckpointContext | RestoreContext,
    ) -> str: ...  # canonical compatibility digest

    async def encode(
        self,
        state: CheckpointState,
        destination: Path,
        context: CheckpointContext,
    ) -> CheckpointManifest: ...

    async def decode(
        self,
        source: Path,
        manifest: CheckpointManifest,
        context: RestoreContext,
    ) -> RestoredCheckpoint: ...
```

Possible codecs:

- NativePyTorchCodec
- DistributedCheckpointCodec
- TRLCodec
- TorchFTCodec
- future FSDP/HSDP-specific codecs

### 2b. Optional validation-only capability (PR-019)

The implemented `xaytune.plugins/v1alpha1` `CheckpointCodec` ABI remains the
descriptor, compatibility key, encode and decode interface above. Historical
v1alpha1 codecs were not required to implement `validate`. They retain ordinary
save, restore and recorded-restore support under the existing provenance and
compatibility checks. Plugin descriptor version validation still applies.

Recovery eligibility additionally requires an explicit declaration of a separate
versioned validation contract. Set `PluginDescriptor.metadata` as follows, using
the publicly exported constant from `xaytune.checkpoints`:

```python
metadata={"checkpoint_validation_api": CHECKPOINT_VALIDATION_API_VERSION}
# CHECKPOINT_VALIDATION_API_VERSION = "xaytune.checkpoint-validation/v1alpha1"
```

The corresponding publicly exported interface is:

```python
class CheckpointValidationCodec(Protocol):
    def validate(self, manifest: CheckpointManifest) -> None: ...
```

The host accepts only the declared validation API version above; absent or
unsupported declarations raise `CheckpointCompatibilityError` and make the
checkpoint ineligible. Structural method presence and runtime protocol checks
do not declare support. A plugin that declares support but omits its implementation
has a programmer error, which propagates. The host never falls back to decode.
The extension inspects encoded layout and captured-state declarations, without decoding or
applying trainer state. `SerializedStateCodec` explicitly declares this extension.
`CheckpointManager.validate_recorded` checks bytes, report provenance and consumer
compatibility before invoking it. Ordinary `restore`/`restore_recorded` use the
original decode path and do not require the extension.

Validation failure contract:

| Failure | Meaning and handling |
|---|---|
| `CheckpointCompatibilityError` | Unsupported layout/consumer compatibility, or absent/unsupported validation declaration; checkpoint is ineligible. |
| `CheckpointCorruptionError` | Malformed or contradictory capture/bytes; checkpoint is ineligible. |
| `ValueError`, including Pydantic `ValidationError` | Supported malformed-capture convention; the manager normalizes it to `CheckpointCorruptionError`, preserving its cause. |
| `OSError` from localization or validation | Unavailable checkpoint bytes; the coordinator marks the checkpoint ineligible. |
| Other exceptions, such as `RuntimeError`, `TypeError` or `KeyError` | Programmer errors propagate; they are not evidence of checkpoint ineligibility and do not produce a plan for that incident. |

The coordinator checks the declaration before catching consumer-compatibility
failures. `CheckpointEligibility.reason` distinguishes `codec lacks validation capability`,
`checkpoint incompatible with intended consumer`, and
`checkpoint bytes/provenance missing or corrupt`; lack of validation support is
not recorded as consumer incompatibility.

The coordinator records supported failures as eligibility evidence and can
continue to an older valid checkpoint. Adding this optional extension does not
change the required v1alpha1 ABI; future incompatible changes to either contract
must use explicit version negotiation under ADR-008.

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
