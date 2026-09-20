# ADR-009 — Checkpoint codec, store, and manager are separate

## Status
Proposed

## Decision

- CheckpointCodec understands training state serialization.
- CheckpointStore understands storage.
- CheckpointManager coordinates lifecycle and compatibility.

## Rationale

S3/PVC/local storage should not understand FSDP/optimizer state semantics.
