# ADR-009 — Checkpoint codec, store, and manager are separate

## Status

Accepted — 2026-09-28

## Decision

- `CheckpointCodec` owns the training-state serialization format and component
  validation, including the references backing declared captured state.
- `CheckpointStore` owns byte/object storage and atomic publication. It must not
  understand trainer semantics.
- `CheckpointManager` owns lifecycle coordination and compatibility checks.

A committed bundle has a versioned, immutable manifest with mandatory manifest
and file digests and producer-attempt, candidate and execution provenance.
Compatibility is checked before decode; unknown compatibility fails closed.
The local implementation uses conservative exact matching and never silently
reshards or downgrades the requested resume guarantees.

Publication is staging → validation/fsync → atomic commit. Incomplete staging
is never a recovery point. Localization and decoding return validated captured
state; they do not mean that trainer state has been applied. An achieved restore
is a separate, later fact. Checkpoint commit reports are durable observations,
not recovery decisions or restore operations.

PR-018 is local-only. Remote/distributed stores, retention/deletion, trainer
capture/application and recovery coordination remain later work.

## Legacy compatibility clarification — 2026-09-28

`xaytune.trainer.checkpointing.load_checkpoint` remains the legacy compatibility
and read surface; PR-018 does not remove or rewrite it. Existing directories
containing `model.pt`, `optimizer.pt`, optional `scheduler.pt` and `scaler.pt`,
and `metadata.json` remain usable through that surface, unchanged.

This clarifies R5 in the historical pre-implementation architecture review:
preserving legacy readability does not require the new `CheckpointManager` to
consume legacy directories. The legacy loader remains in the training stack;
neither `xaytune.core` nor importing `xaytune.checkpoints` depends on torch.

Legacy checkpoints lack the new manifest provenance, `DataCursor`, captured RNG
and intervention evidence. They are not automatically eligible for the new
control-plane recovery path and may never be claimed as `FULL + EXACT`.
Moving one into that system requires a future explicit import/migration feature
that establishes whatever provenance can actually be established and declares
the supported guarantees. PR-018 neither invents candidate/execution/attempt
provenance nor wraps old bytes in a manifest that pretends missing state exists.

## Rationale

Storage should not understand FSDP/optimizer state semantics. Separating byte
integrity, compatibility and trainer application keeps a durable commit from
being mistaken for either a recovery decision or an achieved restore.
