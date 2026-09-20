# ADR-010 — Control-plane core has no ML runtime dependency

## Status
Ratified by merged implementation — 2026-09-21.

Implemented and verified: `xaytune/core/` imports on a bare interpreter with
only `pydantic` and `pyyaml` installed, with no ML stack present.

## Decision

The core import path depends only on lightweight schema/utility packages.

PyTorch, Transformers, TRL, Ray, TorchFT, etc. are optional extras/integrations.

## Rationale

An experiment control-plane client should be importable on a laptop, controller host, or service without installing the full ML runtime stack.
