# ADR-010 — Control-plane core has no ML runtime dependency

## Status
Proposed

## Decision

The core import path depends only on lightweight schema/utility packages.

PyTorch, Transformers, TRL, Ray, TorchFT, etc. are optional extras/integrations.

## Rationale

An experiment control-plane client should be importable on a laptop, controller host, or service without installing the full ML runtime stack.
