# Open Questions and Deferred Decisions

These items should not block Phase 1 unless promoted to ADR.

## 1. Graph persistence representation

Options:

- adjacency table
- parent_ids JSON + index
- dedicated edge table

Recommendation: dedicated `experiment_edges` table.

## 2. Async API style

Possible:

- sync public API with internal async controller
- fully async public API
- both

Recommendation: sync convenience + async internals + async advanced API.

## 3. Controller daemon IPC

Options:

- Unix socket
- localhost HTTP
- SQLite polling
- gRPC

Recommendation: local HTTP/Unix socket after MVP; embedded first.

## 4. Remote controller deployment

Potential future implementations:

- Training Hub service
- OpenShift AI component
- standalone Xaytune controller

Do not choose before local durability works.

## 5. Artifact system

Core only stores refs.

Question: whether Xaytune should include an artifact transfer layer or rely on runtime/platform abstraction.

Recommendation: refs/contracts only initially.

## 6. Search provider concurrency

Need explicit scheduling policy once parallel search is implemented.

## 7. Scientific mutation classification

Partly answered by ADR-011: the node-versus-intervention question is decided by
comparability, and the operational boundary by whether declared training intent is
preserved. What remains open is which *settings* are scientific at all.

Some settings are borderline:

- precision
- world size
- gradient accumulation
- sequence packing
- data loader shuffling
- compilation flags

Recommendation: define a versioned `SemanticImpactPolicy` and default conservatively toward new scientific node where optimizer trajectory can materially change.

## 8. Effective batch equivalence

Preserving effective batch does not guarantee identical optimization due to numerical/order effects.

ExecutionOverride should therefore mean “declared intent-preserving under policy,” not “mathematically identical.”

## 9. Native trainer future

Long-term options:

- keep as reference backend
- reduce to tests/examples
- remove some algorithms in favor of upstream trainers

Do not decide until new adapter architecture is stable.

## 10. Training Hub API shape

Finalize adapter only against actual Training Hub API/contracts at implementation time.

## 11. Model registry integration

Out of core scope.

Potential plugin later.

## 12. Dataset/version resolver

Need provider-specific resolution:

- local file digest
- Hugging Face dataset revision
- S3 object/version manifest
- Iceberg snapshot

Implement minimal provider abstraction first.
