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

## 13. Phase 0 ADR governance — global-gate question resolved

The global gate has been replaced with per-ADR gates in
`15-implementation-plan.md` Phase 0. ADR-002 and ADR-010 are ratified by merged
implementation; ADR-011 through ADR-016 are accepted; ADR-003 is superseded in
substance by ADR-011.

ADR-004, ADR-009 and ADR-017 remain proposed, each gating its
dependent work. ADR-001 and ADR-007 were accepted because accepted ADRs already
depend on them, and ADR-006's open reuse half was split into ADR-017 so no
document carries two statuses at once. ADR-005 was accepted on 2026-09-21, which
opened Band B including PR-004; no remaining open decision blocks work that is
ready to start, and none of them is a reinstated global implementation gate.

## 14. When to unify training and evaluation execution

ADR-015 gives `EvaluationRun`/`EvaluationAttempt` the same state-machine shapes
as `Run`/`RunAttempt` and deliberately does **not** unify them under a generic
`Execution`/`ExecutionAttempt`.

Two aggregates with the same shape are not yet evidence of a shared
abstraction, and the differences are real: training produces checkpoints and
accepts interventions mid-flight; evaluation consumes a subject and a spec
without mutating training state, and has neither lifecycle.

Revisit when a third workload type appears. The likely candidates are a
data-preparation job or a reward-model scoring pass. Unifying before then would
mean carrying training-only concepts into evaluation and weakening both types.

## 15. Two representations of topology, one of which is stale

Raised in PR-012 review. The aggregates carry child lists that duplicate the
normalized relationships:

```text
Experiment.active_node_ids
ExperimentNode.run_ids
Run.attempt_ids
Run.final_attempt_id
```

Nothing writes them. The repository creates nodes, runs and attempts in their
own tables with a foreign key to the parent, and the controller reads
topology from those (`nodes_for_experiment`, `runs_for_node`,
`attempts_for_run`). So a submitted experiment has nodes, runs and attempts in
the tables while these fields stay empty. Nothing reads them yet, so nothing is
wrong yet; but one representation is out of date, and the first code to trust
it would be wrong without knowing.

Decide before planner, retry or multi-node work, which are the first to read
topology in bulk:

- **A.** Maintain them transactionally: every child creation also rewrites the
  parent at a new revision. This couples a child's creation to a write on its
  parent, which then contends with every other writer to that parent.
- **B.** (leaning) Remove them, or reclassify them as derived projections that
  are never stored. The normalized relationship stays the single source of
  truth, which is the rule the rest of the persistence layer already follows
  (ADR-005: aggregates from their own tables, events as provenance, derived
  data rebuildable).

`best_node_id` is not in this list. It records a decision, not a structural
relationship, and B would not remove it.
