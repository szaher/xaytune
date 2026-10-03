# Agent, Planner, Policy, and Budget

## 1. Separation of concerns

Do not combine these concepts:

```text
Planner
  proposes candidate/action

PolicyEngine
  decides whether it is permitted

BudgetLedger
  determines whether resources may be committed

Controller
  executes accepted actions
```

An LLM is one possible planner.

## 2. Planner protocol

```python
class Planner(Protocol):
    descriptor: PluginDescriptor
    spec: PlannerSpec                    # the bound spec it runs under (ADR-016)

    async def propose(
        self,
        context: PlanningContext,
    ) -> tuple[CandidateProposal | ActionProposal, ...]: ...
```

Initial planners (PR-024, `xaytune.planning`):

- RuleBasedPlanner -- ordered, typed mutation rules (first: `increase-lora-rank`)
  applied to the best `COMPLETED` candidate at the planning stage
- NoOpPlanner -- always `()`

A planner proposes; it never decides or applies. Objective and constraint
verdicts are the DecisionEngine's, attempt failures are recovery's, and
creating a node from a `CandidateProposal` is branching's (PR-025).

Later:

- LLMPlanner
- CompositePlanner
- SearchProviderPlanner

## 3. SearchProvider

Search providers propose parameterized candidates.

```python
class SearchProvider(Protocol):
    async def suggest(
        self,
        context: SearchContext,
        count: int,
    ) -> list[CandidateProposal]: ...

    async def observe(
        self,
        result: CandidateObservation,
    ) -> None: ...
```

Adapters:

- RayTuneSearchProvider
- OptunaSearchProvider
- KatibSearchProvider

Xaytune records why each candidate exists.

The search provider does not own scientific lineage.

## 4. Typed actions

Initial action types:

### Operational

- RetryRun
- ResumeCheckpoint
- CancelRun
- PauseExperiment
- ResumeExperiment
- RequestEvaluation

### Execution override

- ChangeMicroBatch
- ChangeGradientAccumulation
- ChangeWorkerCount
- ChangeCheckpointInterval

### Scientific

- ChangeLearningRate
- ChangeScheduler
- ChangeWarmup
- ChangeLoRARank
- ChangeAdapterType
- ChangeOptimizer
- ChangeDataset
- ChangeDatasetMix
- ChangePrecisionPolicy
- ChangeReward
- ChangeAlgorithm
- ChangeBaseModel

### Experiment

- BranchExperiment
- PromoteCandidate
- RejectCandidate
- StopExperiment

## 5. Action proposal

```python
class ActionProposal(BaseModel):
    action: ActionSpec          # a typed, registered spec instance -- never a mapping
    reason: str
    evidence_refs: tuple[str, ...]
    provenance: ProposalProvenance   # planner identity + context fingerprint
```

LLM output must be schema-validated. Only action types that exist can be
proposed; there is no `stop-experiment` or `request-evaluation` type, because
terminal outcomes belong to the DecisionEngine and evaluation to the
evaluation lifecycle.

## 6. Policy engine

```python
class PolicyEngine:
    def evaluate(
        self,
        action: ActionSpec,
        context: PolicyContext,
    ) -> PolicyDecision: ...
```

Checks:

- action allowed?
- user/org deny rules?
- target state valid?
- capability supported?
- budget available?
- approval required?
- safety constraint?
- mutation scientific or operational?
- checkpoint compatibility?
- max experiment/recovery limit?

## 7. Human approval

Policy example:

```yaml
approval:
  required_for:
    - change_base_model
    - change_dataset
    - increase_gpu_count
    - exceed_soft_budget
```

Controller state does not freeze the whole experiment if other branches can continue.

Action becomes:

```text
APPROVAL_PENDING
```

## 8. LLMPlanner

LLM receives curated context:

```python
class PlanningContext(BaseModel):
    objective: Objective

    current_nodes: list[NodeSummary]
    best_node: NodeSummary | None

    evaluation_summaries: list[EvaluationSummary]

    incident_summaries: list[IncidentSummary]

    budget_status: BudgetStatus

    allowed_action_schema: dict[str, Any]

    capability_summary: CapabilitySummary

    experiment_memory: list[PriorExperimentSummary]
```

As built in PR-024, `PlanningContext` (`xaytune.core.domain.planning`) is the
deterministic subset of this sketch. It holds the experiment's status and
objective, each node's summary (status, parents, candidate and its current
fingerprint, decisions, evaluation results by cycle), and the budget status.
The repository assembles it read-only (`planning_context()`). Its identity is
the versioned `planning_context_identity_v1`. Allowed-action schemas,
capability summaries and cross-experiment memory arrive with the planners that
need them, such as an LLM planner.

Do not provide:

- secrets
- raw Kubernetes credentials
- unrestricted shell
- unrestricted filesystem
- arbitrary runtime APIs

## 9. Decision recording

**As of PR-025, a consumed `CandidateProposal` is persisted** on the node it
creates (`ExperimentNode.branch_origin`): its versioned fingerprint, full
provenance, mutation and typed evidence. That makes "why does this node exist,
which planner and configuration, from which context, on what evidence"
answerable from the record. Planning rounds themselves are still not
persisted. What follows was the PR-024 position.

**As of PR-024, nothing here is persisted yet.** Planners are pure and their
output is unconsumed. Each proposal carries its own provenance: the planner's
provider, name and version, the bound `PlannerSpec`, the planning-context
identity version and fingerprint, a concise reason, and evidence refs. That
is enough to materialize and audit it later. The durable recording below
belongs where proposals are consumed: PR-025 keeps an accepted
`CandidateProposal`'s provenance with the branch it creates, and an
`ActionProposal` enters the existing durable Action path. PR-026 adds any
orchestration record it needs, and PR-032 adds LLM/agent audit.

Eventually store:

- provider
- model
- model revision where possible
- prompt template version
- context fingerprint
- structured output
- proposed actions
- policy results
- selected action
- concise rationale

Do not require storing hidden reasoning/chain-of-thought.

## 10. Budget ledger

A simple `consumed` counter is insufficient.

Use:

```text
limit
reserved
committed
consumed
released
```

Budget flow:

```text
candidate proposed
  ↓
estimate resources
  ↓
reserve
  ↓
submit runtime
  ↓
commit reservation
  ↓
collect actual usage
  ↓
consume
  ↓
release unused reserve
```

## 11. Budget dimensions

```python
class BudgetSpec(BaseModel):
    max_runs: int | None
    max_parallel_runs: int | None
    max_gpu_hours: float | None
    max_wall_time_seconds: int | None
    max_tokens: int | None
    max_cost: Decimal | None
    max_failures: int | None
```

Costs are provider-specific and optional.

On-prem environments may use GPU-hours without currency.

## 12. Parallelism safety

Before launching N candidates concurrently:

```text
sum(reservations) <= available budget
```

Avoid oversubscription caused by concurrent planners/controllers.
