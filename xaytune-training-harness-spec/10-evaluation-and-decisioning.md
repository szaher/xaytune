# Evaluation and Decisioning


> **Evaluation has a durable execution lifecycle — see ADR-015.** The
> `EvaluationSpec → EvaluationResult` shape below describes the *scientific*
> contract. Operationally an evaluation is a workload: it queues, fails, gets
> preempted, is retried as a new attempt and can be in flight across a controller restart. It
> therefore has `EvaluationRun` and `EvaluationAttempt`, which follow the same
> **lifecycle principles** as `Run` and `RunAttempt` — every non-terminal state
> reaches `FAILED` and `CANCELLED`, and `PREEMPTED` applies from `QUEUED`
> onwards — using the evaluation-specific tables in ADR-015. They are not the
> same tables: evaluation produces no checkpoints, so it has neither
> `CHECKPOINTING` nor `RECOVERING`.
>
> Reuse of a completed result depends on the evaluator's declared determinism
> class, not on the fingerprint alone — a stochastic evaluator's result is a
> sample, not the answer (ADR-015 §3).
>
> The invariant that makes this worth having is a **reconciliation** rule, not a
> point-in-time assertion: a node in `EVALUATING` either has a non-terminal
> required run (wait), or has all required results and is reconciled forward to
> `DECIDING`, or is stalled and raises `EvaluationStalled`. Without it a node
> sits in `EVALUATING` forever when the evaluation process dies, because there
> is nothing to time out — and stating it as a bare assertion would flag every
> successful evaluation during the instant between the run finishing and the
> node advancing (ADR-015 §5).

## 1. Evaluation is independent from training

Do not expose:

```python
TrainerBackend.evaluate(...)
```

Evaluation has its own protocol and execution path.

## 2. EvaluationSpec

```python
class EvaluationSpec(BaseModel):
    api_version: str = "xaytune.eval/v1alpha1"

    evaluator: EvaluatorSpec          # one; it may measure many metrics

    dataset: DatasetRef | None
    slices: list[str]

    metadata: dict[str, Any]
```

**One evaluator per spec.** One `EvaluationRun` is then one subject, one spec,
one evaluator -- so one `EvaluatorDeterminism` class -- and one seed and
replicate, and whether its result can be reused is never ambiguous. Several
evaluators are several `EvaluationRun`s in one evaluation cycle:

```text
evaluation cycle
├── run A → lm-eval
├── run B → task evaluator
└── run C → LLM judge
```

rather than one run mixing evaluators with different reproducibility.

**`seed` is deliberately not here.** It belongs to `EvaluationRun`, exactly as
seed and replicate belong to `Run` (ADR-015 §3). If the seed were part of the
spec, then

```text
evaluation with seed 1
evaluation with seed 2
```

would be two different *evaluation specifications* rather than two replicates
of one evaluation — which is precisely the identity error that moving `seed`
off `CandidateSpec` fixed for training. It would also put replicate identity
inside `EvaluationFingerprint`, so two samples of the same stochastic
evaluation could never be recognised as samples of the same thing.

## 3. MetricResult

Never reduce evaluation to `dict[str, float]`.

```python
class MetricResult(BaseModel):
    name: str
    value: float

    sample_count: int | None

    dataset_ref: DatasetRef | None
    slice: str | None

    evaluator_name: str
    evaluator_version: str | None

    seed: int | None          # the seed of the EvaluationRun that produced it

    confidence_interval: tuple[float, float] | None
    standard_error: float | None

    metadata: dict[str, Any]
```

## 4. Judge metrics

For LLM-as-judge:

```python
class JudgeMetadata(BaseModel):
    model: str
    revision: str | None

    prompt_version: str
    rubric_version: str

    temperature: float
    top_p: float | None

    provider: str
```

This metadata contributes to `EvaluationFingerprint`.

## 5. EvaluationResult

```python
class EvaluationResult(FrozenDomainModel):
    id: EvaluationResultId            # = EvaluationId

    evaluation_run_id: EvaluationRunId
    node_id: ExperimentNodeId
    subject: ArtifactRef              # what was measured: the run's subject

    evaluation_fingerprint: str

    metrics: tuple[MetricResult, ...] # at least one
    artifacts: tuple[ArtifactRef, ...]  # reports; producer = this result

    created_at: datetime
```

There is no `status`: a result exists only for a run that `SUCCEEDED`, written
in the same commit, so a result is never pending or failed. There are no
`constraints` yet either; constraint evaluation arrives with the DecisionEngine
(PR-015). Every provenance field must agree with the run -- node, fingerprint,
subject (identity and digest), each metric's evaluator, evaluator version and
seed, and each report's producer -- and the repository refuses a result that
does not. The database independently refuses the column-level subset: a result
whose run, node, evaluation fingerprint, subject artifact id or subject digest
disagrees with its run, a second result for one run, and any edit.

`evaluation_run_id` is required, not convenience. After ADR-015 a node can hold
several `EvaluationRun`s over the same subject and fingerprint — replicates 0,
1, 2 of a stochastic evaluation — and without the back-reference there is no way
to answer *which evaluation execution produced this sample*. For a stochastic
evaluator that is provenance, not bookkeeping: the variance estimate is only
meaningful if each sample can be traced to the run that drew it.

`node_id` is retained for querying, but it is a denormalisation of
`EvaluationRun.node_id` rather than the authoritative link.

## 6. Evaluator protocol

```python
class Evaluator(Protocol):
    descriptor: PluginDescriptor
    determinism: EvaluatorDeterminism

    def capabilities(self) -> CapabilityDocument: ...

    def prepare(
        self,
        subject: ArtifactRef,
        spec: EvaluationSpec,
        context: EvaluationContext,   # run id, seed, replicate, output location
    ) -> EvaluationExecutionSpec: ...
```

`prepare()` is **synchronous and deterministic**, like `TrainerCompiler.compile()`:
it builds a request and never performs the evaluation. A re-issued evaluation is
rebuilt from the record and its digest checked against the one recorded, which
an evaluator reading the clock or the network would break. The spec it returns
is resolved, submitted and observed like any other; execution may be local or
remote. `xaytune.evaluation` holds the contract.

## 7. Initial evaluators

- XaytuneMetricEvaluator
- LMEvalEvaluator
- AgentTaskEvaluator
- CustomPythonEvaluator

## 8. Decision engine

```python
class DecisionEngine:
    async def decide(
        self,
        context: DecisionContext,
    ) -> Decision: ...
```

Inputs:

- objective
- metric results
- constraints
- statistical uncertainty
- prior candidates
- budget
- policy
- incident history
- search/planner proposals

## 9. Decision outcomes

```text
CONTINUE_CURRENT
EVALUATE_MORE
BRANCH
PROMOTE
REJECT
PAUSE
STOP_SUCCEEDED
STOP_FAILED
STOP_BUDGET
```

## 10. Noise-aware decisions

Do not assume:

```text
0.83 > 0.81 => 0.83 is definitively better
```

Decision policy may require:

- minimum effect size
- confidence interval separation
- repeated seeds
- minimum sample count
- no constraint regression

Example:

```python
PromotionPolicy(
    min_improvement=0.01,
    require_confidence=True,
    max_regressions={"latency_ms": 0.05},
)
```

## 11. Candidate promotion

Promotion creates a decision and artifact tag/ref.

It does not move model registry state directly unless an external integration plugin explicitly does so.

## 12. Evaluation failure

Evaluation failure is separate from training failure.

Possible actions:

- retry evaluation
- change evaluator runtime
- mark evaluator unavailable
- pause for approval
- fail decisioning

Do not retrain because an evaluator crashed.
