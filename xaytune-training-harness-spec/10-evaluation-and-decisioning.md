# Evaluation and Decisioning


> **Evaluation has a durable execution lifecycle — see ADR-015.** The
> `EvaluationSpec → EvaluationResult` shape below describes the *scientific*
> contract. Operationally an evaluation is a workload: it queues, fails, gets
> preempted, is retried and can be in flight across a controller restart. It
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
> The invariant that makes this worth having: `ExperimentNode.EVALUATING` must
> correspond to at least one non-terminal `EvaluationRun`. Without it a node
> sits in `EVALUATING` forever when the evaluation process dies, because there
> is nothing to time out.

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

    evaluators: list[EvaluatorSpec]

    dataset: DatasetRef | None
    slices: list[str]

    metadata: dict[str, Any]
```

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
class EvaluationResult(BaseModel):
    id: EvaluationResultId

    evaluation_run_id: EvaluationRunId
    node_id: ExperimentNodeId
    artifact_ref: ArtifactRef

    evaluation_fingerprint: str

    metrics: list[MetricResult]
    constraints: list[ConstraintResult]

    status: EvaluationStatus

    artifacts: list[ArtifactRef]

    created_at: datetime
```

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

    def capabilities(self) -> CapabilityDocument: ...

    async def prepare(
        self,
        artifact: ArtifactRef,
        spec: EvaluationSpec,
    ) -> EvaluationExecutionSpec: ...
```

Execution may be local or remote.

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
