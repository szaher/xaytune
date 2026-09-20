# Evaluation and Decisioning

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

    seed: int | None

    metadata: dict[str, Any]
```

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

    seed: int | None

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
    id: EvaluationId

    node_id: ExperimentNodeId
    artifact_ref: ArtifactRef

    evaluation_fingerprint: str

    metrics: list[MetricResult]
    constraints: list[ConstraintResult]

    status: EvaluationStatus

    artifacts: list[ArtifactRef]

    created_at: datetime
```

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
