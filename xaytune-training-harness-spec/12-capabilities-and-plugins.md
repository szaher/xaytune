# Capabilities and Plugin ABI

## 1. Why boolean capabilities are insufficient

This is too weak:

```python
CapabilitySet(
    distributed=True,
    async_checkpoint=True,
)
```

Capabilities are parameterized and versioned.

## 2. CapabilityDocument

```python
class CapabilityDocument(BaseModel):
    schema_version: str = "xaytune.capabilities/v1alpha1"

    precision: PrecisionCapabilities | None
    distributed: DistributedCapabilities | None
    checkpoint: CheckpointCapabilities | None
    elasticity: ElasticityCapabilities | None
    resilience: ResilienceCapabilities | None
    agent_rollout: AgentRolloutCapabilities | None
    algorithms: AlgorithmCapabilities | None

    extensions: dict[str, Any]
```

## 3. Example

```yaml
schemaVersion: xaytune.capabilities/v1alpha1

precision:
  supported: [fp32, fp16, bf16]

distributed:
  strategies: [ddp, fsdp]
  minWorkers: 1
  maxWorkers: 64

checkpoint:
  formats: [torch-dcp]
  async: true
  reshardable: true
  atomicCommit: true

elasticity:
  supported: true
  minWorkers: 2
  maxWorkers: 16
  membershipChange: restart

resilience:
  perStep:
    supported: true
    provider: torchft
    providerVersion: ">=0.1"

agentRollout:
  stateful: true
  asynchronous: true

algorithms:
  sft: true
  dpo: true
  grpo: true
```

## 4. Capability requirements

Training/evaluation specs declare requirements.

```python
class CapabilityRequirements(BaseModel):
    precision: str | None
    distributed_strategy: str | None
    checkpoint_resharding: bool | None
    per_step_recovery: bool | None
    stateful_rollouts: bool | None
```

## 5. CapabilityResolver

Inputs:

- CandidateSpec
- TrainerCompiler capabilities
- RuntimeBackend capabilities
- ResilienceProvider capabilities
- CheckpointCodec capabilities
- organization policy

Output:

```python
ResolvedExecutionPlan
```

or actionable incompatibility report.

## 6. Plugin descriptor

```python
class PluginDescriptor(BaseModel):
    api_version: str
    name: str
    plugin_version: str
    provider: str

    xaytune_version: str
    capabilities_schema: str

    metadata: dict[str, Any]
```

## 7. Plugin groups

Keep existing plugin groups and add:

```text
xaytune.trainer_compilers
xaytune.runtime_backends
xaytune.resilience_providers
xaytune.checkpoint_codecs
xaytune.checkpoint_stores
xaytune.evaluators
xaytune.planners
xaytune.search_providers
xaytune.event_sinks
```

## 8. ABI compatibility

Each plugin load must validate:

- supported plugin API version
- Xaytune version range
- capability schema version

Unknown major API version must fail with a clear message.

## 9. Optional dependencies

Recommended extras:

```text
xaytune[native]
xaytune[trl]
xaytune[torchtune]
xaytune[verl]
xaytune[ray]
xaytune[torchft]
xaytune[mlflow]
xaytune[wandb]
xaytune[training-hub]
xaytune[agent]
xaytune[eval]
```

Core dependency target:

```text
pydantic
PyYAML
typing-extensions if required
```

CLI may additionally use Rich/Typer.

## 10. Import rule

This must work:

```python
import xaytune
from xaytune import Experiment, Objective
```

without importing PyTorch or Transformers.

Heavy integrations load lazily.

## 11. Event sink contract (PR-009a)

`xaytune.core.sinks.EventSink` is the `xaytune.event_sinks` plugin boundary:
`descriptor: PluginDescriptor` and `async consume(event: DomainEvent) -> None`.
Sinks receive committed durable events via the outbox, with at-least-once
delivery and consumer deduplication by event ID. Delivery failures are isolated
from training; external observability tools are never sources of truth.

No collector/exporter capabilities are advertised merely because the contracts
exist. `ObservabilitySpec` expresses requested observation policy; a future
compiler/runtime must report unsupported profiler or tracing requests rather
than silently claim to have honored them. PR-009a does not install any sink,
collector, profiler or tracing SDK.
