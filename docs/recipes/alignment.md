# Alignment

The `align` recipe aligns a language model with human preferences using reinforcement learning from human feedback (RLHF) and related methods. xaytune supports six alignment algorithms.

## Methods

| Method | Full Name | Data Required | Description |
|--------|-----------|---------------|-------------|
| `dpo` | Direct Preference Optimization | Preference pairs | Offline, no reward model needed |
| `grpo` | Group Relative Policy Optimization | Prompts + reward fn | Online, group-based advantage estimation |
| `orpo` | Odds Ratio Preference Optimization | Preference pairs | Combined SFT + preference, single stage |
| `simpo` | Simple Preference Optimization | Preference pairs | Reference-free variant of DPO |
| `ppo` | Proximal Policy Optimization | Prompts + reward fn | Classic RLHF with reward model |
| `reinforce` | REINFORCE | Prompts + reward fn | Policy gradient with reward signal |

## Python API

```python
import xaytune

# DPO alignment
state = xaytune.align(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dataset="data/preferences.jsonl",
    method="dpo",
    format="preference",
    num_epochs=1,
    learning_rate=5e-6,
)

# GRPO alignment
state = xaytune.align(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dataset="data/prompts.jsonl",
    method="grpo",
    num_epochs=1,
    learning_rate=5e-6,
)
```

### Function Signature

```python
def align(
    *,
    config: TrainConfig | None = None,
    model: str | None = None,
    dataset: str | None = None,
    method: str = "dpo",
    format: str = "preference",
    num_epochs: int = 1,
    learning_rate: float = 5e-6,
    batch_size: int = 4,
    **kwargs,
) -> TrainState:
```

- **config** -- A full `TrainConfig` object. If provided, all other arguments are ignored.
- **model** -- Model name or path.
- **dataset** -- Path to preference or prompt data.
- **method** -- Alignment algorithm: `"dpo"`, `"grpo"`, `"orpo"`, `"simpo"`, `"ppo"`, or `"reinforce"`.
- **format** -- Data format (default: `"preference"`).
- **num_epochs** -- Number of training epochs (default: 1).
- **learning_rate** -- Learning rate (default: 5e-6, lower than fine-tuning).
- **batch_size** -- Per-device batch size (default: 4).
- **\*\*kwargs** -- Additional `TrainerConfig` fields.

## YAML Config Examples

### DPO

```yaml
recipe: align
method: dpo

model:
  name: meta-llama/Llama-3.1-8B-Instruct

data:
  path: data/preferences.jsonl
  format: preference

trainer:
  batch_size: 4
  gradient_accumulation: 4
  learning_rate: 5e-6
  num_epochs: 1
  mixed_precision: bf16

output:
  dir: output/dpo-align
```

### GRPO

```yaml
recipe: align
method: grpo

model:
  name: meta-llama/Llama-3.1-8B-Instruct

data:
  path: data/prompts.jsonl
  format: text

trainer:
  batch_size: 4
  gradient_accumulation: 4
  learning_rate: 5e-6
  num_epochs: 1

output:
  dir: output/grpo-align
```

### PPO

```yaml
recipe: align
method: ppo

model:
  name: meta-llama/Llama-3.1-8B-Instruct

data:
  path: data/prompts.jsonl
  format: text

trainer:
  batch_size: 4
  learning_rate: 1e-5
  num_epochs: 1

output:
  dir: output/ppo-align
```

## Preference Data Format

For offline methods (DPO, ORPO, SimPO), prepare data as preference pairs with `chosen` and `rejected` responses:

```json
{
  "prompt": "Explain quantum computing in simple terms.",
  "chosen": "Quantum computing uses quantum bits (qubits) that can be...",
  "rejected": "Quantum computing is really complicated and..."
}
```

For online methods (GRPO, PPO, REINFORCE), provide prompts. The model generates responses during training, and a reward function scores them.

## Custom Reward Functions

Register custom reward functions for online alignment methods:

```python
from xaytune.recipes.align.rewards import reward_registry

@reward_registry.register("length_reward")
def length_reward(prompt: str, response: str) -> float:
    """Reward longer, more detailed responses."""
    return min(len(response.split()) / 100, 1.0)

@reward_registry.register("format_reward")
def format_reward(prompt: str, response: str) -> float:
    """Reward responses that follow a specific format."""
    score = 0.0
    if response.startswith("Answer:"):
        score += 0.5
    if "\n" in response:
        score += 0.5
    return score
```

!!! info "Default reward"
    xaytune includes a `default` reward function (returns 0.0). You should register your own reward function for meaningful alignment results.

## Choosing an Alignment Method

- **DPO** -- Simplest to set up. Requires pre-collected preference pairs. No reward model needed. Good default choice.
- **GRPO** -- Online method with group-based advantage estimation. Works well when you have a reward function but not preference data.
- **ORPO** -- Combines SFT and preference optimization in a single training stage. Can be more efficient than separate SFT + DPO.
- **SimPO** -- Reference-free DPO variant. Avoids the need to keep a reference model in memory, reducing VRAM usage.
- **PPO** -- Classic RLHF. Most flexible but also most complex to tune. Requires a reward model or function.
- **REINFORCE** -- Simpler than PPO. Good for straightforward reward signals.
