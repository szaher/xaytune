# Agent Fine-Tuning

Train models to use tools, follow multi-step reasoning, and complete tasks. xaytune provides the full agent training pipeline: data formats, loss masking, alignment rewards, and evaluation metrics.

---

## Quick Start

```python
import xaytune

# Fine-tune on tool-use conversations
state = xaytune.finetune(
    model="meta-llama/Llama-3.1-8B",
    dataset="data/agent_traces.jsonl",
    method="lora",
    format="function_calling",
    num_epochs=3,
)
```

Just set `format="function_calling"` (or `"react"`, `"trajectory"`, `"multi_agent"`) and xaytune handles loss masking automatically — the model learns to generate tool calls and reasoning, not to predict user prompts or tool outputs.

---

## Data Formats

xaytune supports four agent data formats. Each converts raw data into an intermediate `AgentMessage` representation with per-message `trainable` flags.

### function_calling (OpenAI-compatible)

The most common format. Matches the OpenAI Chat Completions API with `tool_calls` and `tool` messages.

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in London?"},
    {"role": "assistant", "content": null, "tool_calls": [
      {"id": "call_1", "type": "function", "function": {
        "name": "get_weather",
        "arguments": "{\"city\": \"London\"}"
      }}
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 18}"},
    {"role": "assistant", "content": "It's 18°C in London."}
  ]
}
```

**Loss masking:** Only `assistant` messages are trainable. `system`, `user`, and `tool` messages are masked.

### react (ReAct traces)

Thought → Action → Observation loops for reasoning agents.

```json
{
  "task": "Find the population of France",
  "steps": [
    {
      "thought": "I need to search for this.",
      "action": "search",
      "action_input": "France population 2024",
      "observation": "68 million"
    },
    {
      "thought": "I have the answer.",
      "action": "finish",
      "action_input": "68 million"
    }
  ]
}
```

**Loss masking:** `thought`, `action`, `action_input` are trainable. `observation` is masked.

### trajectory (multi-step sessions)

Full agent sessions with tool calls — ideal for code agents, debugging workflows, and multi-step tasks.

```json
{
  "system": "You are a coding assistant.",
  "goal": "Create a hello world script",
  "turns": [
    {"role": "assistant", "content": "Creating file.", "tool_calls": [
      {"name": "write_file", "arguments": {"path": "hello.py", "content": "print('hello')"}}
    ]},
    {"role": "tool", "content": "File written."},
    {"role": "assistant", "content": "Done!"}
  ]
}
```

### multi_agent (collaborative agents)

Multiple named agents working together. Each turn has an `agent` field identifying which agent is acting.

```json
{
  "system": "You are coordinating a research team.",
  "goal": "Research and write about quantum computing",
  "turns": [
    {"agent": "researcher", "role": "assistant", "content": "Searching...", "tool_calls": [
      {"name": "search", "arguments": {"q": "quantum computing"}}
    ]},
    {"role": "tool", "content": "Found 15 papers..."},
    {"agent": "writer", "role": "assistant", "content": "Here's a summary..."}
  ]
}
```

---

## Loss Masking

The agent tokenizer applies per-token loss masking automatically. Each message is tokenized individually to track exact token boundaries:

- **Trainable** (labels = token IDs): assistant messages — the model learns to generate these
- **Masked** (labels = -100): system, user, tool messages — the model attends to these but doesn't predict them

This ensures the model learns *what to say* and *which tools to call*, without wasting capacity predicting user prompts or tool outputs.

```python
from xaytune.data.agent_formats import format_function_calling
from xaytune.data.agent_tokenizer import tokenize_agent_dataset

messages = format_function_calling(sample)
tokenized = tokenize_agent_dataset([messages], tokenizer, max_seq_length=2048)
# tokenized[0]["labels"] has -100 for masked tokens
```

---

## Alignment with Agent Rewards

After SFT, align your agent with GRPO/PPO using agent-specific reward functions.

### Built-in Rewards

| Reward | What it scores | Range |
|--------|---------------|-------|
| `tool_use_quality` | Did the model call the right tools with correct parameters? | 0–1 |
| `task_completion` | Did the agent finish the task? | 0–1 |
| `efficiency` | Fewer tool calls = higher score | 0–1 |
| `agent_composite` | Weighted combination (default: 40% quality, 40% completion, 20% efficiency) | 0–1 |

### Usage in Training Config

```yaml
recipe: align
method: grpo
online_rl:
  enabled: true
  reward_name: agent_composite
  reward_kwargs:
    expected_tools: ["search", "calculator"]
    success_markers: ["Done", "Task complete"]
    max_steps: 5
```

### Custom Tool Call Parsers

By default, rewards parse `<tool_call>` tags. Pass a custom parser for other formats:

```python
from xaytune.recipes.align.agent_rewards import tool_use_quality_reward

def my_parser(response):
    # Parse your custom format
    return [ParsedToolCall(name="...", arguments={...})]

score = tool_use_quality_reward(
    prompt, response,
    expected_tools=["search"],
    parser=my_parser,
)
```

---

## Evaluation

Evaluate agent performance on a dataset of prompt-response pairs.

```python
from xaytune.eval.agent_metrics import evaluate_agent

results = evaluate_agent(
    responses=[
        {"prompt": "Search for cats", "response": "<tool_call>..."},
        {"prompt": "Calculate 2+2", "response": "<tool_call>..."},
    ],
    expected_tools=["search", "calculator"],
    success_markers=["Done"],
    max_steps=5,
)

print(results)
# {
#   'tool_use_accuracy': 0.85,
#   'task_success_rate': 0.90,
#   'step_efficiency': 0.75,
#   'error_recovery_rate': 0.60,
# }
```

### Metrics

| Metric | Description |
|--------|-------------|
| `tool_use_accuracy` | Fraction of samples where the correct tools were called |
| `task_success_rate` | Fraction of samples that completed the task |
| `step_efficiency` | Average efficiency score (fewer calls = higher) |
| `error_recovery_rate` | Fraction of samples that recovered after tool errors |

---

## Synthetic Agent Data Generation

Generate training data for agent fine-tuning without manually collecting trajectories.

### agent_distill — Generate from a Topic

Give the LLM a topic and tool definitions, get back function_calling conversations:

```python
from xaytune.data.prep import generate

result = generate(
    mode="agent_distill",
    topic="customer support with order lookup",
    tools=[
        {"function": {
            "name": "lookup_order",
            "description": "Look up order by ID",
            "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}}
        }}
    ],
    n=100,
    format="function_calling",
    model="gpt-4o-mini",
)
result.save("data/agent_traces.jsonl")
```

### agent_augment — Generate Variations

Create variations of existing agent conversations:

```python
result = generate(
    mode="agent_augment",
    seed="data/agent_traces.jsonl",
    n=200,
    format="function_calling",
    model="gpt-4o-mini",
)
result.save("data/agent_augmented.jsonl")
```

---

## CLI

Agent fine-tuning uses the standard `xaytune train` command with an agent format:

```bash
xaytune train --config agent_config.yaml
```

Where `agent_config.yaml`:

```yaml
recipe: finetune
method: lora
model:
  name: meta-llama/Llama-3.1-8B
data:
  path: data/agent_traces.jsonl
  format: function_calling
trainer:
  num_epochs: 3
  learning_rate: 2e-4
  batch_size: 4
```

---

## Full Pipeline Example

The complete agent training pipeline:

1. **Prepare data** — collect or generate agent trajectories
2. **SFT** — fine-tune on trajectories with `format="function_calling"`
3. **Align** — improve tool use with `reward_name="agent_composite"` and GRPO
4. **Evaluate** — score with `evaluate_agent()` on held-out data
5. **Export** — merge adapters and deploy

See [11_agent_finetuning.ipynb](https://github.com/szaher/xaytune/blob/main/examples/11_agent_finetuning.ipynb) for a hands-on walkthrough covering all steps.
