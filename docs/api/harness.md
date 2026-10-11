# Agent Harness

!!! warning "H03: compile and run in-process only"
    A harness can be compiled and run deterministically against a scripted
    model. It cannot yet be submitted to a controller or a runtime: that needs
    a harness operation target, and the persistence migration
    [ADR-019](https://github.com/szaher/xaytune/blob/main/xaytune-training-harness-spec/adrs/ADR-019-harness-compilation-and-execution.md)
    names. ADR-019's E1–E10 are accepted; that migration is only proposed.
    Cancellation, restart and idempotency hold in-process only. Real model
    adapters, benchmarks, trajectories and scoring come later.

```text
AgentHarnessSpec ─resolve_harness_content(source)─▶ ResolvedHarnessContent   digests verified
      └──── HarnessCompiler.compile(harness, context) ────┐
                                                          ▼
AgentExecutionSpec                                        xaytune.agent-execution/v1alpha1
      ▼ ScriptedHarnessRunner.run(spec, model, toolbox, authorized_tools)
HarnessRunResult                                          xaytune.agent-run-result/v1alpha1
```

## Compile

```python
from xaytune.core.agent_execution import HarnessTaskInput
from xaytune.harness import (
    HarnessCompilationContext, MappingContentSource, ReferenceHarnessCompiler,
    resolve_harness_content,
)

content = resolve_harness_content(harness, MappingContentSource(blobs))  # verified, or refused
spec = ReferenceHarnessCompiler().compile(
    harness,
    HarnessCompilationContext(
        task=HarnessTaskInput(task_id="support/refund-001", variables={"ticket": "..."}),
        seed=7,
        replicate=0,
        authorized_tools=("kb.search",),   # resolved by the host, never by the harness
        content=content,
        output_uri="/runs/r1",
    ),
)
```

- **Resolution comes first.** Every content reference the model can reach is
  fetched and checked before compilation. Reachable content is the system
  prompt, the task template, and the description and contract of each enabled
  tool. A reference is refused if:
  - its bytes don't have the declared SHA-256;
  - its media type isn't `text/*`;
  - it isn't valid UTF-8.

  Refusals name the role of the content, never the locator. H03 ships only an
  in-memory source.
- **Compilation is deterministic.** It calls no model, runs no tool and reads
  no file or network. The same harness in the same context always compiles to
  the same spec.
- **Refusals carry every reason.** `supports()` and `UnsupportedHarnessError`
  list every reason, not only the first. The reference compiler refuses:
  - any model other than the scripted one (`provider="xaytune"`,
    `model="scripted"`, pinned by `artifact_digest`);
  - non-empty `provider_parameters`;
  - `jinja2` templates, and `python-format` fields that are not plain
    `{name}`;
  - a missing task template;
  - tool schemas outside xaytune's agent-schema subset;
  - unauthorized tools;
  - task variables that don't match the template exactly;
  - an `output_uri` that is not an absolute local directory.

## Three identities

| | Answers | Changes with |
|---|---|---|
| `spec.harness_fingerprint` | Same harness proposition? | Behavior fields only (ADR-018) |
| `agent_execution_fingerprint(spec)` | Same way of running it? | Also seed, replicate, task, output location, resources, compiler release |
| `agent_run_request_digest(spec)` | Literally the same request? | Any byte of the spec, including metadata |

## Run

```python
from xaytune.harness import HarnessToolbox, ScriptedHarnessModel, ScriptedHarnessRunner

result = ScriptedHarnessRunner().run(
    spec,
    model=ScriptedHarnessModel(script),          # exactly this class; its script must match the pin
    toolbox=HarnessToolbox({("kb.search", "1.2.0"): search}),
    authorized_tools={"kb.search"},              # checked again here
    cancel=threading.Event(),                    # optional
)
result.termination   # "final-answer", "max-turns", "token-budget", "cancelled", ...
```

The runner applies every behavior field of the harness:

- it renders the system prompt and the task template;
- it shows the model only the enabled tools, with their descriptions and
  contracts, and refuses calls to any other tool without running them;
- it validates tool arguments against each tool's schema;
- it selects context by `full-history` or `recent-turns`, and enforces
  `max_context_tokens` (`fail`, or `truncate-oldest` by whole turns);
- it truncates tool results and output to their limits, and cuts output at
  stop sequences;
- it stops on `max_turns`, `max_total_tokens` and `max_tool_calls`, and on the
  declared `terminate_on`.

Tokens are whitespace-separated words (`xaytune.whitespace-tokens/v1`), which
is a named rule and not a provider tokenizer.

Before anything executes, the runner refuses every requirement it does not
implement, with reasons: a container, secrets, dependencies, inputs,
arguments, environment, resources, other capabilities, or another entrypoint.

**Only the trusted scripted model runs.** The `model` must be exactly
`ScriptedHarnessModel`, holding the pinned script. A subclass is refused, and
so is any other object that reports the right digest.

**One result, published once.** The result is written atomically to
`<output_uri>/agent-run.json` with a no-overwrite link.

- **Same request again:** the run returns that result without executing,
  but only after checking that it parses consistently and agrees with the
  spec on every attribution field.
- **Different request, or an inconsistent record:** `HarnessRunConflictError`.
- **Concurrent runs on one host:** a file lock serializes them, so one
  executes and the rest replay its result.
- **Crash before publishing:** nothing is left behind, and the rerun is
  identical.
- **Cancelled run:** its result is terminal, like any other.

This is **not exactly-once**. A run on another host, or a rerun after a
crash, repeats the model and tool calls. That is harmless with a
deterministic script and pure tools, and is not a guarantee to build
production adapters on.

The result is attributable: it carries the request digest, the execution and
harness fingerprints, the compiler, runner and model identities, the task,
seed and replicate, and the counts. It is not a score.
