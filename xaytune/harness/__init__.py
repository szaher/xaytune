"""Compiling and running agent harness candidates (H03, ADR-019).

```text
AgentHarnessSpec ─resolve_harness_content─▶ ResolvedHarnessContent  (verified by digest)
      └──────── HarnessCompiler.compile(harness, context) ───────────┐
                                                                     ▼
AgentExecutionSpec ── ScriptedHarnessRunner.run(model, toolbox, authorized_tools)
      ▼
HarnessRunResult
```

This package translates a harness into something runnable and runs it
deterministically. It knows nothing of experiments, storage, controllers or
runtimes: no runtime can submit a harness workload yet, because a harness
operation target needs the persistence migration ADR-019 names. Real model
adapters are later work (H08–H11); the scripted model and runner here exist
so execution semantics can be tested exactly.
"""

from __future__ import annotations

from xaytune.harness.compiler import (
    HARNESS_COMPILER_API,
    SCRIPTED_MODEL_NAME,
    SCRIPTED_PROVIDER,
    SCRIPTED_RUNNER,
    TOKEN_COUNTER,
    HarnessCompilationContext,
    HarnessCompiler,
    HarnessCompilerCapabilities,
    ReferenceHarnessCompiler,
    UnsupportedHarnessError,
)
from xaytune.harness.content import (
    ContentSource,
    MappingContentSource,
    UnverifiedContentError,
    resolve_harness_content,
)
from xaytune.harness.replay import replay_prerequisites, verify_replay
from xaytune.harness.runner import (
    RUNNER,
    CancelSignal,
    HarnessMessage,
    HarnessModel,
    HarnessModelReply,
    HarnessModelRequest,
    HarnessRunConflictError,
    HarnessRunRefusedError,
    HarnessToolbox,
    HarnessToolCall,
    HarnessToolView,
    ScriptedHarnessModel,
    ScriptedHarnessRunner,
    ScriptExhaustedError,
    UnattestedResultError,
    run_scripted_harness,
)

__all__ = [
    "HARNESS_COMPILER_API",
    "RUNNER",
    "SCRIPTED_MODEL_NAME",
    "SCRIPTED_PROVIDER",
    "SCRIPTED_RUNNER",
    "TOKEN_COUNTER",
    "CancelSignal",
    "ContentSource",
    "HarnessCompilationContext",
    "HarnessCompiler",
    "HarnessCompilerCapabilities",
    "HarnessMessage",
    "HarnessModel",
    "HarnessModelReply",
    "HarnessModelRequest",
    "HarnessRunConflictError",
    "HarnessRunRefusedError",
    "HarnessToolCall",
    "HarnessToolView",
    "HarnessToolbox",
    "MappingContentSource",
    "ReferenceHarnessCompiler",
    "ScriptExhaustedError",
    "ScriptedHarnessModel",
    "ScriptedHarnessRunner",
    "UnattestedResultError",
    "UnsupportedHarnessError",
    "UnverifiedContentError",
    "replay_prerequisites",
    "resolve_harness_content",
    "run_scripted_harness",
    "verify_replay",
]
