# ADR-008 — Plugin interfaces and capabilities are versioned

## Status
Accepted — 2026-09-22. **This gates band C, including PR-010, which may now
start.**

Accepted as written rather than expanded: the decision was already small,
already consistent with what band C shipped, and the parts that were merely
aspirational have been implemented rather than reworded.

What was true at acceptance, and what was not:

```text
every plugin provides a PluginDescriptor      TrainerCompiler and RuntimeBackend both declare one
capabilities are versioned                    CapabilityDocument.schema_version, PluginDescriptor.capabilities_schema
unknown API versions fail closed              NOT IMPLEMENTED -- api_version was declared and never read
```

The third clause is the one that makes this an ABI contract rather than a
naming convention, so it was implemented as part of acceptance:
`require_supported_plugin()` refuses any descriptor outside
`PLUGIN_API_VERSIONS`, and `LocalRuntime` applies it to the compiler that
produced a plan, recording the refusal as a rejected operation rather than
discovering the mismatch inside a worker.

**Known gap, deliberately not closed here.** The legacy entry-point loader in
`xaytune/plugins.py` — recipes, models, formats, metrics — predates this ADR,
never sees a `PluginDescriptor`, and swallows load failures with
`logger.warning`. It therefore fails *open*, which is the opposite of this
decision. It is out of scope for band C, which governs the control-plane
plugin surface (compilers and runtimes), and is listed for extension in the
pre-implementation review. Recorded here so the gap is a known debt rather
than a contradiction nobody noticed.

## Decision

Every plugin provides a `PluginDescriptor`.

Capabilities use a versioned parameterized schema.

Unknown major plugin API versions fail closed with actionable errors.

## Rationale

Trainer/runtime APIs evolve quickly. Boolean support flags and unversioned entry points are insufficient.
