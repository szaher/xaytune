# ADR-016 — The durable experiment stores specs, not implementation objects

## Status
Accepted — 2026-09-21.

Required before PR-005. Determines what can legally appear in a persisted
experiment record.

## Context

The public API example reads:

```python
xaytune.experiment(
    planner=xaytune.RuleBasedPlanner(),
    runtime=xaytune.TrainingHubRuntime(endpoint=..., token=...),
)
```

This is good ergonomics for an embedded Python caller and a bad durable
representation. A live `RuleBasedPlanner` instance cannot be written to SQLite,
survive a controller restart, or be interpreted by a future remote controller
in a different process.

The tension is not hypothetical. ADR-004 requires the controller to be
restartable and attachable, and ADR-005 requires the experiment to be durable.
An experiment whose planner exists only as a Python object in the process that
created it satisfies neither: after a restart the controller knows an
experiment is `ACTIVE` but not how to plan its next node.

It is also a provenance problem. "Which planner produced this decision?" is a
question the record must answer a year later, when that planner class may have
been modified or deleted.

## Decision

### 1. The durable record holds specs

```python
class PlannerSpec(FrozenDomainModel):
    kind: str                   # plugin entry-point name
    version: str                # resolved at bind time, recorded
    config: FrozenDict          # canonical JSON values only

class RuntimeSpec(FrozenDomainModel):
    kind: str
    version: str
    config: FrozenDict
    credentials_ref: SecretRef | None

class ControllerHostSpec(FrozenDomainModel):
    kind: Literal["embedded", "daemon", "remote"]
    config: FrozenDict
```

Each resolves through the plugin registry (ADR-008) to an implementation:

```text
PlannerSpec(kind="rule-based", version="1.2.0")  ──→  RuleBasedPlanner
```

The spec is what persists, what fingerprints, and what appears in provenance.
The instance is a detail of one process's lifetime.

### 2. Credentials are referenced, never stored

`RuntimeSpec.credentials_ref` is a `SecretRef` resolved at bind time from the
environment or a secret store. A token in the example above must never reach
the database — and an experiment record is exactly the kind of thing that gets
copied into a bug report.

### 3. Ergonomic constructors are sugar over specs

```python
xaytune.RuleBasedPlanner(threshold=0.02)
# returns, or is coerced to, PlannerSpec(kind="rule-based",
#                                        config={"threshold": 0.02})
```

Callers keep the readable form. The system stores the spec. A constructor that
cannot produce a spec — because it closes over a live object, a file handle or
a lambda — is not usable in a durable experiment, and the API must reject it at
submission rather than at restart.

**Rejecting late is the failure to avoid.** An experiment that runs for six
hours and then cannot be recovered is worse than one that refuses to start.

### 4. Implementation identity is part of provenance

`version` is resolved and recorded at bind time, not left to float. A decision
made by `rule-based@1.2.0` is not necessarily reproducible under
`rule-based@2.0.0`, and the record has to say which one ran. This is the same
argument ADR-006 makes for compiler and framework versions in
`ExecutionFingerprint`.

## Consequences

- Restart is possible: the controller rebuilds planner and runtime from the
  record.
- A remote controller becomes feasible without redesigning the record.
- Provenance answers "what planned this" with a version, not a class name.
- Callers who want to pass a live object get a clear error at submission, and
  the in-process compatibility path of `19-backward-compatibility.md` is where
  that case is served.
- Plugin authors must make their configuration JSON-expressible. This is a real
  constraint and is the point.

## Acceptance criteria

1. `Experiment` persists `PlannerSpec`, `RuntimeSpec` and `ControllerHostSpec`;
   it holds no live implementation objects.
2. All three spec configs are canonical JSON values, deep-frozen per the value
   contract.
3. Secrets appear only as `SecretRef`; no credential value is ever persisted.
4. `version` is resolved at bind time and recorded.
5. Passing a non-serializable implementation object to a durable experiment
   raises at submission time, naming the offending field.
6. A controller restart rebuilds planner and runtime from the record alone,
   with no state from the creating process.
