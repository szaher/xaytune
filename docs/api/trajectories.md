# Trajectories

!!! warning "H05: local, single-host recording"
    A scripted harness run can record a verifiable trajectory beside its
    result. Nothing is stored in the database, streamed to a controller, or
    scored. Liveness, publication proof and reconciliation hold on one host
    only. The decisions are in
    [ADR-021](https://github.com/szaher/xaytune/blob/main/xaytune-training-harness-spec/adrs/ADR-021-trajectory-artifacts-and-provenance.md),
    accepted on 2026-10-12.

A trajectory is **evidence of an execution, not proof that it can be
reproduced**. It records what happened, and declares how much of it was
captured and how much is still available.

## Recording a run

Recording is opt-in. Without it, the compiler emits the H03 spec byte for
byte, with the same fingerprint and request digest.

```python
from xaytune.core.trajectory import TrajectoryRecordingSpec
from xaytune.harness import HarnessCompilationContext, ScriptedHarnessRunner
from xaytune.trajectory import WRITER

context = HarnessCompilationContext(
    ...,
    output_uri="/runs/r1",
    trajectory=TrajectoryRecordingSpec(writer=WRITER),  # bounds, replay level, tiers, retention
)
spec = compiler.compile(harness, context)   # outputs: the result, then the trajectory
result = ScriptedHarnessRunner().run(
    spec, model=model, toolbox=toolbox, authorized_tools=tools,
    trajectory_secrets={"sk-..."},   # frozen when the attempt opens
)
```

```text
/runs/r1/agent-run.json                       the result: the same inode as one attempt's result.json
/runs/r1/trajectory/attempts/<ainst_...>/     one directory per attempt instance, owner-only
    open.json  tail.jsonl  progress.json  chunks/  blobs/{ordinary,restricted}/
    result.json  manifest.staged.json  manifest.json  expiry.jsonl
```

**Each execution is its own attempt instance.** Its id is minted and
persisted before any call. A retry or a crash starts a new one.

**Every start is durable before its call.** A crash leaves a started call
with no finish.

**The commit point.** The final chunk, `result.json` and a staged manifest
are written and verified first. Only then is the result hard-linked to its
declared location, and the staged manifest linked into place.

- Before the commit point, a failure publishes nothing.
- After it, reconciliation completes the seal, or reports the result as
  unattested.

**Replay is checked.** A recording-enabled run returns a published result
only when the attempt that published it is sealed `complete` and verifies.
Otherwise it raises `UnattestedResultError`, and never runs again.

## Verifying

```python
from xaytune.core.trajectory import validate_manifest
from xaytune.trajectory import (
    check_availability, reconcile_trajectories, verify_publication, verify_trajectory,
)

validate_manifest(data)                          # the manifest alone: structure, bindings, chain
verify_trajectory(store, paths, manifest)        # evidence: chunks, events, blobs, result
verify_publication(paths, manifest, location)    # this attempt's link (st_dev, st_ino)
check_availability(store, paths, manifest)       # available, expired, missing or corrupt
reconcile_trajectories(store, location)          # seal what dead writers left
```

| Capture disposition (sealed, fingerprinted) | Availability (observed, never written back) |
|---|---|
| `captured`, `captured-restricted`, `redacted`, `not-provided` | `available`, `expired`, `missing`, `corrupt` |

**Expiry.** `expire_blob` removes a blob only after its tier's retention,
and records the removal first. The sealed manifest and its fingerprint never
change.

Retention is measured from the seal time of the manifest stored in the
attempt. A supplied manifest must be that manifest, field for field. The seal
time is not part of the fingerprint, so a copy with an earlier seal time would
otherwise still validate.

**Missing or corrupt content.** A blob that is absent without an authorized
expiry record is `missing`; one whose bytes changed is `corrupt`. Both are
integrity incidents. A record authorizes an expiry only if it names the
sealed fingerprint, the tier's declared retention and the deadline that
retention gives, and a removal at or after that deadline.

## Replay

```python
from xaytune.harness import verify_replay

verify_replay(store, paths, manifest, spec, verifier=my_verifier)
# equivalent | diverged (sequence, field) | prefix-equivalent | ineligible (reasons)
```

A re-drive serves the recorded model replies and tool outputs, and calls no
model and no host tool. It shows that the harness logic makes the same
decisions, under `xaytune.trajectory-replay-projection/v1`. It never shows
that a provider or a tool would answer the same again.

`recorded-responses` is the minimum level eligible for replay. A trajectory's
writer can't verify its own replay.

Only the manifest sealed at the attempt's `manifest.json` is re-driven, never
a staged one. A complete trajectory also needs its result published by this
attempt at the spec's location. A partial one is checked as a prefix.

## Secrets

**Frozen set.** The registered-secret set is frozen when the attempt opens.
A value from it is never written, and neither is a digest of one.

**Detection.** If a frozen-set value would be written, the run stops before
the commit point, and the attempt is sealed as a sanitized `partial`.

**Late discovery.** `contain_attempt` quarantines the whole attempt, blobs
included, and restricts a published result. It makes no claim about what was
written before.
