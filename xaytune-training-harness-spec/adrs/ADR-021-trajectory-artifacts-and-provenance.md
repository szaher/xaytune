# ADR-021 — Trajectory artifacts and provenance

## Status

Accepted — 2026-10-12. Proposed 2026-10-11 with H05, and revised the same
day after architecture reviews of
[PR #77](https://github.com/szaher/xaytune/pull/77) at `cb8fb1d`, `2dbc68b`
and `40144cf`. PR #77 merged it as `Proposed` (`c07c4af`).

The project owner approved decisions **T1–T11** in the architecture review of
PR #77. This status records that architectural acceptance only. It does not
approve the implementation in
[PR #78](https://github.com/szaher/xaytune/pull/78), which passes its own code
review and CI before it merges.

**Wording aligned at acceptance** (no decision changed), after the
implementation review of PR #78:
- T3's causality: a context change names the event it follows, and the model
  call it prepared names the context change;
- F17: changed harness logic diverges; a changed spec is `ineligible`.

**Revision 2 changes:**
- a persisted attempt-instance identity (T3);
- the lifecycle and crash-recovery contract (T4);
- capture disposition kept apart from current availability (T6);
- replay prerequisites and a versioned comparison projection, with no
  assumption that host tools are pure (T7);
- the recording extension reconciled with H03's single-output contract (T9);
- the four review questions decided (§"Decisions on the first review's
  questions");
- held-out references distinguished from a legitimately equal answer (T8);
- failure scenarios and negative acceptance tests.

**Revision 3 changes** (second review, at `2dbc68b`):
- result publication is the commit point, with everything staged and
  verified before it; reconciliation after it; replay checks the
  trajectory (T4);
- per-attempt blob storage, and the deletion rules for any future shared
  blobs (T6);
- the registered-secret set is frozen at open, and a late discovery
  contains the whole attempt with no retrospective claim (T8);
- manifest validation separated from trajectory verification (T5);
- the link-proof conditions (T4).

**Revision 4 changes** (third review, at `40144cf`):
- pre-commit evidence verification (`verify_trajectory`) is separated from
  post-commit publication verification (`verify_publication`), with tests
  for both (T4, T5);
- the secret-digest guarantee is limited to the frozen set; containment
  creates no new digest (T8);
- sealed evidence is immutable, and open-state and retention metadata are
  the only mutable state (T1).

Builds on:
- ADR-018 (accepted), whose question 7 left the chunk format, retention and
  replay minimums open for H05;
- ADR-019 (E1–E10 accepted), whose E7 makes H03's telemetry one terminal
  result and leaves the turn-by-turn trajectory to H05, and whose E8 states
  H03's idempotency and replay rules;
- ADR-020 (accepted), whose task fingerprints and verified assignments a
  trajectory is attributed to;
- ADR-014 (accepted), whose sequencing rules this ADR reuses.

## Context

H03 can run a harness on one task and publish one `HarnessRunResult`. That
result says how the run ended and what it counted. It does not say what
happened in between: which model calls were made, what each was shown and
answered, which tool calls were made and with what result, when context was
truncated, and where an error occurred.

Chapter 23 §8 asks for exactly that record: a versioned trajectory manifest
that references immutable, bounded event chunks, outputs and optional replay
inputs. Core rows keep typed metadata, digests and references, never unlimited
raw contexts.

**One distinction governs every decision below.** A recorded trajectory is
**evidence of an execution**. It is not proof that the execution can be
reproduced exactly:
- a provider can answer differently to the same request;
- a tool can observe a world that has changed;
- redacted or expired content can't be replayed at all.

The contracts here record what happened and say exactly how much of it was
captured, and how much is still available. They never claim more.

## The boundary H05 stops at

H05 adds no scoring, no durable controller integration and no database
migration.

- **The durable path is still ADR-019's.** A harness can't yet be submitted
  to a controller or runtime. That needs ADR-019's M1–M6 migration, which is
  proposed and not accepted. So H05 writes trajectories from the in-process
  runner, beside its result, through a storage protocol, not into
  repository rows.
- **No streaming telemetry.** `xaytune.agent-telemetry/v1alpha1` stays a
  single terminal result (ADR-019 E7). A trajectory is an artifact the run
  publishes, not a stream a controller consumes.
- **Scoring is H06's.** A trajectory is a subject that evaluation will
  measure. H05 defines nothing about measuring it.
- **Recovery is local.** H05 reconciles abandoned attempts on one host
  (T4). Liveness across hosts needs the durable path's leases.

## Decisions

### T1. Artifacts: blobs, a write-ahead tail, chunks and a manifest

A trajectory is made of four kinds of object:

```text
TrajectoryManifest      xaytune.agent-trajectory/v1alpha1     sealed once, immutable
  chunks: [ChunkRef(index, first_sequence, last_sequence, digest, previous, bytes, events)]
      ↓ by digest
TrajectoryChunk         canonical JSON Lines of TrajectoryEvent  immutable, bounded
      ↑ rolled from
write-ahead tail        the open attempt's newest events          append-only, fsynced
      ↓ events reference, by digest
content blobs           messages, outputs, tool arguments and results   immutable
```

- **Events are small and typed.** Each sealed event lives in exactly one
  chunk.
- **Content is never inline in an event.** Messages, model outputs, tool
  arguments and tool results are blobs, referenced by digest, size, media
  type and tier. That keeps chunks bounded, and lets redaction and access
  control act per blob (T8).
- **Sealed evidence is immutable; two kinds of metadata are not.** Every
  object of the evidence never changes once written: blobs, chunks,
  `result.json`, the staged manifest and the sealed manifest. Mutable state
  is kept separately, and is never covered by a fingerprint:
  - **open-state**, only while the attempt is open: the tail is
    append-only, and is rolled into immutable chunks when the chunk bounds
    are reached and when the attempt is sealed. `progress.json` is replaced
    atomically, and only ever advances (T4);
  - **retention metadata**: the append-only `expiry.jsonl` (T6). It records
    removals beside the evidence and never edits it.
- **The manifest is the only entry point** to a sealed trajectory. Nothing
  reads a chunk except through a manifest that names its digest.

**Chunk bounds are per run, and validated.** The recording spec (T9) declares
`max_events` and `max_bytes`. The compiler supplies defaults of 1,000 events
and 1 MiB. Values outside these ranges are refused at compile time:
- `max_events` from 1 to 10,000;
- `max_bytes` from 4 KiB to 16 MiB.

A single event larger than `max_bytes` is refused when it is written. That
can only be a writer fault, because content lives in blobs. The bounds are
operational, so they enter execution identity, never scientific identity.

### T2. A closed, versioned event schema

`TrajectoryEvent` is a discriminated union, closed in v1. A reader refuses an
unknown `kind` or an unknown field. Adding a kind is a new schema version,
never a silent extension.

| Kind | Records |
|---|---|
| `run-started` | the attribution snapshot (T3), the effective tool set, context and limit configuration by digest, and the writer and runner identities |
| `model-call-started` | the request: the messages shown, by content digest; the tools shown, by digest; and the context selection applied |
| `model-call-finished` | the outcome (`replied`, `failed` or `cancelled`), the output content, any tool calls requested, tokens with their counter, latency, and any stop-sequence or truncation flags |
| `tool-call-started` | the tool id and version, and the arguments, by content digest |
| `tool-call-finished` | the outcome (`executed`, `refused` or `failed`), the result content, the truncation flag, and a sanitized refusal or failure reason |
| `context-changed` | a selection, truncation or (later) compaction: the before and after digests of the message list, the turns dropped, and the rule that applied |
| `error-observed` | a sanitized error class and code. Never a provider's body, a stack trace or an input value |
| `run-terminated` | the termination reason (H03's `TerminationReason`), whether it was a completion, the answer's content reference, and the digest of the `HarnessRunResult` about to be published (T4) |

- **Start and finish are separate events**, and a start is durable before
  the call it announces (T4). A crash between them is therefore visible as a
  started call with no finish. That is the same "intended before called"
  evidence PR-032 records for agent invocations.
- **v1 has no delegation, handoff, memory or retrieval events.** H03's runner
  has none of these behaviors. Each later kind is added with the adapter or
  feature that produces it, under a new schema version.

### T3. Identity: four identities, kept apart

A retry, a crash and a replicate are easy to confuse, so each question has
its own identity:

| Identity | Answers | Source | In H05 |
|---|---|---|---|
| Request digest (`agent_run_request_digest`) | Is this literally the same request? | the spec (ADR-019 E4) | present; shared by every execution of one spec |
| Replicate identity | Which sample of the candidate is this? | seed and replicate in the spec | present; shared by every attempt at that replicate |
| Durable attempt ordinal | Which numbered `RunAttempt` did the controller assign? | the durable path's repository | absent until ADR-019's durable path; then optional attribution |
| **Attempt instance** (`AttemptInstanceId`) | Which single execution produced this record? | minted by the writer when the attempt opens | always present; unique; persisted first |

**The attempt instance is the trajectory's identity.**
- An `AttemptInstanceId` is a typed, ULID-shaped id (prefix `ainst_`, 80
  random bits), minted once by the writer, from the project's existing
  `TypedId` generator.
- It is persisted **before anything else happens**: the attempt's open
  record is created with no-overwrite semantics (T4) before the first model
  call. An id that already exists is refused, never reused.
- `trajectory_id` is the attempt instance id. Every event, chunk and
  manifest of the trajectory carries it.

**Why it is minted, not derived.** A derived id, such as request digest plus
ordinal, is ambiguous exactly when it matters:
- after a crash, the process that restarts can't know whether the previous
  execution recorded its ordinal;
- two hosts can execute one request at once (ADR-019 E8: H03 is not
  exactly-once);
- a replicate is executed by several attempts when attempts fail.

A minted, persisted id is unique however those happen. The other three
identities are recorded beside it as **attribution**, never combined into
it.

**Attribution** records:
- the request digest;
- the execution and harness fingerprints;
- the task: H04's id, revision and `task_fingerprint`, or H03's
  `HarnessTaskInput` id until H06 wires H04 tasks into the spec;
- the seed and replicate;
- when one exists, the verified `TaskAssignment` cohort;
- when the durable path exists, the experiment, node, run and attempt ids
  and the attempt ordinal.

**Sequence.** Every event has a `sequence`: gapless, from 0, within its
attempt instance.
- One writer per attempt instance assigns it, which is ADR-014 §1a's
  supervisor rule.
- `sequence` is the order. `emitted_at` is advisory only.

**Stable event ids.** `event_id` is derived from `(trajectory_id, sequence)`.
So a redelivered or re-read event has the same id. Two events with one id and
different content are a conflict, and are refused.

**Causality.**
- Every event except `run-started` names a `parent`, the sequence of an
  earlier event:
  - a finish names its start;
  - a tool call names the model call that requested it;
  - a context change names the event it follows: the run's start, or the
    finish that preceded it;
  - a model call names the event it follows, which is the context change
    when one prepared it;
  - `run-terminated` names the event that ended the run.
- Parents always have lower sequences, so the causal graph is acyclic by
  construction.
- A reader verifies every parent: it exists, it comes earlier, and its kind
  is allowed.

**A retry is a new attempt instance, never a new sample** (chapter 23 §9).
Two attempt instances at one replicate are two attempts at one sample.

### T4. Lifecycle and crash recovery

An attempt instance is either open or sealed. Its open-state (the tail and
`progress.json`) changes only while it is open, and only forward. Its
evidence never changes once written (T1).

```text
        open record created (no-overwrite)
                  ↓
OPEN ── events appended to the tail (fsynced); chunks rolled
  │
  ├─ run-terminated → final chunk → result.json → staged manifest, verified
  │                                   │
  │       ═══ COMMIT POINT: result.json linked to the declared result ═══
  │                                   │
  │                                   ├─ staged manifest linked ──→ SEALED complete
  │                                   └─ staged evidence lost ────→ SEALED partial,
  │                                                                 result reported unattested
  │
  └─ before the commit point: writer lost, recording failed
     or secret detected ─────────────────────────────────────→ SEALED partial, no result
```

Sealing is done by the writer, or by reconciliation when the writer is gone.

**Layout.** Under the run's declared trajectory output, each attempt has its
own directory, `attempts/<attempt_instance_id>/`:

| Object | Mutability | Created by |
|---|---|---|
| `open.json`: attempt instance id, attribution, writer identity, recording spec digest, open time | immutable; no-overwrite create | the writer, first |
| `tail.jsonl` | append-only while open; fsynced at each write-ahead point | the writer |
| `progress.json`: the rolled chunks, the head chain digest, the last sequence | replaced atomically; only ever advances | the writer, after each roll |
| `blobs/<tier>/<digest>`: this attempt's content (T6) | immutable; content-addressed within the attempt | the writer, before any event references it |
| `chunks/` | immutable; content-addressed | the writer |
| `result.json`: the attempt's copy of its result, linked to the declared result location at the commit point | immutable | the writer, at termination |
| `manifest.staged.json`: the complete manifest, fully verified before the commit point | immutable once written | the writer, before the commit point |
| `manifest.json`: the staged manifest, linked into place | immutable; no-overwrite link | the writer or the reconciler, once |
| `expiry.jsonl`: authorized expiry records (T6) | append-only | the retention process |
| `.lock` | advisory `flock`, held by the live writer | the writer |

**Write-ahead rules.** These are the durability points that make crash
evidence trustworthy:
1. The open record is durable before `run-started` is written.
2. A blob is durable before any event that references it.
3. A `*-started` event is durable (tail fsynced) **before the call it
   announces is made**. A call with no durable start was never made.
4. A `*-finished` event is written after its call returns. A crash in
   between leaves a started call with no finish, and it is reported that
   way.
5. Rolling the tail into a chunk publishes the chunk before `progress.json`
   names it. The tail is truncated only after both.

**Publication order for a completed run.** The result's publication is the
**commit point**. It can't be rolled back, so everything that can be staged
and verified is done before it.

*Before the commit point: stage and verify everything.*
1. `run-terminated` is written durably. It names the digest of the
   `HarnessRunResult` the run will publish, which is deterministic once the
   run has ended.
2. The tail is rolled into a final chunk, and `progress.json` names it.
3. The result is written durably inside the attempt directory, as
   `result.json`. It is not yet published.
4. The complete manifest is written durably, as `manifest.staged.json`. The
   writer then runs **pre-commit evidence verification**,
   `verify_trajectory` (T5), on the staged manifest:
   - every chunk's bytes and the chain;
   - the sequence, event ids and parents;
   - every captured blob's bytes;
   - the counts, and agreement with `result.json`.

   It checks nothing about publication, because no link exists yet.

**A failure in steps 1–4 publishes no result.** The run stops, and the
attempt is sealed `partial`, with abandonment `recording-failed` or the more
specific reason. This is the only window in which a recording failure can
prevent publication, and it covers every write except the two links that
follow.

*The commit point.*

5. The writer publishes `result.json` by a no-overwrite **hard link** to
   the declared location (H03's publish, ADR-019 E8). The link has
   preconditions and post-conditions:
   - **Same filesystem.** The attempt directory and the result's directory
     are on the same device. This is checked before the attempt opens
     (T9), and again here.
   - **No path substitution.** No component of the result location or the
     attempt directory is a symlink. Files are opened without following
     links, and paths are checked with `lstat`. A substituted path is
     refused before linking.
   - **Proof.** After linking, the writer compares the published path's
     `st_dev` and `st_ino` with those of `result.json`. Equal means this
     attempt published the result.
   - **Durability.** The result's directory is fsynced after the link.

   If the location already holds a file, nothing is published, and the
   attempt seals `partial`: `result-published-by-another-attempt` when the
   other file claims this request (F8), or H03's conflict when it doesn't.

*After the commit point: finish the seal.*

6. The writer links `manifest.staged.json` to `manifest.json`, then fsyncs
   the attempt directory. This is one link of a file that was already
   verified.

**After the commit point, the result stands.**
- If step 6 fails, or the writer dies, reconciliation completes it.
- If the staged evidence no longer verifies by then (a chunk or blob was
  lost or changed), reconciliation seals `partial`, with abandonment
  `post-commit-evidence-lost`. It names the published result as
  `unattested_result`, by digest, and writes an incident record.
- The result is never unpublished, and never silently trusted (see replay,
  below).

**Why link identity, not a digest.** H03's scripted runs are deterministic,
so two attempts at one request can produce byte-identical results. A digest
match can't say which attempt published the result. Link identity can, on one
host. **H05 makes no claim of execution ownership across hosts.** Proving it
is deferred, with cross-host reconciliation.

**Sealing `complete` requires both verifications to pass (T5):**
- **Pre-commit evidence verification** (`verify_trajectory`) of the staged
  manifest. `run-terminated` is present, and all staged bytes, event
  identities, counts and result agreement verify.
- **Post-commit publication verification** (`verify_publication`):
  - a result is published at the declared location;
  - that result is the same file (`st_dev` and `st_ino`) as the attempt's
    own `result.json`;
  - its digest is the one `run-terminated` named.

**Reconciliation.** `reconcile_trajectories(output)` examines every attempt
directory that has an open record and no manifest:
- **Writer alive:** the attempt's lock is held, so the attempt is skipped.
  On one host, `flock` is the liveness proof.
- **Writer gone:** the lock can be acquired. The reconciler then works by
  the furthest point the attempt reached:
  - **past the commit point** (the published result is this attempt's
    file): it runs `verify_trajectory` on the staged manifest again, then
    `verify_publication`, and links the manifest, sealing `complete`. If the
    evidence no longer verifies, it seals `partial`, with
    `post-commit-evidence-lost` and the result unattested;
  - **before the commit point:** it reads the tail, discards a torn final
    line (recording only its byte count), verifies the remaining prefix,
    and seals `partial`, with one of these `abandonment` reasons:
    - `writer-lost`;
    - `recording-failed`;
    - `result-unpublished`: `run-terminated`, and possibly `result.json`,
      exist, but no link was made;
    - `result-published-by-another-attempt`: a result is at the location,
      but it isn't this attempt's file;
    - `unverifiable-tail`: the prefix itself fails verification. The
      verified part is kept.
- **Idempotent and race-free.** Sealing is a no-overwrite link, so two
  reconcilers produce one manifest, and the loser verifies it.
- **Cross-host liveness is deferred.** A local lock proves nothing about
  another host, so H05 reconciles only directories on the local host. The
  durable path supplies leases.

**Partial records attest no result.** A partial manifest has:
- no attested `result` binding;
- `termination` set only if `run-terminated` was recorded;
- an `abandonment` reason.

The one exception is `post-commit-evidence-lost`. That manifest names the
published result as `unattested_result`, so the inconsistency is explicit
rather than hidden. Nothing requires a `HarnessRunResult` that was never
published. A later execution of the same request is a **new** attempt
instance; it never appends to an abandoned one.

**Replay checks the trajectory before returning a result.**
- **Recording off:** H03's replay is unchanged. When it returns an
  already-published result without running (ADR-019 E8), no attempt is
  opened and nothing is written.
- **Recording on:** before returning a published result, the runner:
  1. finds the **winning attempt**, the one whose `result.json` is the
     published file (link identity);
  2. if that attempt isn't sealed: waits while its writer holds the lock,
     and otherwise reconciles it;
  3. returns the result only if the attempt is sealed `complete` and passes
     both `verify_trajectory` and `verify_publication` (T5).

  Otherwise it refuses with an `UnattestedResultError` that names the
  reason. That covers:
  - no attempt claims the result;
  - the attempt is sealed `partial`;
  - verification fails.

  It never re-executes, because the result is committed. It never returns
  the result without its trajectory.

**Recording failure.**
- Before the commit point, a run that can no longer record (the store
  refuses a write, the disk is full, or a secret is detected) stops calling
  the model and tools. It publishes no `HarnessRunResult`, and its attempt
  is sealed `partial`.
- After the commit point, the result stands, and the seal is completed or
  the result is reported unattested, as above.

A completed result is never returned as attested without its complete
trajectory.

### T5. Integrity and verification

**Chunk integrity.**
- A chunk is canonical JSON Lines. Its digest is the SHA-256 of its exact
  bytes.
- Each chunk records the digest of the chunk before it, so the chunks form a
  hash chain. The last chunk's digest commits to the whole ordered history:
  reordering, dropping or replacing a chunk breaks the chain.
- Each chunk covers a contiguous sequence range, starting where the previous
  one ended.

**The sealed manifest.**
- A sealed manifest never changes.
- Sealing computes the `trajectory_fingerprint`, chapter 23's "Trajectory /
  RunHistory" identity. It is a versioned projection of:
  - the attempt instance id and the attribution;
  - the ordered chunk digests;
  - the capture dispositions (T6);
  - the completeness, and for a partial manifest its abandonment reason;
  - the attested result digest when complete, or the unattested result
    digest after `post-commit-evidence-lost`;
  - the recording spec digest.
- Locators, timestamps and current availability are excluded.
- Evaluation (H06) accepts only sealed `complete` manifests, unless it
  explicitly asks for partial evidence.

**Agreement with the result.** For a complete trajectory, a verifier
recomputes the result's counts from the events and refuses any difference:
- turns;
- tool calls (executed, refused, failed);
- context truncations;
- prompt, output and total tokens;
- termination and answer digest.

The trajectory and the result are two records of one run, and they must
agree. A partial trajectory has no result to agree with.

**Four separate operations**, following ADR-020 B4's split between
structure and bytes. Evidence and publication are verified separately,
because publication happens after the evidence is staged.

| Operation | Reads | Proves | Does not prove |
|---|---|---|---|
| `validate_manifest(manifest)` | the manifest alone | its schema; that completeness, abandonment and result bindings agree with each other; that chunk ranges are contiguous and each chunk ref's declared `previous` is the prior ref's digest; that the disposition table is well formed; and its fingerprint | anything about chunk bytes, events or the result |
| `verify_trajectory(manifest, store)`: **evidence**, before or after the commit | the manifest, every chunk, every captured blob, and the attempt's `result.json` | each chunk's bytes match its digest, and the chain is real; events parse; the sequence, ids and parents are valid; the events' content references are exactly the manifest's disposition table; captured blobs have their digests; counts and agreement with `result.json` | anything about publication: whether, where, or by whom a result was published |
| `verify_publication(manifest, attempt, result_location)`: **publication**, after the commit only, on the local host | the published path and the attempt's `result.json` | the published file has the digest `run-terminated` named, and is the same file (`st_dev`, `st_ino`) as this attempt's `result.json`. That is this attempt's publication, on this host | anything about the evidence; and anything across hosts |
| `check_availability(manifest, store)` | the blobs and the expiry log | each captured blob's current availability (T6) | anything about events |

- A manifest whose chunks are all absent passes `validate_manifest`, and
  fails `verify_trajectory`.
- Before the commit point, only `verify_trajectory` can run. Sealing
  `complete`, reconciliation past the commit, and the recording-enabled
  replay check (T4) require both it and `verify_publication`.
- `verify_trajectory` is portable: it runs anywhere the evidence is.
  `verify_publication` proves a local link, so a trajectory read on another
  host relies on its sealed `complete` status and makes no new ownership
  claim.
- Evaluation (H06) requires `verify_trajectory` on a sealed `complete`
  manifest.
- Replay also requires `check_availability` for the observations it serves.

### T6. Capture disposition is immutable; availability is observed

These are two different facts, recorded in two different places.

**Capture disposition** is what happened when content was written. It is
recorded in the sealed manifest and covered by its fingerprint:

| Disposition | Meaning |
|---|---|
| `captured` | stored in the ordinary tier, with its digest |
| `captured-restricted` | stored in the restricted tier (T8), with its digest |
| `redacted` | removed by a policy rule before writing; the rule is recorded; no digest for a secret |
| `not-provided` | the producer never had it, such as a provider that hides reasoning |

**Current availability** is what a verifier finds now. It is never written
into the manifest, so retention can't change a sealed record or its
fingerprint:

| Availability | Meaning | Treated as |
|---|---|---|
| `available` | present, with its digest | fine |
| `expired` | absent, and an **authorized expiry record** names it | expected, after its retention |
| `missing` | absent, with no expiry record | an integrity incident |
| `corrupt` | present with the wrong digest | an integrity incident |

**Blobs are stored per attempt.** In v1, each attempt stores its own blobs
under `attempts/<id>/blobs/<tier>/<digest>`, content-addressed **within the
attempt only**.
- Two trajectories that reference the same digest hold **two physical
  objects**. Expiring one can't remove the other.
- One attempt's blobs share one retention per tier, so no single physical
  object ever carries two retention obligations.
- The same digest in two tiers is two objects. Tiers are never
  deduplicated together.

This duplicates storage when content repeats across attempts. It also
makes a quarantine contain everything an attempt wrote (T8).

**Rules for any future cross-attempt deduplication.** The deletion rules are
decided now, even though the garbage collector is deferred. A shared physical
object may be deleted only when all of these hold:
- an index of **every** manifest and open attempt that references the
  object in that tier is consulted;
- the deletion time is at or after the **latest** retention deadline among
  them;
- no reference is `indefinite`, and no reference comes from an open
  attempt, whose deadline isn't fixed yet;
- deletion and new references are serialized by a lock on the object, so a
  reference added during deletion either blocks it or re-creates the
  object.

One `ExpiryRecord` is then written for each referencing manifest.

**Authorized expiry.**
- Retention removes **blobs only**, never chunks, manifests or open
  records. Those are the evidence spine, and they are small.
- Each removal is recorded first, in the attempt's append-only
  `expiry.jsonl`. An `ExpiryRecord` names:
  - the blob digest and tier;
  - the attempt and manifest it was captured under;
  - the retention rule and its deadline;
  - the actor;
  - the time.
- An expiry before the retention the manifest declared is refused. A blob
  removed early anyway is reported `missing`, not `expired`.
- An expiry record that names a blob the manifest doesn't reference is
  refused.

**Rules:**
- Missing is never read as empty. Unexplained absence and corruption are
  reported as incidents, never folded into `expired`.
- A digest attests only the content that was captured. It is never evidence
  of a complete transcript that wasn't recorded (chapter 23 §8).

### T7. Replay evidence: levels, prerequisites and a comparison projection

**Levels.** The recording spec declares one:

| Level | What is captured | What it permits |
|---|---|---|
| `none` | events and outputs only | explaining and evaluating the run |
| `recorded-responses` | every model output and tool result the run consumed | re-driving the harness against the recorded responses |

**`recorded-responses` is the minimum level eligible for replay.** A
trajectory recorded at `none` is never called replayable.

**A re-drive checks harness logic only.** It runs the same spec's harness
logic, but:
- a **recording model** serves each recorded model output in order;
- a **recording toolbox** serves each recorded tool result in order.

**No real model or tool is called.** A re-drive shows that the harness logic
reproduces its decisions given the same observations. It is not evidence
that the provider, a tool, or the world would answer the same again.

**Host tools are not assumed pure.** H03's model is a pinned script, but its
tools are host-supplied functions with no purity evidence. So:
- no trajectory, including the scripted runner's, claims that re-executing
  against live tools reproduces it;
- re-execution is not a replay level;
- a future tool-determinism declaration would need its own evidence and its
  own review.

**Prerequisites.** A trajectory may be re-driven only if all of these hold:
1. It was recorded at `recorded-responses`.
2. Its manifest is sealed `complete`. A partial trajectory may be re-driven
   as a **prefix check**, reported as such, and is never eligible to be
   called replayable.
3. Every model output and tool result it consumed has disposition
   `captured` or `captured-restricted`, and is currently `available`. A
   consumed observation that was `redacted`, `not-provided`, `expired`,
   `missing` or `corrupt` makes it ineligible, and each one is named.
4. The `AgentExecutionSpec` is available, and its request digest matches
   the attribution.
5. A runner is available whose identity matches the recorded one, or that
   declares compatibility with it for re-driving. The runner used is
   recorded.
6. The verifier can read the restricted tier, when consumed content was
   restricted.

**The comparison projection** is versioned:
`xaytune.trajectory-replay-projection/v1`. It maps each event to the fields
that state a harness decision:

| Compared | Not compared |
|---|---|
| the sequence, kind and parent | the event id and attempt instance id |
| model call: the messages-shown digest, the tools-shown digest, and the context selection | timestamps, `emitted_at` and latency |
| tool call: the id, version and arguments digest, and the outcome | the writer identity and the chunk boundaries |
| context change: the rule, the turns dropped, and the before and after digests | tiers, retention and current availability |
| termination: the reason, whether it completed, and the answer digest | redaction markers (their content is not compared) |
| tokens, compared only with the same counter | |

The served recorded responses are inputs to the re-drive, so they are not
compared. A re-drive that asks for a response the trajectory doesn't have
(an extra call) diverges at that call.

**Replay verification is independent of the writer.**
- Its result is a separate `ReplayVerification` artifact. It names:
  - the verifier identity and version;
  - the projection version;
  - the sealed manifest's fingerprint;
  - the runner used;
  - the outcome: `equivalent`, `diverged` (with the first sequence and
    field), `prefix-equivalent` for a partial trajectory, or `ineligible`
    (with every unmet prerequisite).
- The writer has no API that produces one.
- A manifest has no verification field.
- A verification whose verifier identity is the trajectory's writer is
  refused.

### T8. Security: secrets, redaction, tiers, held-out references, access and retention

**Secrets: the guarantee and its limits.**
- `SecretRef` stays the only representation of a credential. H03's runner
  refuses secrets outright (ADR-019 E9).
- A real adapter must redact before serialization, which is the existing
  rule in `xaytune.core.observability`.

**The registered-secret set is frozen when the attempt opens.**
- The host registers secret values before the attempt opens.
- The open record notes how many were registered and by which source. It
  never records the values or their digests.
- The writer's API refuses registration after open.

**What is guaranteed, and only that.** The writer scans every blob and event
for the frozen set **before writing it**. No value in that set is ever
written. For values outside the set, only the redaction policy applies, and
the ADR makes no claim about them.

**Detection of a frozen-set value fails closed, with a sanitized partial.**
The scan precedes the write, so nothing persisted contains the value. The
writer:
1. does not perform the write;
2. stops the run, before the commit point (T4);
3. writes a sanitized `error-observed` event (code `secret-detected`, and
   the field role only);
4. seals the attempt `partial`, with abandonment `secret-detected`.

No `HarnessRunResult` is published.

**A secret discovered late is contained, not disproved.** A host can learn of
a secret after the attempt opened, for example because a credential was
rotated, or another control found a value in an output. It can't register
it; it calls the writer's `contain(reason)` instead:
- **Before the commit point:**
  - the run stops, and no result is published;
  - the **whole attempt directory** is moved to an owner-only local
    quarantine. Because blobs are stored per attempt (T6), that includes
    every blob, chunk, tail and staged object the attempt wrote. Nothing of
    it remains in the ordinary tier;
  - no manifest is sealed;
  - an incident record is written in the quarantine and reported to the
    host. It carries the reason and the attempt id, and never the value or
    a digest.
- **After the commit point:**
  - the attempt is quarantined the same way;
  - the published result is the same inode as the attempt's `result.json`,
    so its mode is set to owner-only. That restricts the published path as
    well;
  - the result is reported unattested, and replay refuses it (T4).
- **No retrospective claim.** The writer doesn't scan the quarantined
  content to prove the secret is absent. It can't establish that, so it
  claims nothing. Copies a reader made before containment can't be
  recalled, and none is claimed to be.
- **What the guarantee covers about digests.**
  - For the frozen set, no value **and no digest** is ever written.
  - Containment of a late-discovered secret **creates no new digest** of
    it: the incident record carries none, and no secret redaction records
    one. Whether writes before containment contained the value, or anything
    derived from it, is not established, and is not claimed.
  - A digest of a low-entropy secret can be guessed offline, which is why
    no secret redaction ever records one.

**Redaction is declared and recorded.**
- The writer applies the run's `RedactionPolicy` (key and pattern
  selectors) before a blob is written.
- Each redaction becomes a `redacted` disposition: the policy rule that
  applied, the field role, and for non-secret content only, optionally the
  digest of what was removed.
- The recording spec carries the policy's digest.

**Two storage tiers.**
- **Ordinary:** chunks, the manifest, open records, and non-sensitive blobs.
  Readable by whoever may read the experiment.
- **Restricted:** blobs the harness or a tool declares sensitive. They are
  stored in the attempt's `blobs/restricted/`, with the same digest
  addressing. Ordinary chunks reference them by digest and tier only.

**Held-out references are excluded; an equal answer is not a leak.**
- **What is excluded.** A task's held-out reference artifact never enters a
  trajectory:
  - no `ContentRef` record of it, no locator, and no blob fetched from it;
  - no field with a reference role, in either tier.

  The runner never receives the reference (ADR-020 B2), and the writer has
  no API that accepts one.
- **What is allowed.** The agent's own answer is recorded as the agent
  produced it. A correct answer can equal the held-out answer byte for byte,
  and its blob can then have the same digest as the reference.
- That equality is the agent's output, not an exposure. Tests check that
  the reference artifact is unreachable, **not** that the answer's bytes or
  digest differ from it.

**Access control** is declared on the manifest and enforced by the store:
- the reader class for each tier;
- in H05, the local store uses owner-only file permissions;
- multi-tenant authorization is deferred.

**Retention** is declared on the manifest, per tier, as a duration or
`indefinite`. It is set by the experiment, and enforced only through
authorized expiry (T6).

**Untrusted observations.** Model text and tool output in a trajectory are
evidence, never instructions. Nothing reads a trajectory into experiment
memory, a planner prompt or a permission grant without an explicit, typed,
policy-controlled transformation (chapter 23 §8).

### T9. The recording extension, and compatibility with H03

H03's runner refuses any spec that doesn't declare **exactly one**
`xaytune.agent-run-result/v1alpha1` output. Its `config` must be exactly
`{compiler_api, token_counter}` (ADR-019 E9). Recording is added as a
supported, versioned extension of that contract, not as an exception to it.

**Recording disabled: byte-identical.**
- The compiler's recording option defaults to off. Off, it emits exactly
  the H03 spec: one result output, and the same two config keys.
- `agent_execution_fingerprint` and `agent_run_request_digest` are
  therefore unchanged, and so are the H03 pinned values.
- `AgentExecutionSpec` gains **no field**. Its schema, `api_version` and
  identity projections are untouched. A new field would change every
  request digest, because the digest hashes the whole spec.

**Recording enabled: two existing fields carry it.**
- **`outputs`** declares, in this order:
  1. the result (`name="result"`, kind `xaytune.agent-run-result/v1alpha1`);
  2. the trajectory (`name="trajectory"`, kind
     `xaytune.agent-trajectory/v1alpha1`, an absolute local directory).
- **`config`** gains one key, `trajectory`, holding a versioned
  `TrajectoryRecordingSpec`
  (`xaytune.agent-trajectory-recording/v1alpha1`). It declares:
  - the chunk bounds;
  - the replay level;
  - the redaction policy digest;
  - the tier and retention declarations;
  - the writer identity.
- Both fields are already in the execution identity and the request digest.
  A recording run therefore has its own fingerprint and digest, as any
  different request does. All of these settings are operational; none
  enters `HarnessFingerprint`.

**The runner contract becomes exactly one of two shapes:**
- the H03 shape: one result output, and the two config keys;
- the recording shape: the result output and one trajectory output, in that
  order, and the two config keys plus a `trajectory` block that validates.

Everything else is refused, with reasons:
- two trajectory outputs;
- a trajectory output without the config block, or the block without the
  output;
- an unknown recording schema version;
- the outputs in the other order;
- bounds out of range;
- a trajectory directory on a different filesystem from the result
  location, checked before the attempt opens. Link publication (T4) would
  be impossible.

**Identities stay put.**
- For every spec H03 already accepted, the compiler's output and the
  runner's behavior are unchanged. So the compiler identity in the spec and
  the runner identity in `HarnessRunResult` keep their versions.
- A result published before H05 still replays, because ADR-019 E8's
  attribution check includes the runner.
- The trajectory writer has its own identity, recorded in the open record
  and the manifest.
- **Fail closed elsewhere.** A runner without recording support refuses a
  recording-shaped spec, as H03's runner already does. It never runs one
  silently without recording.

**Unchanged by H05:**
- `harness_identity_v1` and the harness fingerprint;
- H04's task, suite and cohort identities;
- `HarnessRunResult` v1alpha1. The manifest references the result, and the
  result never references the manifest;
- `agent-telemetry/v1alpha1`.

### T10. Storage reuses existing abstractions, with no migration

**A `TrajectoryStore` protocol** stages, verifies and publishes immutable
objects, following `CheckpointStore`'s pattern. It offers:
- put a blob, in a tier, within an attempt;
- append to and roll the tail;
- write `progress.json`;
- write the result copy and the staged manifest;
- commit, by linking the result;
- seal, by linking the manifest;
- contain, by quarantining an attempt;
- append an expiry record;
- get by reference.

**Publication** is content-addressed and no-overwrite, using H03's
`os.link` publish. Publishing again with identical bytes is a no-op; with
different bytes it is a conflict.

**H05 ships a local store only**, under the run's declared trajectory
output, beside its `HarnessRunResult`. Its files are owner-only, and blobs
are stored per attempt (T6).

**References:**
- a sealed manifest is referenced by an `ArtifactRef` of a **new, dedicated
  `ArtifactKind`, `agent_trajectory`**, with its digest;
- blobs and chunks are referenced by digest within the manifest, as H02 and
  H04 reference content.

Adding a value to the `ArtifactKind` literal changes no existing
serialization or fingerprint, and a test pins that.

**No repository table and no migration.** Indexing trajectories in SQLite
belongs to the durable path (ADR-019 M1–M6) or a later review. Core rows will
hold only manifest references, digests and bounded summaries.

### T11. Deferred

Each of these needs its own review:
- durable indexing and persistence of manifests, and the ADR-019 M1–M6
  migration;
- cross-host liveness and reconciliation, with leases;
- streaming trajectory events through runtime telemetry, as a new telemetry
  protocol version;
- event mappings for real adapters, and their redaction and replay
  declarations (H08–H11);
- delegation, handoff, memory, retrieval and compaction event kinds;
- tool-determinism declarations and their evidence;
- scoring and evaluating trajectories (H06), and lineage between
  `EvaluationRun`s and trajectories;
- the retention enforcement process (garbage collection), deletion of
  sealed manifests, and legal hold;
- cross-attempt blob deduplication, under the deletion rules T6 already
  fixes;
- operator handling of quarantined attempts;
- multi-tenant authorization and encryption at rest;
- exactly-once execution across hosts;
- Merkle proofs for partial disclosure. The hash chain proves the whole
  history, not a single event alone.

## Decisions on the first review's questions

| Question | Decision |
|---|---|
| Minimum replay level | `recorded-responses` is the minimum level eligible for replay (T7). Below it, a trajectory is evidence only |
| Secret detected | The registered set is frozen at open. A frozen-set value is caught before writing, and the attempt is sealed as a sanitized `partial`. A secret discovered late contains the whole attempt, including its blobs (and the published result, if there is one), and claims nothing retrospectively (T8) |
| Chunk bounds | Declared per run in the recording spec, with defaults, validated against fixed ranges, and part of execution identity (T1) |
| Artifact kind | A dedicated `ArtifactKind`, `agent_trajectory`, not `provenance_bundle` (T10) |

## Failure scenarios

The implementation must handle each scenario as stated:

| # | Scenario | Required outcome |
|---|---|---|
| F1 | Crash before the open record is durable | Nothing is recorded, and no call was made. The next execution opens a new instance |
| F2 | Crash after `model-call-started` is durable, before the call returns | Reconciliation seals `partial` (`writer-lost`), showing a started call with no finish |
| F3 | Crash while appending: a torn last line | The torn line is discarded and counted, and the verified prefix is sealed `partial` |
| F4 | Crash after `run-terminated`, before the commit point | Sealed `partial` (`result-unpublished`). The next execution is a new instance; nothing appends to the old one |
| F5 | Crash after the commit point, before the seal link | Reconciliation runs `verify_trajectory` and `verify_publication`, and seals `complete` |
| F6 | Two reconcilers race on one attempt | One manifest is linked; the other verifies it and stops |
| F7 | A reconciler meets an attempt whose writer is alive | Skipped: the lock is held |
| F8 | Two executions of one request race, with byte-identical results | Two attempt instances with distinct ids. One link wins. Only the attempt whose `result.json` is the published file seals `complete`; the other seals `partial` (`result-published-by-another-attempt`). The race is on one host in H05; cross-host is deferred |
| F9 | H03 replays a published result | No attempt instance is opened, and nothing is written |
| F10 | A store write fails before the commit point, at any of steps 1–4 | The run stops, publishes no result, and seals `partial` (`recording-failed`) |
| F11 | A frozen-set secret appears in a model output, detected before writing | Sanitized `partial`, no result, and no bytes or digest of the secret written anywhere |
| F12 | A secret is discovered after the attempt opened | Registration is refused. `contain` quarantines the whole attempt directory, including its blobs. No manifest, and an incident with no value or digest. After the commit point, the published result is also made owner-only and reported unattested |
| F13 | A blob expires under its declared retention | Reported `expired`. The manifest and its fingerprint are byte-identical |
| F14 | A blob is deleted with no expiry record, or before its retention | Reported `missing`, as an integrity incident |
| F15 | A blob's bytes change | Reported `corrupt` |
| F16 | An edited, dropped, reordered or replaced chunk, or an edited sealed manifest | Refused, with the specific reason |
| F17 | A re-drive with changed harness logic under the same `AgentExecutionSpec` | `diverged`, at the first differing sequence and field. A changed spec has another request digest, so it is `ineligible` (prerequisite 4), not a divergence |
| F18 | A re-drive of a trajectory whose consumed tool result was redacted or expired | `ineligible`, naming each unmet prerequisite |
| F19 | The writer attempts to record its own replay verification | Refused |
| F20 | A correct agent answer equals the held-out reference | Recorded normally. The reference artifact remains unreachable |
| F21 | The seal link (or its directory fsync) fails immediately after a successful commit | The result stands. The writer reports the failure, and reconciliation completes the seal. Until then, replay doesn't return the result |
| F22 | After the commit point, a staged chunk or blob is lost or changed before the seal | Sealed `partial` (`post-commit-evidence-lost`), naming the result as `unattested_result`, with an incident. Replay refuses with `UnattestedResultError` |
| F23 | Replay of a recording-enabled request | The result is returned only after the winning attempt is sealed `complete` and passes `verify_trajectory` and `verify_publication`. A tampered chunk makes replay refuse |
| F24 | One digest is referenced by two manifests, with retention of 7 and 90 days | Two physical objects. Expiring the first at 7 days leaves the second `available`. An expiry of the second at 7 days is refused |
| F25 | A published result at a recording-enabled location that no attempt claims | Replay refuses it as unattested |
| F26 | The result location or an attempt path component is replaced by a symlink | Refused before the commit point; nothing is published |
| F27 | The result directory is on another filesystem | Refused before the attempt opens |

## Acceptance criteria for the implementation PR

Each criterion needs positive tests, and the **negative tests listed**.

1. **Schema.**
   - Every v1 event kind parses.
   - Negative: an unknown kind, an unknown field, inline content, a bad
     digest, an oversize event.
2. **Identity.**
   - Attempt instance ids are minted, persisted before `run-started`, and
     unique across concurrent executions of one request.
   - Attribution carries the request, replicate and (when present) durable
     ordinal separately.
   - Negative: reusing an existing id (refused); deriving an id from the
     request digest (two concurrent executions must differ).
3. **Ordering and causality.**
   - Gapless sequences.
   - Deterministic event ids.
   - A redelivered identical event is deduplicated.
   - Negative: a conflicting event with one id, a gap, a parent that is
     later, missing or of a disallowed kind.
4. **Lifecycle and recovery.**
   - F1–F10, F21–F23 and F25–F27 each produce the stated outcome.
   - **Commit point:** a fault-injecting store fails at every write and
     link in turn.
     - A failure before the commit point leaves no published result.
     - A failure after it leaves the result published, and the seal
       completed by reconciliation or the result reported unattested.
   - **Link proof:** the result is published by a real hard link, and
     `st_dev` and `st_ino` are compared. A same-filesystem check runs
     before publication, symlinks are refused, and directories are fsynced
     after the link and after the seal (checked with an fsync spy).
   - Every write-ahead point is fsynced before its call. A test model
     asserts the start event is on disk when it is called.
   - Negative: replay of a recording-enabled request returning a result
     whose winning attempt is unsealed, partial or fails verification; a
     replay that re-executes after a commit.
   - Negative: sealing `complete` without a result, with a mismatched
     result digest, or with disagreeing counts; sealing `complete` for an
     attempt whose result digest matches but which didn't publish the file
     (F8); a partial manifest that
     binds a result; appending to a sealed or abandoned attempt.
5. **Integrity and the four operations.**
   - `validate_manifest` accepts a well-formed manifest whose chunks are
     absent. `verify_trajectory` refuses it.
   - **Pre-commit:** `verify_trajectory` passes on a staged manifest
     before any result is linked, and its outcome doesn't depend on whether
     a link exists.
   - **Post-commit:** `verify_publication` passes only after this
     attempt's link.
   - Negative (`verify_publication`):
     - no published file;
     - another attempt's byte-identical file;
     - a copy rather than a link (same bytes, different inode);
     - a published digest different from `run-terminated`'s;
     - a symlink at the published path.
   - Negative (sealing): sealing `complete` with only `verify_trajectory`
     passing, or with only `verify_publication` passing.
   - Negative: F16's tamperings, each with its own reason; a chain break; a
     chunk whose range doesn't continue the previous one; events whose
     content references differ from the manifest's disposition table.
6. **Agreement.**
   - For every H03 termination reason, a complete trajectory agrees with its
     `HarnessRunResult`.
   - Negative: each counted field changed by one is refused.
7. **Disposition and availability.**
   - F13–F15.
   - Expiry leaves the manifest bytes and fingerprint identical.
   - Negative: an expiry record for an unreferenced blob; expiring a chunk,
     a manifest or an open record; an early expiry, reported `missing`;
     `missing` or `corrupt` presented as `expired`.
   - Negative (F24): one digest referenced by two manifests with different
     retention. Expiring the shorter one must leave the longer one's object
     available, and an expiry of the longer one before its deadline is
     refused.
8. **Replay.**
   - A `recorded-responses` re-drive of a scripted run is `equivalent`
     without calling a real model or any host tool. A test toolbox that
     fails if called proves it.
   - A partial trajectory gives `prefix-equivalent`.
   - Negative: F17–F19; re-driving a `none` trajectory (`ineligible`); a
     verification naming an unknown projection version.
9. **Security.**
   - F11 and F12.
   - Registration after the attempt opens is refused.
   - Redactions record their rule.
   - Restricted blobs are referenced only from ordinary chunks.
   - Error events carry no provider bodies or input values.
   - Negative: a frozen-set secret's bytes or digest appear anywhere,
     including the quarantine's incident record; containment writing any
     digest of a late-discovered secret; after `contain`, any blob,
     chunk or tail of the attempt still readable outside the quarantine,
     or a published result still readable by others; a writer claiming a
     quarantined attempt clean; a writer API accepting a reference
     artifact; a reference locator or `ContentRef` appearing in any blob,
     event or manifest.
   - F20: an answer equal to the reference is recorded, and the test does
     **not** fail on a digest coincidence.
10. **Recording extension and compatibility.**
    - Recording off: the spec is byte-identical, and the H03 pinned
      execution fingerprint and request digest are unchanged.
    - A pre-H05 result still replays.
    - `HarnessRunResult` v1alpha1 and the H02–H04 pins are unchanged.
    - No migration or repository table is added.
    - Adding the `agent_trajectory` kind changes no existing serialization.
    - Negative: every refused shape in T9.
    - Negative: a runner without recording support accepting a recording
      spec.
11. **Chunk bounds.**
    - Defaults apply, and bounds are respected.
    - Negative: zero, below or above each range, a non-integer, a boolean.
12. **Purity.** The trajectory contracts import no storage, runtime, agent,
    torch or transformers module. The store, writer and reconciler live
    outside core.

## Consequences

- A harness run leaves a verifiable account of what happened, attributable
  to exactly one execution, with every omission declared.
- Crash evidence is trustworthy: a call without a durable start was never
  made, and abandoned attempts are sealed rather than lost.
- Evaluation (H06) gets a sealed, fingerprinted subject to measure, and the
  agreement check between trajectory and result.
- Real adapters (H08 onward) inherit one format, with explicit obligations
  for redaction and replay rather than an implied guarantee.
- Per-attempt blob storage duplicates repeated content across attempts. It
  is the price of simple retention and complete containment, until
  deduplication is reviewed.
- Write-ahead fsyncs add latency to every call. That is acceptable for
  model and tool calls that take far longer than a sync.
- Storage cost grows with every recorded run. Retention is declared from the
  start, but enforcing it is deferred.
- Nothing becomes durable in a database yet. Trajectories live beside their
  results until the durable path exists.

## Alternatives considered

- **One JSON transcript per run:** rejected. It is unbounded, a crash leaves
  nothing verifiable, and redaction and access control would have to act on
  the whole document.
- **Embedding the trajectory in `HarnessRunResult`:** rejected. That changes
  H03's accepted result schema, makes the result unbounded, and mixes a
  summary with its evidence.
- **A new `AgentExecutionSpec` field for recording:** rejected. The request
  digest hashes the whole spec, so every recording-disabled digest would
  change.
- **Deriving the attempt id from the request digest and an ordinal:**
  rejected. The ordinal is unknown after a crash and duplicated across
  hosts, so the id would be ambiguous exactly when it is needed.
- **Sealing before publishing the result:** rejected. A sealed `complete`
  manifest would attest a result that a crash could leave unpublished, and
  a sealed manifest can't be retracted. Staging and verifying first, then
  committing the result and linking the verified manifest, puts only one
  link after the commit point.
- **Unpublishing a result when sealing fails:** rejected. Publication is
  H03's idempotency commit, and a later reader may already have replayed
  the result. The failure is completed or reported unattested instead.
- **Shared content-addressed blobs across attempts in v1:** deferred. That
  needs a reference index and deletion locking (T6's rules). Per-attempt
  storage avoids both, and makes containment complete.
- **Scanning a quarantined attempt to prove a late secret absent:**
  rejected. A scan can't establish that, so nothing is claimed.
- **Recording availability in the manifest, and rewriting it on expiry:**
  rejected. Retention would change sealed evidence and its fingerprint.
- **Publishing a chunk at every write-ahead point instead of a tail:**
  rejected. Every call would produce tiny chunks, and the chain would grow
  with every call.
- **Aborting without publishing anything when a secret is detected:**
  rejected, except when the prefix can't be proven clean. A clean, sanitized
  partial keeps the evidence that the run was stopped and why.
- **Fixed chunk bounds for every run:** rejected. Workloads differ, and a
  declared, validated per-run bound is just as reproducible.
- **Reusing `provenance_bundle` for trajectories:** rejected. A reader must
  be able to tell an execution record from a provenance bundle without
  opening it.
- **Events as SQLite rows now:** rejected. It is a migration before the
  durable path is decided, and it puts raw content into core rows, which
  chapter 23 §8 rules out.
- **Streaming through `RuntimeEventEnvelope` and
  `xaytune.telemetry/v1alpha3`:** rejected for v1. Its payload families are
  training and evaluation, there is no harness target kind, and ADR-019 E7
  already decided terminal-only telemetry for this runner.
- **OpenTelemetry spans as the canonical format:** rejected. Spans have no
  digest chain, completeness or redaction semantics, and exporters may
  sample or drop them. An export can be added later from the canonical
  record.
- **Random event ids:** rejected. A redelivered event would look new.
- **Content inline in events:** rejected. Chunks would be unbounded, and one
  sensitive tool result would force a whole chunk into the restricted tier.
- **Hashing redacted secrets to prove what was removed:** rejected. A digest
  of a low-entropy secret can be guessed offline.
- **Treating re-execution against live tools as replay:** rejected. Host
  tools carry no purity evidence.
- **A Merkle tree instead of a hash chain:** deferred. v1 has no
  partial-disclosure requirement.
- **Claiming reproducibility from a recorded trajectory:** rejected. A
  recording is evidence. Replay is a separately verified claim about harness
  logic only.
