# ADR-019 — Harness compilation and execution

## Status

Accepted — 2026-10-11. Proposed 2026-10-11 with H03.

The project owner approved the architecture and accepted decisions **E1–E10**
as implemented in [PR #75](https://github.com/szaher/xaytune/pull/75) at head
`2a99938`, after review 1's fixes.

**Not accepted by this ADR:** §"Migration strategy for the durable path"
(M1–M6) is a proposed strategy. It is not implemented, and it is not one of
this ADR's decisions. It is the starting point the durable follow-up reviews
and decides on its own, so no part of this ADR is half-accepted.

**Durable path progress (Track B).** Step 1 of M6's order, the downgrade
guard, is implemented: `migrate()` refuses a database whose ledger records a
migration the build does not ship (`SchemaTooNewError`), from every entry
point that migrates, before anything is written. The first release that
contains it is the **minimum compatible version** of any database a later
rebuild-mode migration (M2–M3) changes.

Step 2, the versioned candidate snapshot (ADR-018 D10), is implemented. A
node's snapshot at version `"2"` holds an `ExperimentCandidate`, and
migration 021 adds `experiment_nodes.candidate_kind`. The column is added,
not rebuilt, so 021 needs no rebuild mode.

- **Integrity, in two layers.** Neither does the other's job.
  - *SQL* (021's triggers, on every write path): a snapshot's `spec_version`
    must be exactly `"1"` or `"2"`, in that version's shape. A missing, unknown,
    non-text or mixed one is refused, not read as training. On insert, the
    `candidate_kind` and `candidate_fingerprint` columns must equal what the
    payload declares, and an update may change neither, nor the payload's
    declaration. This is consistency between stored columns and declared
    fields. SQL cannot compute a fingerprint, so it does not establish that the
    declaration is true.
  - *The repository* (`decode_node_row()`, on every repository and graph read):
    parses the snapshot, recomputes the candidate's identities, and refuses
    (`NodeRecordMismatchError`) a row whose columns are not the candidate's
    actual kind and scientific identity. That includes a row written around
    the triggers.
- **Fails closed.** A harness node can be recorded and read, but a run for one
  is refused, so harness work fails closed (M1).
- **Release order.** 021 ships only in a release **after** the guard-bearing
  one (1.0.0a2). Builds with the guard and without 021 refuse a migrated
  database. Builds older than the guard cannot read a version 2 harness
  snapshot, so they must not meet one: the guard-bearing release comes first,
  and it is the minimum compatible version for 021 as for the rebuilds.

Steps 3–5 remain proposed, each for its own review.

Builds on [ADR-018](ADR-018-agent-harness-candidates.md), accepted on
2026-10-11 as a separate decision.

**Scope, plainly:** H03 is a compiler and a local, in-process runner. It is
**not** end-to-end harness execution through Xaytune. Nothing here submits a
harness to a controller or a runtime. The durable path is the follow-up
milestone described in §"Migration strategy for the durable path".

## Context

H02 made a harness candidate a typed, fingerprinted proposition that nothing
could execute. H03 makes it executable: a compiler turns it into a workload
spec, and a runner executes that spec. Chapter 23 §6 sketched the path:

```text
AgentHarnessSpec → HarnessCompiler → AgentExecutionSpec
  → capability resolution → ResolvedExecutionPlan → RuntimeBackend
```

Before adding to it, H03 reviewed the existing execution contracts that path
would cross. Each one is durable:

- **Request digests.** `ResolvedExecutionPlan.request_digest` hashes the
  whole plan, including its spec and its `target`. Every recorded training
  and evaluation submission is keyed by one.
- **The spec union.** `ExecutionSpec` is `TrainingExecutionSpec |
  EvaluationExecutionSpec`, discriminated by `api_version`. Adding a member
  does not change either member's bytes, so it would not move a digest.
- **The target.** `ResolvedExecutionPlan.target` is a `RuntimeOperationTarget`
  with `kind: Literal["training-attempt", "evaluation-attempt"]`. The plan
  validator pairs each kind with its spec, and `RuntimeEventEnvelope` pairs
  each kind with its telemetry payload family. An import-time check refuses
  a kind without a family.
- **Persistence.** The target kind is the name of an attempt **table**. The
  repository resolves `target_id` in it. SQLite `CHECK` constraints limit
  `target_kind` to the two kinds in `actions` (003), `incidents` (010) and
  `recovery_episodes` (012), and limit `subject_kind` in the budget ledger
  (`budget_ledger`, 008). Triggers in 010 and 012–014 join through it.
- **Candidate storage.** Nodes and snapshots still hold `CandidateSpec` only
  (ADR-018 D10). A harness candidate cannot be a node, so it cannot have a run
  or an attempt.

## The boundary H03 stops at

A harness workload can reach a `RuntimeBackend` only as a
`ResolvedExecutionPlan`, and a plan needs a target. A new target kind, such
as `agent-attempt`, needs all of the following before any runtime could be
asked to run one:

- a harness attempt table, with a lineage it cannot have until nodes can
  hold an `ExperimentCandidate` (a versioned snapshot column; ADR-018 D10);
- widened `CHECK` constraints in some of `actions`, `incidents`,
  `recovery_episodes` and `budget_ledger`. SQLite cannot alter a constraint,
  so each widened table must be rebuilt under live foreign keys, triggers and
  views;
- a telemetry payload family for the new kind, and a controller reconciler
  for it.

That is a new table and up to four rebuilt ones, on top of the candidate
snapshot migration ADR-018 D10 deferred. §"Migration strategy for the durable
path" proposes how, and argues that the first durable slice needs to rebuild
only two of them. Doing it safely needs its own review, and the H03 brief says
to stop at this boundary rather than work around it. **H03 therefore does not add a target kind, does not add
`AgentExecutionSpec` to the `ExecutionSpec` union, and does not resolve a
harness into a `ResolvedExecutionPlan`.** Every historical plan, request
digest and execution fingerprint is unchanged because nothing they hash was
touched.

What H03 does deliver is the part of the path that needs no migration:
compilation, the execution spec and its identity, and a deterministic
in-process runner. These are the contracts a durable path will carry later.

## Decisions

**E1. `HarnessCompiler` is a sibling of `TrainerCompiler`.** It is a
versioned protocol:

- **API version:** `xaytune.harness-compiler/v1alpha1`.
- **`descriptor`:** a `PluginDescriptor`, checked with
  `require_supported_plugin` like every plugin.
- **`capabilities()`:** a typed `HarnessCompilerCapabilities` document. It
  names the harness schemas, model providers, template syntaxes, context
  selections, overflow modes and termination conditions the compiler can
  emit, the tool-schema dialect it accepts, and the runner it targets.
- **`supports(harness)`:** a `SupportResult` listing every reason the harness
  is refused, not just the first.
- **`compile(harness, context)`:** an `AgentExecutionSpec`. It raises
  `UnsupportedHarnessError` carrying every reason.

Compilation is deterministic and has no side effects. It never calls a model,
runs a tool, reads a file or the network, or submits a workload. A compiler
that cannot honour a setting refuses it; it never drops it.

**E2. Content is resolved and verified before compilation.**
`resolve_harness_content(harness, source)` fetches every **reachable**
`ContentRef` from a `ContentSource`. Reachable content is the system prompt,
the task template, and the description and contract of each enabled tool.
Each fetched value must:

- have the SHA-256 the reference declares;
- decode as UTF-8;
- be a `text/*` media type.

Otherwise it is refused before use. Content for disabled tools is never
fetched. The result is a `ResolvedHarnessContent`, which re-verifies every
digest whenever it is constructed or parsed. H03 ships only an in-memory
`MappingContentSource`. No network source exists, so nothing fetches a URI.

**E3. `AgentExecutionSpec` is a sibling workload spec.** Its API version is
`xaytune.agent-execution/v1alpha1`. It subclasses the same transport base as
the training and evaluation specs, and is **not** a member of the
`ExecutionSpec` union (see the boundary above). It carries:

- the compiler identity;
- the whole `AgentHarnessSpec` and its `harness_fingerprint`;
- the verified reachable content;
- the explicit execution context: task input, seed, replicate;
- the tools whose authorization the compiler checked (`effective_tools`);
- the outputs, and a telemetry contract.

Its validators re-derive every derived field: the fingerprint, the content
digests and coverage, `effective_tools` against the enabled set, and the
task variables against the template. A spec edited after compilation
therefore fails to parse instead of running something else.

**E4. Scientific identity and execution identity are separate.**
`harness_fingerprint` (ADR-018 D3) is the scientific identity, and nothing in
H03 changes it. The new execution identity is
`agent_execution_fingerprint`, built from an explicit projection
`agent_execution_identity_v1` (`type="agent-execution"`, version 1).

| | Included |
|---|---|
| **In** | `harness_fingerprint` (never the harness again), compiler name and version, entrypoint, arguments, config, environment, secret references, dependencies, container, inputs, outputs, resources, the telemetry contract, required capabilities, task, seed, replicate, `effective_tools`, and content digests |
| **Out** | spec and harness `metadata`, observability settings, content text (its digest is already in) |

So an operational change, such as a seed, a replicate, an output location,
resources or a compiler release, changes execution identity and leaves
scientific identity alone. The idempotency key for a run is a separate,
literal digest of the whole spec (`agent_run_request_digest`), for the same
reason ADR-013 keeps `request_digest` apart from the execution fingerprint.

**E5. Authority comes from outside the candidate, and is checked twice.**
A harness grants nothing (ADR-018 D9).

- **At compile time,** the context supplies the authorized tool set. The
  compiler calls `require_tools_authorized` and records the result as
  `effective_tools`.
- **At run time,** the runner is given its own authorized set by its host,
  not by the spec. It refuses to start unless every enabled tool is in that
  set and the host's toolbox supplies an implementation at the declared
  version.
- **During the run,** a call to a tool that is not enabled is answered with a
  refusal and is never executed.

The runner does not import tools by name. It does not run code from the spec
and does not open a network connection.

**E6. The first runner is deterministic and honest about it.**
`ScriptedHarnessRunner` executes an `AgentExecutionSpec` in-process against a
`ScriptedHarnessModel`.

- **The model is the script.** A harness compiled for it must bind
  `provider="xaytune"`, `model="scripted"`, with an `artifact_digest` equal to
  the script's digest. The runner refuses a script whose digest does not
  match. A harness naming any other provider is refused at compile time,
  because the reference compiler cannot run it.
- **Only the trusted implementation runs.** A digest is a claim, and any
  object can make it. The runner accepts exactly `ScriptedHarnessModel`: not
  a subclass, and not another object reporting the pinned digest. It
  recomputes the digest from the script that model holds. Real adapters need
  their own runner, with an implementation binding of their own (H08+).
- **It consumes the declared configuration.** Every field it reads changes
  what it does:
  - it renders the system prompt and the task template;
  - it shows the model only the enabled tools, with their descriptions and
    contracts;
  - it validates tool arguments against each tool's input schema;
  - it selects context by `full-history` or `recent-turns`, and enforces
    `max_context_tokens` with `fail` or `truncate-oldest`;
  - it truncates tool results and model output to their token limits;
  - it applies stop sequences;
  - it stops on `max_turns`, `max_total_tokens` and `max_tool_calls`, and on
    the declared `terminate_on` conditions.
- **Its tokens are counted, not estimated.** The counter is
  `xaytune.whitespace-tokens/v1`, which counts whitespace-separated words. It
  is recorded in the spec and in the result, so nobody mistakes it for a
  provider's tokenizer.
- **It does not score.** The result says how the run ended and what it
  produced. Whether that was any good is H06's question.

Template syntaxes it supports are `literal`, and `python-format` with plain
`{name}` fields only. `jinja2` is refused, because the reference compiler has
no sandboxed renderer. Tool schemas must be in xaytune's closed agent-schema
subset; any other keyword is refused. Provider parameters must be empty,
because the scripted model accepts none.

**E7. Telemetry is one terminal result.** The telemetry contract is
`xaytune.agent-telemetry/v1alpha1`. It streams no events: the run publishes
one `HarnessRunResult` (`xaytune.agent-run-result/v1alpha1`) as its declared
output. The result records:

- **attribution:** the request digest, the execution and harness
  fingerprints, the compiler and runner identities, the model digest, the
  task, seed and replicate;
- **how it ended:** the termination reason, whether that was a declared
  completion, and the answer;
- **counts:** turns, tool calls (executed and refused), and prompt, output
  and total tokens, with the name of the token counter.

A turn-by-turn trajectory is H05's artifact and is not invented here.

A result is **internally consistent or it does not parse**, because results
are read back and replayed. Every result must satisfy all of these:

- `completed` holds exactly for a completing termination;
- only a completed run has an answer, and every completed run has one;
- only a `model-error` run records an error, and it always does;
- `total_tokens` is `prompt_tokens + output_tokens`;
- refused and failed tool calls are counted among `tool_calls`;
- tool calls and tokens need model turns;
- digests are well formed and counts are strict integers.

**E8. A run is idempotent, restartable and cancellable.**

The guarantees are stated exactly, because a production adapter must not
inherit a stronger claim than this runner makes.

- **One published result.** The result is published with a no-overwrite
  link at its declared location. A different request at the same location is
  refused as a conflict.
- **Replay is checked, not believed.** A published result is returned only
  if it parses (with the invariants of E7) and agrees with the current spec
  on every field derivable from it:
  - the request digest;
  - the execution and harness fingerprints;
  - the compiler, runner and model digest;
  - the task, seed and replicate;
  - the token counter.

  A record that claims the request but disagrees on any of these is refused
  as a conflict, naming the fields.
- **One execution per output location, on one host.** Runs are serialized by
  an advisory `flock` on a lock file beside the result. A second run waits,
  then replays the first's result without calling the model or any tool.
- **Not exactly-once.** A run on another host, or a rerun after a crash
  before publication, executes again and repeats its model and tool calls.
  That is harmless here, with a deterministic script and pure host tools. It
  is not an exactly-once execution guarantee. The durable path gets
  idempotency from the operation journal (ADR-013), not from this runner.
- **Restart:** a crash before publication leaves no result and no temporary
  file. The next run executes from the start, and because it is
  deterministic it produces the same result.
- **Cancellation:** a cancellation signal is checked before every model call
  and every tool call. A cancelled run publishes a terminal `cancelled`
  result, which later runs of the same request return, as ADR-013 does for
  terminal records.

**E9. Everything else fails closed.** `ScriptedHarnessRunner.supports(spec)`
refuses, with reasons, every requirement it does not implement:

- a container, secrets, dependencies, artifact inputs, arguments or
  environment variables;
- any resource requirement;
- any capability requirement other than its own runner identity;
- an entrypoint other than its own;
- any compiler API other than `xaytune.harness-compiler/v1alpha1`.

It does not pretend to schedule what it cannot.

**E10. Deferred.** Each of these needs its own review:

- the durable path, proposed in §"Migration strategy for the durable path";
- persistence of harness candidates;
- real model adapters (H08–H11);
- benchmarks and task contracts (H04);
- trajectories (H05);
- evaluation and optimization (H06/H07);
- memory, delegation and joint mode.

H03's `HarnessTaskInput` (an opaque task id plus string variables) is a
placeholder that H04's task contract replaces.

## Acceptance criteria: passed here, and deferred

H03 tests execution semantics **in-process only**. The criteria below are
**deferred to the durable follow-up**. They are not passed and must not be
reported as passed:

| Criterion | H03 (in-process runner) | Durable path (follow-up) |
|---|---|---|
| Deterministic compilation | Passed | -- |
| Scientific vs. execution identity | Passed | -- |
| Unsupported configuration refused before execution | Passed | Runtime capability resolution: **deferred** |
| Content verified before use | Passed | -- |
| Runner consumes the declared harness and attributes its output | Passed | -- |
| Idempotency | Passed for one output location on one host: one execution under the host lock, and replays verified against the spec. Not exactly-once across hosts or crashes | Operation-journal `request_digest` get-or-create: **deferred** |
| Restart | Passed for a crash before publication | Controller restart and reconciliation of an in-flight attempt: **deferred** |
| Cancellation | Passed for an in-process signal | Cancel `Action`, cancel operation and durable terminal state: **deferred** |
| Controller submission | Not attempted | **Deferred** |
| Runtime execution (Local, Ray, KubeRay) | Not attempted | **Deferred** |
| Existing training and evaluation behavior | Passed (full suite, pinned digests) | Re-checked after the migration |

## Migration strategy for the durable path (proposed; not implemented; not accepted)

This section is the design the follow-up starts from. It is **not** among the
decisions accepted above. H03 implements none of it, and each step needs its
own review and decision.

**M1. Scope: rebuild as few tables as the first durable slice needs.**

| Table | First durable slice | Why |
|---|---|---|
| `agent_attempts` (new) | Created | The table a harness target kind names; its lineage needs M6 step 2 first |
| `actions` | **Rebuilt**: `target_kind` gains the harness attempt kind | Cancelling an attempt is an `Action` aimed at it (ADR-005 §5, ADR-013) |
| `budget_ledger` | **Rebuilt**: `subject_kind` gains it | Attempts are charged and settled like the others |
| `runtime_operations` | Unchanged | It deliberately has no `CHECK` on `target_kind`. The new kind is a domain change (`OperationTargetKind` and the repository's kind-to-table maps), not a rebuild |
| `incidents` | **Not rebuilt** | No incidents or recovery for harness attempts in the first slice. The existing `CHECK` keeps refusing the new kind, which fails closed |
| `recovery_episodes` | **Not rebuilt** | As above |

**Harness recovery fails closed explicitly, not by accident.** The first
durable slice must never route a harness attempt into training or evaluation
recovery logic. A harness incident, recovery plan or recovery `Action` is
refused at the domain layer with a reason naming the missing ownership
contract, as well as by the unchanged `CHECK`s, and tests cover each refused
path. Harness recovery arrives only with its own ownership triggers and
closure semantics.

Leaving `incidents` and `recovery_episodes` alone matters beyond saving work:

- **Ownership triggers.** `incidents_belong_to_their_attempt` (010) and
  `recovery_episode_ownership` (012) prove ownership by joining training or
  evaluation lineage, including `candidate_fingerprint`. A harness kind would
  need its own branch, proving harness lineage.
- **The closure view.** `recovery_episode_closure` (012) uses
  `CASE … ELSE evaluation_attempts`. A third kind would silently fall into
  the evaluation branch. Before any slice adds harness recovery, that `ELSE`
  must become an explicit branch for each kind.

**M2. The migration runner needs a reviewed rebuild mode first.**

- **Today's runner cannot drop `actions`.** `migrate()` runs every file inside
  `BEGIN IMMEDIATE`, on a connection opened with `foreign_keys = ON`. SQLite
  ignores `PRAGMA foreign_keys` inside a transaction. So `DROP TABLE actions`
  would run its implicit delete against every row that references `actions`,
  and abort.
- **Eight columns reference `actions(id)`:**
  - `actions.parent_action_id`;
  - `runtime_operations.caused_by_action_id`;
  - `policy_decisions.action_id`;
  - `recovery_action_bindings.action_id`;
  - `recovery_execution_receipts.action_id`;
  - `numerical_recovery_action_bindings.action_id`;
  - `training_interventions.action_id`;
  - `numerical_recovery_executions.action_id`.

The rebuild mode follows SQLite's documented procedure for other schema
changes. It must be explicit per migration, never the default:

1. Set `PRAGMA foreign_keys = OFF` *before* `BEGIN IMMEDIATE`.
2. Run the file.
3. Run `PRAGMA foreign_key_check`, and roll back if it returns any row.
4. Insert the ledger row, then `COMMIT`.
5. Restore `foreign_keys = ON` on success **and** on failure, in a `finally`.
   Then read `PRAGMA foreign_keys` back and refuse to continue unless it
   returns `1`. A connection left with enforcement off would silently accept
   every later orphan.

The daemon's flock and the controller lease (ADR-004 §8) keep any other
writer out for the whole procedure.

**M3. The rebuild procedure, for each rebuilt table.** In one transaction:

1. **Drop dependents.** Drop every trigger and view whose SQL references the
   table, including triggers on *other* tables.
   - For `actions`, these are:
     - `policy_decisions_belong_to_their_action` (009);
     - `recovery_action_binding_authority` and
       `recovery_execution_receipt_ownership` (013);
     - `numerical_recovery_action_binding_authority` and
       `training_intervention_authority` (014).
   - For `budget_ledger`, they are its own two triggers (008).
   - The list is taken from `sqlite_master` when the migration is written,
     and is pinned by a test (M5).
2. **Create the new table.** Create `<table>_0NN` with the same columns in
   the same order, including columns added later such as
   `actions.parent_action_id` (004). It keeps the same keys, defaults,
   `REFERENCES` clauses and other `CHECK`s. Only the `target_kind` or
   `subject_kind` list changes.
3. **Copy the rows.** `INSERT INTO <table>_0NN (<explicit columns>) SELECT
   <same columns> FROM <table>`. Rows are copied as they are: none is
   rewritten, relabeled or dropped.
4. **Swap the tables.** `DROP TABLE <table>`, then `ALTER TABLE <table>_0NN
   RENAME TO <table>`. The rename runs after step 1, so SQLite's schema
   re-parse finds no dangling reference, and `legacy_alter_table` stays off.
5. **Recreate dependents.** Recreate every index, trigger and view from its
   current definition, verbatim. This includes the immutability and
   permanence triggers. The transaction is never committed without them, so
   no other connection ever sees the table unguarded.
6. **Check foreign keys.** Run `PRAGMA foreign_key_check`, as in M2.

**M4. Invariants and existing rows.**

- **Keys:**
  - primary keys and ids are copied unchanged, so every referencing row
    still points at the same row;
  - every `REFERENCES` clause is the same;
  - `foreign_key_check` proves that before the commit.
- **Triggers and views:**
  - ownership, authority, immutability and permanence triggers mean what
    they meant before, and are recreated text-for-text;
  - the only schema difference is the widened `CHECK`.
- **Existing rows:**
  - every training and evaluation row keeps its kind and its values;
  - no historical row is ever labeled with the new kind.
- **Historical request digests:**
  - `runtime_operations` is not rebuilt, so its stored `request_digest`
    column is untouched;
  - the plan classes a digest is recomputed from are unchanged: the target
    kind is widened by adding a value, and the `ExecutionSpec` union gains a
    member without changing either existing one;
  - so re-issuing a recorded training or evaluation operation computes the
    digest it recorded. H03's pinned literals must stay green across the
    migration, and M5 adds a test that recomputes stored digests.

**M5. Verification the follow-up must add.**

- A migration test against a fixture database that has rows in `actions`,
  `budget_ledger` and all seven other referencing tables. It checks that:
  - row counts and every column value are equal before and after;
  - `foreign_key_check` is empty;
  - `sqlite_master` is text-equal to the pre-migration schema except for the
    widened `CHECK`.
- The pinned training and evaluation request digests and execution
  fingerprints from H03, plus a test that re-resolves recorded attempts after
  the migration and compares their digests with the stored
  `request_digest`s.
- Fault injection: a failure at every step of M3 leaves the database exactly
  as it was, and still at the previous migration version. That includes a
  failure partway through the row copy and a failure while recreating
  dependent triggers and views. After each failure, `PRAGMA foreign_keys`
  reads back as `1` on that connection.

**M6. Rollback and order.**

- **Within the migration:** it is one transaction, so a failure at any step
  rolls back to the previous schema and data, and the ledger does not record
  it.
- **After commit:** there is no down-migration. A row of the new kind cannot
  be represented under the old `CHECK`. Rollback is restoring a backup that
  the runner takes immediately before any rebuild-mode migration, using
  SQLite's online backup API under the same lock.
- **Binary downgrade:** an older build would open a migrated database and
  read a target kind its types reject. Today `migrate()` silently proceeds
  when the database has migrations the build does not know.
  - **A guard only protects builds that contain it.** A guard shipped in a new
    release cannot stop a build that is already deployed. So the follow-up
    must:
    1. ship the guard (refuse a database whose newest migration the build
       does not know) in a release that comes *before* the rebuild;
    2. record that release as the **minimum compatible version** for the
       rebuilt database;
    3. test that the supported older deployments, from that release on,
       refuse an upgraded database.
  - **Builds older than the minimum** have no guard. Running the rebuild
    therefore requires a **coordinated upgrade**: every process that opens
    the database must be at or above the minimum first. The migration's
    release notes and the daemon's startup check must say so.
- **Order of the follow-up**, each step separately reviewed:
  1. the downgrade guard;
  2. the versioned candidate snapshot (ADR-018 D10), so a node can hold a
     harness;
  3. the runner's rebuild mode, `agent_attempts`, and the `actions` and
     `budget_ledger` rebuilds;
  4. the target kind, the `ExecutionSpec` member and the telemetry payload
     family;
  5. controller submission and reconciliation, with durable cancellation,
     restart and idempotency tests.

## Consequences

- No existing production module changes, so no recorded training or
  evaluation plan, request digest, execution fingerprint, candidate or
  harness fingerprint moves. Pinned regression tests hold each of them.
- A harness can now be compiled and run deterministically end to end in
  tests. It cannot be submitted to a controller, which is accurate.
- The durable path is a separate, reviewed milestone. Its migration is
  designed above instead of being assumed, and its acceptance criteria are
  listed as deferred, not as passed.

## Alternatives considered

- **Add `agent-attempt` now and leave storage behind:** rejected. It creates a
  domain object the repository cannot persist, and an import-time pairing
  check that would have to be bypassed.
- **Reuse `evaluation-attempt` for harness runs:** rejected. It carries
  `EvaluationExecutionSpec` semantics (a subject and an evaluation
  fingerprint) that a harness run does not have, and it would put harness
  runs in evaluation tables.
- **Stream telemetry under `xaytune.telemetry/v1alpha3`:** rejected. Its
  payload families are training and evaluation; a harness family would be a
  new protocol version, which belongs with the durable path.
- **Run a real model in the reference runner:** rejected for H03. It would
  make the runner nondeterministic and need credentials. Real adapters are
  H08–H11.
