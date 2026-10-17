# Changelog

## Unreleased

## 1.0.0a2

The second pre-release of the experiment control plane. `pip install xaytune`
still installs `0.6.0`; install this one with `pip install "xaytune==1.0.0a2"`
(or `pip install --upgrade --pre xaytune`). Expect the control-plane API to
change before `1.0.0`. The legacy trainer API from `0.6.0` is still included.

`1.0.0a1` decided one candidate. This release closes the **adaptive loop**: a
missed target branches, a planner proposes the next candidate, branching
records it, and the controller trains it, all from one submission and all on a
record a restarted controller resumes from. Around it are the pieces that make
that loop safe to run: budgets on a durable ledger, typed actions under a
policy engine with human approval, incident detection, checkpoints and governed
recovery (CUDA OOM and numerical instability), a local daemon with a CLI and
durable controller leases, and Ray, KubeRay, Ray Tune and TorchFT integrations
behind the existing runtime, search and resilience boundaries. Studio gains a
read-only HTTP API and a React web client, with loss charts from a durable
metric store.

The **agent harness track** (H02–H07) ships as local, in-process building
blocks: harness candidates and their identity, a compiler and deterministic
runner, pinned benchmarks, trajectory artifacts, harness evaluation and a
local harness search. **None of it is durable yet.** A harness cannot be
submitted to a controller; that is the next milestone.

**Upgrading from 1.0.0a1.** Opening a `1.0.0a1` state database applies
migrations 008–020 automatically, in order, each in its own transaction.

- **This is the minimum compatible version for later releases.** It is the
  first release with the downgrade guard: a build refuses a database that a
  newer Xaytune migrated (`SchemaTooNewError`; the daemon exits with status
  6), before reading or writing anything. Later releases add migrations that
  older builds cannot read. Upgrade every process that opens a database to
  this release first, then to a later one; `1.0.0a1` has no guard and would
  open a newer database regardless.
- **Upgrade with no unconfirmed submissions in flight.** A submission's
  request digest covers its plan, and the plan names the Xaytune release
  that built it. So a submission that `1.0.0a1` recorded but never delivered
  (the controller stopped before the runtime received it) is not re-issued by
  this release: reconciliation refuses it as a different request
  (`ImplementationMismatchError`) and leaves it as recorded. A workload the
  runtime has, running or found by lookup, is adopted as usual. Stop a
  controller only once its submissions are confirmed.
- **Breaking:** the Gradio Studio's Code tab is removed, and `xaytune studio`
  listens on `127.0.0.1` by default (see *Security*).
- `ExperimentNode.run_ids` and `evaluation_run_ids`, `Experiment.active_node_ids`
  and `Run.attempt_ids` / `final_attempt_id` are removed; the repository's
  relationships are the only source of topology (see *Changed*). Stored
  records still read.

**Known limitations of this alpha.**

- Agent harness candidates run locally and in-process only (above).
- Budgets enforce runs, failures, parallelism and wall time. GPU-hours,
  tokens and cost are refused at submission rather than pretended.
- The LLM planner proposes typed actions; the controller records and
  escalates them, and carries none out.
- TorchFT is a resilience *request* on a training plan. No built-in runtime
  hosts it yet, so every built-in runtime refuses one.
- Decisions compare recorded values with thresholds; nothing is claimed
  statistically. Harness comparisons are exploratory.
- Workers can occasionally hang while loading a model
  ([#35](https://github.com/szaher/xaytune/issues/35)).

### Security

- **Studio no longer runs arbitrary Python or listens on every interface** (UX-00). Studio has no authentication, so anyone who could reach it could use everything it served.
  - **Breaking:** the Code tab and `xaytune.studio.code_runner` are removed. They ran any submitted Python inside the server process, and Gradio exposes every UI callback as an API endpoint, so hiding the tab would not have been enough. **Generate Code** still writes a training script for you to run yourself.
  - `xaytune studio` and `xaytune.studio.server.launch()` now listen on `127.0.0.1` by default instead of `0.0.0.0`. `--host` still accepts another address and logs a warning that Studio is then reachable without authentication.
  - `--share` is refused (exit code 2), and `launch(share=True)` raises `StudioExposureError` before Gradio starts. A public link would let anyone on the internet start jobs on the machine.
  - **Trust Remote Code** runs a model repository's own Python. `create_app()` takes `allow_remote_code` (default `False`), and `xaytune studio` turns it on only for a loopback host. When it is off, the checkbox is locked and the submit endpoint refuses such a job before logging in to the Hub or submitting anything, so a direct API call cannot bypass it.

### Added

- **Xaytune Studio web client** (UX-02, ADR-023). `xaytune studio-web --state state.db` now serves a React + TypeScript client at `http://127.0.0.1:8765/`, beside `/api/v1`, from the same origin and behind the same front door.
  - Views: Overview, Experiment Explorer, Experiment (candidate graph with lineage and decisions, runs, decisions, actions, activity with incidents and recovery, budget, configuration), Candidate, Run, Evaluation run and Compare. Monitoring, Trajectories and Optimization say what they wait for.
  - The URL is the state; absent values are never shown as zero; recorded text is never rendered as markup; the comparison subtracts only values the API marks comparable.
  - Its types are generated from the committed OpenAPI document, and CI fails if they drift.
  - Browser tests (Playwright) record a real experiment through a real controller host and drive `xaytune studio-web` over it, including a restart mid-test.
  - The wheel carries the built client, so no Node is needed to run it. A source checkout serves the API only until `npm run build` is run in `web/`.

- **Studio API: a read-only HTTP view of the control-plane record** (UX-01, ADR-023). `xaytune studio-web --state state.db` serves `/api/v1` from the `studio-web` extra (`fastapi`, `uvicorn`).
  - Experiments, the candidate graph, candidates, runs, attempts, evaluation runs and results, decisions, actions with policy decisions, budget, incidents with recovery episodes, planner invocations, the event journal, and a two-candidate comparison, all read through the existing stores.
  - Read-only by construction: `mode=ro` plus `query_only`, never migrated, `GET`/`HEAD` only. Stateless, so restarts lose nothing. One snapshot per response. The schema version is checked per request (`503` with the reason).
  - Loopback only, with a `Host` allow-list against DNS rebinding, no CORS, a CSP and `no-store`. Errors are `problem+json` and never echo ids, values, paths or exception text. Actor metadata is never returned.
  - `ETag`/`304` and `X-Xaytune-Event-Sequence` for polling; keyset pagination with oversized limits refused.
  - `/compare` keeps lineage (`lineage.distinct_candidates`) apart from measurement comparability. Each metric pair shows both sides' evaluation fingerprint and unit, and gives a `difference` only when they match; otherwise it is `comparable: false` with `reasons`. Harness comparisons (ADR-022 C1–C4) stay H06's.
  - The OpenAPI document is committed (`xaytune/api/v1/openapi.json`). A test fails on any drift in paths, parameters, request and response schemas, types, formats, nullability or references; only presentational text is ignored.
  - New store reads: `AggregateStore.list_experiments()`, `EventJournal.latest_sequence()`, `recorded_versions()` (reads the migration ledger without creating it) and `xaytune.core.sqlite.connect_read_only()`.
  - The legacy Gradio Studio's tests now run in CI (a `studio` job). Its six stale `TestPollJob` tests are updated to `_poll(mgr, job_id)`.

- **Harness evaluation, scoring and comparison** (H06, ADR-022). Local and in-process; no migration, and `Objective`, the decision engines and every existing fingerprint are unchanged.
  - **Task binding:** `bind_task()` and `HarnessCompilationContext.task_binding` carry a pinned benchmark task into a spec's `config["task"]`. The compiler checks it against the resolved suite and refuses other variables, a wrong fingerprint or position, attachments, and a bound spec that does not record. The trajectory attribution records the task's revision, fingerprint and cohort. Unbound specs are byte-identical to H03's.
  - **Evidence:** `assemble_evidence()` builds `HarnessRunEvidence` (artifact kind `agent_run_evidence`) for one execution replicate from the store: every attempt of every task, each entry `attested` or `no-attested-result`. `verify_run_evidence()` rebuilds every fact from the suite and the store and refuses, with every reason, omitted, added, open or quarantined attempts, empty entries, mismatched attribution, unverifiable trajectories, hidden or downgraded results and a wrong digest.
  - **Evaluators:** `harness-outcome` (`DETERMINISTIC`) and `harness-judge` (`STOCHASTIC`, via `invoke_agent_model`) on the existing `Evaluator` protocol, with a canonical typed config, run through `run_harness_evaluation()`, which walks the existing evaluation run and attempt state machines. Every run verifies its subject first.
  - **Metrics:** the `xaytune.harness-metrics/v1` catalog with units and absence reasons; zero never stands in for an absence. A stop with no answer scores `task_success` 0 and leaves `quality` absent (`not-applicable`), never imputed. Per-task reference-derived scores and judge exchanges are restricted; a `ReferenceEgressGrant` is required before a judge sees a reference, checked at `prepare()` and again at execution against the host's grants. Under the default `aggregate` disclosure, a single-task reference-derived run value and every `successful-tasks` slice are withheld; `per-task` releases them. `provider_feedback()` never carries a per-task reference-derived score, a reference-derived run value over one task, or a `successful-tasks` slice.
  - **Comparison:** `summarize()` (`replicate-mean/v1`), `comparability()` (C1–C4, every failure) and `compare()` (`paired-difference/v1`, an exploratory verdict per metric and the vector relation).

- **Policy and approval for proposed actions** (PR-023). `ExperimentHandle.propose(spec, reason=..., proposed_by=...)` validates a typed action, has the host's `PolicyEngine` judge it, and records the result. Nothing is carried out.
  - Validation is built in and comes first. It checks that the experiment is active, the target exists in it and the target is in a state the type can act on. `change-worker-count` also needs the runtime to declare that its worker count can change (`elasticity.supported`) and a count within every range it declares; otherwise it is refused.
  - `PolicyVerdict` is `ALLOW` (the action stays `VALIDATED`, authorized), `DENY` (`REJECTED`) or `REQUIRE_APPROVAL` (`APPROVAL_PENDING`). `RulePolicyEngine` applies the first matching rule, by action type or mutation class, then its default. **With no policy configured, every proposal is denied**, with a recorded decision.
  - Policy sees the proposer as `PolicyProposer` (type and id), never the actor's metadata, and every nested model and mapping (`parameters`, `provider`, capability `extensions`) is rebuilt as its exact identified type, so a subclass's newer fields or attributes never reach a policy: everything a policy can read is in the snapshot's fingerprint. A proposal signed by an engine other than the one asked is refused.
  - Migration 009 adds an append-only `policy_decisions` table. Each decision records its engine, rules, verdict and reasons, and the full snapshot it judged, including the runtime's capabilities, fingerprinted. It is written in one transaction with its action, and never against a state that changed while the policy was judging.
  - `host.approve_action()` and `host.reject_action()` accept only a human actor, approve the recorded proposal without judging it again, and never rewrite an earlier answer. A replay by the same human (type and id), with the same answer and reason, changes nothing, whatever the actor's metadata.
  - Cancellation is never proposed (`CancellationNotGovernedError`); it keeps its own path.
  - An action waiting for approval or execution does not keep `wait()` from returning. `ExperimentResult.next_stage` gains `"action-approval"` and `"action-execution"`, which take precedence over the stages derived from the candidates.
- **Typed, versioned actions** (PR-022). Every Action type has a registered schema (`ActionSpec`) and a declared `MutationClass`: `OPERATIONAL`, `SCIENTIFIC_INTERVENTION` or `EXPERIMENT`. They record intent only; nothing is carried out except cancellation, as before.
  - Built in: `resize-microbatch`, `change-gradient-accumulation`, `change-worker-count` (logical workers, runtime-neutral), `change-checkpoint-interval`, `change-learning-rate`, `change-scheduler`, `change-warmup`, `reject-candidate` and `promote-candidate`, plus the three cancellation types.
  - `action_from_spec()` builds the Action, and `spec_of()` reads the typed spec back. The payload is a canonical `{"schema_version", "parameters"}` envelope that repeats neither type nor target. Cancellations keep their `{}` payload as version 1, so existing records are byte-for-byte unchanged.
  - A new Action is validated against its schema before it is written. Loading one never needs its schema, so history outlives an uninstalled plugin; `spec_of()` fails closed instead.
  - Plugins register typed actions with `register_action(ActionDescriptor.for_spec(...))`, optionally with a static validator. Their actions record the plugin's contract (provider, plugin name, API version), but not its plugin version, so a compatible upgrade reads and retries what an older one wrote. There is no entry-point discovery.
  - **Breaking:** `register_action_type(name)` now raises, because a type without a schema could record payloads nothing can check. `SupportResult` moved to `xaytune.core.support`; `xaytune.compilation` still exports it.
- **Experiment budgets on a durable ledger** (PR-016). `ExperimentSpec.budget` is recorded with the experiment, and `ExperimentResult.budget` reports each limited dimension's limit, reserved, committed, consumed and remaining.
  - Migration 008 adds an append-only ledger. The database refuses updates and deletes, every amount is positive, and each (subject, dimension, kind) is written once. Each entry commits in the same transaction as the change that causes it, and balances are derived from the entries rather than counted: `outstanding = max(reserved − consumed − released, 0)`, `remaining = limit − consumed − outstanding`. A commit subtracts nothing a second time.
  - `max_runs` is a hard quota per **run**: reserved when the run is recorded, committed when the runtime accepts it, consumed when it ends, and released if it never reached the runtime. `max_failures` is a hard quota: one per `FAILED` training or evaluation attempt, not preempted or cancelled ones. `max_parallel_runs` is a capacity: a full one makes the next attempt wait, and never exhausts the experiment.
  - A used-up quota refuses the next run, attempt or evaluation before anything is written. The experiment becomes `BUDGET_EXHAUSTED`, with a `BudgetExhausted` event, only once nothing it owns is still running. Passing a limit, rather than reaching it, is recorded as a `BudgetOverrun` event; nothing is stopped for it.
  - `max_wall_time_seconds`, `max_gpu_hours`, `max_tokens` and `max_cost` are refused at submission (`UnsupportedBudgetError`): nothing measures them authoritatively yet. Attempt timestamps are when the controller observed a change, so wall time would be wrong across a restart.
  - `BudgetSpec` now validates its limits: finite and non-negative, and `max_parallel_runs` at least 1.
- **The `lm-eval` evaluator** (`xaytune.evaluation.lmeval`, worker `xaytune.workers.eval_lmeval`), registered by default. It runs one lm-eval-harness task on the trained model, such as `arc_easy` or `hellaswag`, and reports `acc` and `acc_norm`, each with lm-eval's standard error and the number of documents actually scored (after `limit`). It is `SEEDED`: the run's seed is lm-eval's Python, NumPy, Torch and few-shot seed.
  - A task name is **resolved at submission** into a binding: the task's definition by digest, under lm-eval 0.4.13 exactly, and its Hub dataset by commit. The binding is stored in `evaluator.config["binding"]`, so the evaluation fingerprint names the pinned task, not the name. The worker refuses to run if the installed definition no longer matches the digest, and loads the dataset at the recorded commit.
  - It accepts one registered, YAML-defined task scored by log-likelihood (`multiple_choice` or `loglikelihood`) that reports only `acc` and `acc_norm`. It refuses groups and tags, tasks implemented in custom Python, unsafe-code tasks, datasets that need remote code or cannot be pinned to a commit, and generation tasks. Generation brings decoding settings, stop sequences, filters and answer extraction that no binding pins yet.
- **`ResolvableEvaluator.resolve(spec)`**, an optional evaluator step and the only one allowed to use the network. The host calls it once, at submission, between two `supports()` checks: `supports(declared) → resolve() → supports(resolved)`. The resolved spec is what is recorded and fingerprinted. After a restart the request is rebuilt from the record, and `resolve()` is never called again. The `Evaluator` contract itself is unchanged: an evaluator without `resolve()` has nothing to pin, its spec is recorded as declared, and custom evaluators keep working as they are.
- **`examples/control_plane/06_train_and_benchmark.py`**: train, run an lm-eval task, and decide on `acc`.

- **The adaptive loop** (PR-015b, PR-024, PR-025, PR-026). One submission runs to completion with no further calls: train, evaluate, decide, plan, branch, train the next candidate.
  - `DecisionOutcome.BRANCH` and `AdaptiveThresholdDecisionEngine` (`adaptive-threshold` 1.0.0): a missed target with constraints held finishes the candidate (`COMPLETED`) and leaves the experiment `ACTIVE` with `next_stage == "planning"`. `ThresholdDecisionEngine` 1.0.0 is unchanged and stays the default.
  - `RuleBasedPlanner` (`rule-based` 1.0.0): a pure planner over the repository's read-only `planning_context()`, with a LoRA-rank mutation rule. It proposes a `CandidateProposal` with full provenance and never writes.
  - Branching: `materialize_candidate_proposal()` turns a verified proposal into a `PLANNED` child node with a repository-minted `branch_origin`, in one transaction. A replayed proposal returns the same node, and a different one for the same candidate is a conflict.
  - Realization: the controller trains a branched node automatically, inheriting the parent run's seed (`RunSeedOrigin`), and resumes from the record after a restart at either new resting point.
- **Incidents, checkpoints and governed recovery** (PR-017 to PR-021b).
  - Incidents (migration 010): deterministic detectors classify structured failure observations (process failure, CUDA OOM, checkpoint failure, NaN/Inf) into immutable, replay-safe incidents with their evidence and detector provenance.
  - Checkpoints (migration 011): `SerializedStateCodec`, `LocalCheckpointStore` and `CheckpointManager` package, publish atomically and verify checkpoint bundles; committed checkpoints are recorded from telemetry.
  - `RecoveryEpisode` and `RecoveryPlan` (migration 012): one decision-only coordinator per failed attempt, with append-only evidence and plan revisions.
  - CUDA OOM recovery (migration 013, execution receipts): a governed `ResizeMicrobatch` action halves the micro-batch and doubles gradient accumulation (the effective batch is kept), and a successor attempt resumes from a FULL + EXACT optimizer-boundary checkpoint. `LocalRuntime` restores the single-worker Native trainer exactly.
  - Numerical recovery (migrations 014 and 015): a governed `ChangeLearningRate` action becomes a `TrainingIntervention` on the same run, applied by a successor attempt restored from a checkpoint before the unsafe step, and replayed after rollback. There is no default reduction factor; a policy must declare one.
- **A local daemon and its CLI** (PR-027 to PR-029). `python -m xaytune.daemon --state … --config module:factory` runs one controller over a state database, singleton by `flock`, with a SQLite request mailbox (migrations 016 and 018), atomic admission and controlled shutdown.
  - A durable controller lease (migration 017) fences every repository write, and a restarted daemon reconciles every experiment it owns.
  - `LocalDaemonControllerHost` is the caller's side: the same `ExperimentHandle` over the daemon. The CLI: `xaytune submit`, `attach`, `status`, `watch`, `events`, `results`, `actions`, `cancel`, `approve` and `reject`.
- **Agent models and the LLM planner** (PR-030 to PR-032). `xaytune.agent` defines `AgentModel`, canonical `AgentModelRequest` identity and `ScriptedAgentModel`. Provider exceptions never cross the boundary: only their type name does.
  - `LLMPlanner` proposes zero or one typed, explicitly allowed action, with the model's identity in its spec. It has no default registration; a host binds it explicitly.
  - Every agent call is a durable `AgentInvocation` (migration 019), recorded as intended before the model is asked, then answered, then completed or refused.
- **Ray, KubeRay, Ray Tune and TorchFT** (PR-033a to PR-035), each behind an existing boundary.
  - `RayJobsRuntime` (`ray-jobs`) runs each plan as one supervised worker on an existing Ray cluster. `RayTrainRuntime` (`ray-train`) runs training as a Ray Train worker group at every size.
  - `KubeRayJobsBackend` submits either runtime's jobs as KubeRay `RayJob` resources, with deterministic names, through the Kubernetes API.
  - A generic search contract (`SearchSpace`, `SearchProvider`, `SearchContext`) and `RayTuneSearchProvider`. Ray Tune proposes parameters; Xaytune owns the candidates, lineage and execution.
  - A generic resilience contract (`ResilienceSpec`, `ResiliencePolicy`) and `TorchFTResilienceProvider`, which adds one checked request to a training plan. No built-in runtime hosts it yet.
- **Agent harness building blocks** (H02 to H07), local and in-process; nothing is persisted or submitted.
  - H02: `AgentHarnessSpec` and its versioned fingerprint, and the `ExperimentCandidate` envelope, which carries training candidates byte for byte.
  - H03: `ReferenceHarnessCompiler`, `HarnessExecutionSpec` and a deterministic in-process runner (ADR-019 E1–E10).
  - H04: pinned benchmark suites and tasks, resolution and seeded assignment (ADR-020).
  - H05: trajectory artifacts: a crash-safe writer, a local store, verification, reconciliation, retention and replay (ADR-021).
  - H07: harness search (ADR-024 §P12). Typed `HarnessPatch` operations that only narrow tools, `HarnessSearchPolicy` with explicit `autonomous_search: authorized`, provider-safe feedback and comparisons, two deterministic providers, verification of every proposal, and a local search loop under a spending envelope that reserves each unit of work before it starts. Branching a harness proposal into an experiment is refused.
- **The metric-point store** (migration 020). Training telemetry's scalars are recorded while the controller ingests each attempt's stream: idempotent on the stream position, provenance kept, the telemetry cursor untouched. `GET /api/v1/attempts/{id}/metrics` serves them (one series per name, unit, worker and rank, downsampled at read time), and Studio's run page charts them, polling until the run settles.
- **The downgrade guard** (ADR-019 M6, step 1). `migrate()` refuses a database whose ledger records a migration this build does not ship, raising `SchemaTooNewError` (a `MigrationError`) before anything is written, from every entry point that migrates. `python -m xaytune.daemon` exits with status 6 and the refusal as its message.

### Changed

- **Topology has one source of truth** (OQ-15). `Experiment.active_node_ids`, `ExperimentNode.run_ids` and `evaluation_run_ids`, and `Run.attempt_ids` and `final_attempt_id` are removed; nothing ever wrote them. Topology comes from the repository's relationships. A stored record holding a retired field at its never-written default still reads; one holding a value fails closed.
- **ADR-017, reuse policy v1:** training artifact reuse is disabled. A matching fingerprint is evidence, never authority to skip a run.
- `PolicyContext` normalizes every subclass to exactly the v1 view, so what a policy engine can read is what the policy input fingerprint covers.
- The `eval` extra pins `lm-eval[hf]==0.4.13` (it was `lm-eval>=0.4`). An lm-eval release defines the tasks it ships, so the evaluator binds and runs tasks under that release only. The legacy `benchmark_evaluate()` uses the same extra.

## 1.0.0a1

The first release of the experiment control plane, and a **pre-release**:
`pip install xaytune` still installs `0.6.0`; install this one with
`pip install "xaytune==1.0.0a1"` (unpinned, and upgrading an existing
install: `pip install --upgrade --pre xaytune`). It closes one
loop end to end: declare an experiment and a candidate, train it, evaluate
the trained model, and make a durable, deterministic decision, with every
step on a record that a restarted controller resumes from.

Expect the control-plane API to change before `1.0.0`. The legacy trainer API
from `0.6.0` (`finetune`, `align`, `evaluate`, the CLI and pipelines) is still
included.

**Upgrading from 0.6.0.** Nothing is removed. Two changes can alter what an
existing program sees:

- Metrics from `xaytune.eval.evaluate()` and in-training evaluation change,
  because the earlier values were wrong (see *Fixed — evaluation metrics*).
- Top-level imports are lazy: `import xaytune` no longer imports torch (see
  *Changed*).

**Known limitations of this alpha.**

- One runtime, `LocalRuntime`: workers are processes on the controller's
  machine. Ray, Kubernetes and other runtimes are planned.
- One candidate per experiment. There is no planner, branching, budget or
  policy gate yet. An experiment whose candidate is rejected stays `ACTIVE`
  with `next_stage == "planning"`, and nothing proposes the next candidate;
  cancel it to end it.
- Decisions compare point estimates with thresholds; nothing is claimed
  statistically. An objective without a target is not decided.
- The built-in evaluator is `native` (next-token loss, perplexity and token
  accuracy). lm-eval benchmarks are a planned integration.
- The TRL trainer supports exactly `trl` 1.13 and `transformers` 5.17.
- Workers can occasionally hang while loading a model
  ([#35](https://github.com/szaher/xaytune/issues/35)).

### Added — the experiment control plane

- **Durable persistence** (`xaytune.storage`): SQLite, versioned migrations 001–007, revision-based optimistic concurrency, and atomic state + event + outbox transactions. A `RuntimeOperation` journal records every external effect before it is attempted, and Actions record requests such as cancellation together with their outcomes.
- **`CandidateSpec` and versioned fingerprints**: `CandidateFingerprint`, `RunHistoryFingerprint`, `ArtifactLineageFingerprint`.
- **Compile/execute boundary** (`xaytune.compilation`): `TrainerCompiler`, `TrainingExecutionSpec` and `ResolvedExecutionPlan`, with `NativeCompiler` and `TRLCompiler` for full-parameter SFT. A candidate that leaves a training-relevant value undeclared is refused, with every reason.
- **`LocalRuntime`** (`xaytune.runtimes`): workers run as separate processes, submission is idempotent through `submit_or_get`, and telemetry is versioned (v1alpha2 for training, v1alpha3 for evaluation) with a durable cursor.
- **`EmbeddedControllerHost` and `ExperimentHandle`** (`xaytune.experiment`): `submit`, `status`, `wait`, `cancel`, `events`, and `attach`, which adopts running work after a restart and never submits it twice.
- **The durable evaluation lifecycle**: `EvaluationSpec`, evaluation runs, attempts and results, evaluation cycles, and the `Evaluator` contract.
- **`NativeEvaluator`, the built-in evaluator** (`xaytune.evaluation.native`, worker `xaytune.workers.eval_native`), registered by default as `native`. It measures next-token loss, perplexity and token accuracy (token-weighted, logits at *i* against the token at *i + 1*) on a local JSONL file of held-out text, pinned by its content digest (`local_dataset()`). The worker refuses a file whose bytes changed. It declares itself `SEEDED`, records the run's seed and evaluator version on every metric, and writes a report naming the device and library versions. Everything that changes a number (format, truncation, batching, precision, metrics) is required configuration, part of the `EvaluationFingerprint`.
- **`Evaluator.supports(spec)`**: the host asks at submission, so an evaluation that cannot run exactly as declared is refused before anything trains, with every reason. A refusal from `prepare()` after training (a subject the evaluator cannot read) now fails the evaluation run with its reasons and reports the cycle stalled, instead of escaping the controller. `UnsupportedEvaluationError` carries `evaluator` and `reasons`.
- **`examples/control_plane/05_train_and_evaluate.py`**: train, evaluate the trained model, and decide against a loss `--target`.
- **Durable decisions (PR-015)**: `DecisionContext`, `DecisionProposal` and `Decision` (`xaytune.core.domain.decision`), and the `DecisionEngine` contract with the built-in `ThresholdDecisionEngine` (`xaytune.decision`). The engine is pure: it returns a proposal with no id, time or actor, which the repository adds when it records it. The input fingerprint is an explicit, versioned projection of the evidence (`decision_input_identity_v1`).
  - The engine decides from the record alone: the objective, and the evaluation results of the candidate's current cycle. It compares recorded values with the target and constraints (all six operators, applied exactly). A violated constraint gives `REJECT`, a target met gives `STOP_SUCCEEDED`, and a target missed gives `STOP_FAILED`.
  - Without a target, or with a metric missing, nothing is decided: the candidate stays `DECIDING`, with a `DecisionDeferred` event.
  - Migration 007 records each decision, with its evidence and an input fingerprint, in one commit with what it causes: `STOP_SUCCEEDED` completes the node and succeeds the experiment, setting `best_node_id`; `STOP_FAILED` rejects the node and fails the experiment; `REJECT` rejects the node and leaves the experiment `ACTIVE`.
  - `next_stage` gains `"planning"`: the experiment is `ACTIVE` and every candidate was rejected on its merits. It comes only from that scientific outcome; a failed or cancelled candidate is `"failure-handling"`, even beside a rejected one.
  - Idempotent per cycle: a different decision for a decided cycle is refused, and a controller that restarts decides once.
  - Attributable: recording recomputes the input fingerprint from the stored objective and the cycle's results, and refuses a proposal whose fingerprint differs, that names a result twice, or whose evidence cites a result outside the cycle. The outcome itself is not re-checked; decision policy belongs to the engine.
  - `EmbeddedControllerHost` takes an optional `decision_engine`.
- **`examples/control_plane/`**, and control-plane *Getting started* and *Concepts* pages in the docs.
- **A release gate**: publishing a GitHub release fails unless its tag is `v` plus the `pyproject.toml` version, and `xaytune/_version.py` agrees (`scripts/check_release_version.py`). Manual TestPyPI dispatches are not gated.
- **One runtime version**: `xaytune/_version.py`, an import-free module. `xaytune.__version__` and the `xaytune_version` of every built-in plugin descriptor (`NativeCompiler`, `TRLCompiler`, `LocalRuntime`, `NativeEvaluator`) come from it instead of hardcoded literals, and a test refuses any new literal.
- **An sdist allow-list**: the source distribution contains `xaytune/`, `pyproject.toml`, `README.md`, `CHANGELOG.md` and `LICENSE`, whatever else is in the working tree. `scripts/check_sdist.py` checks the built archive in CI and before every upload, including that `LICENSE` ships and `pyproject.toml` declares `Apache-2.0`. The wheel is built from the sdist.
- **`LICENSE`**: the Apache License 2.0 text, verbatim, which the package already declared and did not ship.
- **A wheel smoke test** in CI and before every upload. The built wheel is installed into a fresh virtualenv, then must import without torch, transformers or TRL, migrate a new store, compile a candidate, round-trip the plan, and import the control-plane examples (`scripts/wheel_smoke.py`).

### Fixed — evaluation metrics (correctness)

- **`token_accuracy` scored the wrong token** in `xaytune.eval.evaluate()` and in the in-training evaluation (`eval.every_n_steps`). It compared the logits at position *i* with the label at *i*. In a causal LM those logits predict the token at *i + 1*, so a model that merely echoed its input scored perfectly: an untrained test model reported 1.0 where its next-token accuracy was 0.0. It now compares each prediction with the next label, and labels of `-100` count in neither the numerator nor the denominator (issue #36).
- **`loss` and `perplexity` depended on batching.** They averaged per-batch losses, so a batch with 3 next-token targets weighed as much as one with 300. Each batch's loss is now weighted by its target count, which gives the mean over every target in the dataset: the same examples batched differently give the same answer. `perplexity` is `exp` of that loss. A batch with no next-token target contributes nothing.
- **Reported values change.** This is a correctness fix, not an API change: the functions, their arguments and the metric names are unchanged, and the earlier values were wrong. `compute_loss` and `compute_perplexity` take an optional `weights`, one non-negative target count per loss: a mismatched or negative weight raises `ValueError`, and weights summing to zero report `0.0`, as no losses always have. Without `weights` they average plainly, exactly as before.

### Changed — positioning and dependencies

- The package description is now "Agent-native experiment control plane for model post-training and adaptation". The README and docs separate what is available today from what is planned, and label the notebooks as the legacy trainer API.
- `packaging` is a declared dependency. The TRL worker imports it to check installed releases.
- The `trl` extra pins `trl>=1.13,<1.14` and `transformers>=5.17,<5.18`, the releases the TRL worker is classified against, and `uv.lock` resolves them.

### Added

- **`xaytune.core` — control-plane domain foundation.** Typed sortable identifiers (`ExperimentId`, `RunId`, …), the `Experiment` / `ExperimentNode` / `Run` / `RunAttempt` aggregates, their separate state machines, `ExecutionOverride`, and the immutable value objects (`Actor`, `DatasetRef`, `ModelRef`, `ArtifactRef`, `RuntimeRef`, `ControllerHostRef`, `CheckpointRef`, `ResourceUsage`). The package imports without torch, transformers, peft, trl, ray or kubernetes installed, enforced by `tests/test_core/test_architecture.py`. Nothing in the existing training path uses it yet.
- **`docs/api/core.md`** documenting the new package.

### Changed

- **Top-level imports are now lazy.** `xaytune.finetune`, `.pretrain`, `.align`, `.evaluate`, `.lr_find`, `.JobManager` and `.discover_plugins` resolve on first access (PEP 562) instead of at import time, so `import xaytune` no longer pulls in torch. Every public name behaves as before; `xaytune.pipeline` is still bound eagerly because it collides with the `xaytune/pipeline.py` submodule.

### Fixed

- **`xaytune pipeline` crashed on any stage that omitted `trainer:` or `lora:`.** `_run_train_stage()` used `TrainerConfig()` and `LoraConfig()` above their import, raising `UnboundLocalError`. The `or` short-circuit meant it only fired when a stage left those sections out. `xaytune/pipeline.py` had no test coverage at all; `tests/test_pipeline.py` now covers it.
- **`import xaytune.trainer` as the first xaytune import raised `ImportError`.** `xaytune.trainer` → `xaytune.eval` → `xaytune.recipes` → `xaytune.trainer` was a circular import masked by the eager package `__init__`; `eval_callback` now resolves the metric registry lazily.
- **`apply_lora()` skipped `prepare_model_for_kbit_training` silently** if the inline import failed. The import is now hoisted alongside the other peft imports.
- **`evaluate()` raised `IndexError` on non-tensor `labels`** while explicitly passing non-tensor batch values through the device move. Labels are now coerced before masking.
- **DeepSpeed + `resume_checkpoint_dir` raised `AttributeError`.** The DeepSpeed path sets `optimizer` to `None`, but the resume branch called `optimizer.load_state_dict()` unguarded while the adjacent scaler and scheduler branches both checked. The skip is now logged rather than passing silently, since DeepSpeed restores optimizer state through its own checkpoint API.
- **CPU-only distributed training raised `AttributeError`.** `init_distributed()` called `torch.cuda.set_device()` unconditionally, so a gloo process group could not start without CUDA.
- **ORPO produced `NaN` when a sequence log-probability reached 0** (BUG-011). The odds `p / (1 - p)` diverge as `p` approaches 1, and `p == 1` is reachable: a fully-masked sequence sums to exactly 0. Probabilities are now clamped just below 1; the clamp is a no-op for realistic inputs.

### Known issues

Two tests are recorded as strict `xfail` pending a product decision, not because of a defect:

- `constant` scheduler with `warmup_steps > 0` auto-upgrades to warmup behaviour, while `test_constant_ignores_warmup_steps` asserts warmup is ignored. Deciding for the implementation makes `constant_with_warmup` redundant; deciding for the test means a requested warmup is silently dropped.
- `global_step` counts optimizer steps, while `test_gradient_accumulation_reduces_optimizer_steps` expects micro-steps. The rest of the loop agrees with the implementation, but `global_step` is user-visible in checkpoints and logs.

### Changed

- **`xaytune.core` domain records are now deeply immutable, and payloads must be canonically persistable.** `FrozenDict` freezes at construction and is backed by a `MappingProxyType`, so an instance cannot contain a mutable container however it was built. Value objects inherit `FrozenDomainModel`, whose `model_copy(update=...)` re-validates rather than assigning — Pydantic's own version skips validation, which would otherwise hand back a plain mutable dict or an unvalidated id. The four aggregates inherit `AggregateModel`, which refuses updates outright: schema validation cannot check that a transition is legal or that the revision moved, so `model_copy(update={"status": ...})` would yield a `SUCCEEDED` attempt with no `started_at` and revision 0. Aggregates change only through transition methods such as `with_status()`. Non-string mapping keys, sets, `NaN`/infinity, `bytes` and arbitrary objects are rejected with `InvalidDomainValueError` — a record that cannot round-trip cannot be fingerprinted. Pydantic's `frozen=True` blocks attribute assignment but not mutation of the containers behind it, so `snapshot.payload["optimizer"]["lr"] = 7` succeeded on a supposedly frozen scientific record, and the model kept a live reference to the caller's dict. Mappings now become `FrozenDict` and sequences become tuples at validation time, recursively; `thaw()` returns a mutable copy. Note id lists and `metadata` are now tuples and mappings rather than `list`/`dict`.
- **The state machines cover failure and cancellation.** The first tables encoded only the transitions the architecture spec drew, which left real gaps: a node could not fail while `ACTIVE`, an attempt could not fail while `STARTING`, `CHECKPOINTING` or `RECOVERING`, and nothing could be cancelled before it started. Any non-terminal state can now reach `FAILED` or `CANCELLED`, and an attempt can be `PREEMPTED` from `QUEUED` onwards. `ExperimentNodeStatus` gains `CANCELLED`, which is distinct from `REJECTED` — the latter is a judgement on merit, reachable only from `DECIDING`.

### Notes

- `mypy` is configured at `python_version = "3.12"` (was `3.10`); numpy's bundled stubs use 3.12-only syntax, and CI already runs mypy under 3.12. This means mypy no longer verifies 3.10 compatibility while `requires-python` remains `>=3.10`.
- Markdown is excluded from `ruff`; ruff 0.16 formats Markdown code blocks, which the project never opted into.

## v0.3.0

### Added

- **Tokenization pipeline** (`tokenize_dataset()`, `collate_tokenized()`) — automatic tokenization of text-format data before training. Converts `{"text": "..."}` samples to `{"input_ids", "labels", "attention_mask"}` tensors.
- **Real model integration tests** — end-to-end tests using `sshleifer/tiny-gpt2` covering forward pass, gradient flow, loss decrease, Trainer loop, and eval-during-training.
- **`@pytest.mark.slow` marker** — integration tests that download models are deselected by default.

### Changed

- `setup_training()` now auto-tokenizes text-format datasets and uses a proper `collate_fn` for padding/batching.
- `validate_batch()` now requires `input_ids` — text-only batches are rejected since tokenization is handled upstream.

## v0.2.0

### Added

- **Algorithm-specific parameters** (`method_params`) — configure DPO beta, GRPO kl_coeff, PPO clip_eps, ORPO lambda_weight, SimPO beta/gamma via config, CLI, Python API, and Studio UI.
- **Studio Simple/Advanced mode** — toggle between minimal form (recipe, model, data) and full control with all training parameters.
- **Auto chat template** — tokenizer chat templates are automatically applied for `chat` and `sharegpt` data formats when a tokenizer is available.
- **Pre-flight validation** (`preflight_check()`) — checks GPU availability, quantization CUDA requirement, data path existence, and output directory writability before training starts.
- **Dynamic method params in Studio** — selecting an alignment method (DPO, GRPO, etc.) shows its configurable hyperparameters with defaults and descriptions.

### Changed

- `align()` one-liner now accepts algorithm kwargs directly (e.g., `align(model="m", dataset="d", beta=0.2)`).
- `build_config()` accepts `method_params` dict for Studio integration.
- All alignment example configs now include `method_params` with documented defaults.
- `setup_training()` passes tokenizer to `load_dataset()` for automatic chat template application.

## v0.1.0

### Added

- Recipe-based training: `finetune`, `pretrain`, `align` recipes with registry pattern.
- Alignment methods: DPO, GRPO, PPO, ORPO, SimPO, REINFORCE.
- Fine-tuning methods: full, LoRA, QLoRA.
- Pydantic config schema with YAML parsing, dot-notation overrides, and config inheritance.
- Cross-field config validation with actionable error messages.
- CLI: `train`, `list`, `eval`, `export` (merge/gguf/push), `compare`, `lr-find`, `studio`, `launch`.
- Training Studio: Gradio web UI with Train/Monitor/History tabs, live loss plotting.
- Data pipeline: format registry (alpaca, sharegpt, chat, text), sequence packing, eval splits.
- Evaluation: metric registry (loss, perplexity), lm-eval-harness benchmark integration.
- Export: LoRA merge, GGUF conversion, HuggingFace Hub push.
- Trainer: mixed precision, gradient accumulation, gradient clipping, LR schedulers.
- Checkpointing: periodic saves, save-last, async checkpoint, resume from checkpoint.
- Early stopping with configurable patience, metric, and min delta.
- LR finder with EMA smoothing and suggested LR.
- Distributed training: DDP, FSDP, DeepSpeed via `xaytune launch`.
- Logging backends: console, WandB, MLflow, TensorBoard.
- Progress bar with Rich.
- Python API one-liners: `finetune()`, `pretrain()`, `align()`.
