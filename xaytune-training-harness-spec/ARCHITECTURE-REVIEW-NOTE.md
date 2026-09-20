# Architecture Review Note — Xaytune 2.0 Control Plane

Status: review only. No code changed.
Scope reviewed: spec `README.md`, `01-product-and-scope.md`, `02-architecture.md`, `14-repo-refactor-map.md`,
`15-implementation-plan.md`, `17-coding-agent-contract.md`, ADR-001…ADR-010, and the current `xaytune` v0.6.0
repository (96 Python modules, ~13k LOC).

---

## 1. Current repository, mapped to the target architecture

| Existing module | LOC | Disposition per ADRs | Notes |
|---|---:|---|---|
| `xaytune/config/schema.py` | 373 | **Reuse, split** | Pydantic v2 models already separate `ModelConfig` / `LoraConfig` / `DataConfig` / `TrainerConfig` / `FSDPConfig` / `DeepSpeedConfig`. The scientific/runtime split ADR-006 wants is *almost* already there: `Model`+`Lora`+`Data` → `TrainingSpec`; `Trainer.strategy/mixed_precision` + `FSDP`/`DeepSpeed` → `TrainingExecutionSpec`. `TrainerConfig` is the one class that straddles the boundary (LR/epochs/seed are scientific; batch_size/grad_accum/activation_checkpointing are executional — and they are exactly the fields `ExecutionOverride` mutates). |
| `xaytune/trainer/loop.py` + `callbacks.py` + `checkpointing.py` + `distributed.py` | ~650 | **Reuse behind `NativeCompiler`/`NativeWorker`** (PR-010) | `Trainer.train()` is already a self-contained loop driven by `TrainState` + `CallbackManager`. It becomes the worker entrypoint, not the control-plane API. |
| `xaytune/recipes/finetune.py`, `pretrain.py`, `align/` | ~1500 | **Keep; become `TrainingSpec` builders** | `align/` (PPO, DPO/ORPO loss dispatch, rollout buffer, rewards) is the newest and most valuable code — do not disturb it during Phase 1–2. |
| `xaytune/eval/` | ~330 | **Wrap behind `Evaluator`** (PR-014) | `metric_registry` exists; return shape is plain `float` and must become `MetricResult` (ADR-007). |
| `xaytune/logging/` (`base.py` `LoggingBackend`/`LoggingManager`, console/tb/wandb/mlflow) | ~230 | **Reuse as outbox event sinks** | The `LoggingBackend` ABC is a near-exact fit for the outbox consumer contract in `07-persistence-and-events.md` §7–8. |
| `xaytune/studio/events.py` (`EventBus`, `TrainingEvent`, `register_event_callbacks`) | 77 | **Reuse as in-process bus** | Already bridges `CallbackManager` → typed events. Useful for `RuntimeBackend.events()` on `LocalRuntime` without new code. |
| `xaytune/studio/jobs.py` (`JobManager`, `JobInfo`, `JobStatus`, log buffer, JSON persistence) | 401 | **Prior art; do NOT extend** | This is a second, informal control plane (submit/status/list/cancel/persisted metadata). Its API shape is what `ExperimentHandle` should supersede. Per refactor map §1, leave Studio alone until Phase 10. |
| `xaytune/plugins.py` | 68 | **Extend, don't replace** (ADR-008) | 4 entry-point groups, no version validation, failures are swallowed with `logger.warning` — ADR-008 requires fail-closed on unknown major versions, and §7 of the contract forbids swallowing. |
| `xaytune/pipeline.py` | 301 | **Keep independent** | Deterministic train→export→eval stages. Must not be merged with adaptive `Experiment`. |
| `xaytune/cli.py` | 762 | **Freeze; add subcommands only** | `train`/`eval`/`pipeline` are compatibility-committed (contract Rule 3). `submit`/`attach`/`watch` are additive (PR-029). |
| — | — | **New** | `core/`, `experiment/`, `compilation/`, `runtimes/`, `storage/`, `policy/`, `budget/`, `resilience/`, `agents/`, `search/`, `provenance/` do not exist yet. |

No existing module conflicts architecturally with the target; the conflicts are all *packaging and import-graph* conflicts (§3).

---

## 2. Target files — Phase 1 (PR-001 … PR-006)

Phase 0 exit criteria are unmet: all ten ADRs are `Status: Proposed`. Per `15-implementation-plan.md` §Phase 0,
no feature implementation should start until the six foundation ADRs (001–006) are accepted. The file list below
is what PR-001…PR-006 should create once that gate clears.

```
xaytune/core/__init__.py          # no heavy imports; pydantic + stdlib only
xaytune/core/ids.py               # ExperimentId/NodeId/RunId/AttemptId/… , prefixed sortable IDs (03 §13)
xaytune/core/errors.py            # XaytuneError, InvalidTransitionError, ConcurrentModificationError
xaytune/core/refs.py              # ArtifactRef, DatasetRef, ModelRef, Actor          -> PR-001
xaytune/core/domain/experiment.py # Experiment, ExperimentNode, Objective
xaytune/core/domain/run.py        # Run, RunAttempt, ExecutionOverride
xaytune/core/domain/action.py     # Action, Incident, Decision                        -> PR-002
xaytune/core/state/machines.py    # per-aggregate transition tables (ADR-002)
xaytune/core/state/transitions.py # typed transition objects (AttemptFailed, …)       -> PR-003
xaytune/core/events/event.py      # Event schema, sequence, aggregate_type
xaytune/core/protocols.py         # ExperimentRepository Protocol (no impl)
xaytune/storage/sqlite/schema.sql # from spec schemas/sqlite-schema.sql
xaytune/storage/sqlite/repository.py  # BEGIN IMMEDIATE, revision CAS                 -> PR-004/005
xaytune/storage/sqlite/outbox.py
xaytune/experiment/graph.py       # lineage, roots, descendants, cycle prevention     -> PR-006
```

Deliberately **not** in Phase 1: any edit to `trainer/`, `recipes/`, `eval/`, `studio/`, `cli.py`, or `pyproject.toml`
dependency lists. Phase 1 is purely additive (contract Rule 2 — one seam per PR).

---

## 3. Compatibility risk

### R1 — `xaytune/__init__.py` makes Invariant H / ADR-010 unachievable (blocking, highest priority)

`xaytune/__init__.py` eagerly imports `evaluate`, `pipeline`, `align`, `finetune`, `pretrain`, `JobManager`,
`lr_find` and then calls `discover_plugins()` at import time. The chain
`xaytune.recipes.base:6 → from torch.utils.data import DataLoader` and `xaytune.eval.evaluate:5 → import torch`
means `import xaytune.core` executes the package `__init__` and therefore **requires torch**, no matter how clean
`core/` itself is. `pyproject.toml` also lists `torch`, `transformers`, `peft`, `bitsandbytes`, `datasets` as
hard `dependencies`, not extras.

Verified locally: `python3 -c "import torch"` → `ModuleNotFoundError`, so the core-import test is meaningful here today.

Resolution options, in order of preference:

1. **Lazy `__getattr__` in `xaytune/__init__.py`** (PEP 562). Preserves `xaytune.finetune(...)` exactly (Rule 3),
   defers torch to first attribute access, and lets `import xaytune.core` succeed torch-free. Also moves
   `discover_plugins()` out of import time.
2. Move heavy deps to a `xaytune[training]` extra and keep the base install lightweight. Higher user-visible blast
   radius — needs a migration note and a CHANGELOG entry.

Option 1 is a prerequisite for PR-001's "no heavy imports" test and should be its own small PR before PR-001.

### R2 — the baseline is red; every CI gate fails on `main` (blocking, supersedes all Phase 1 work)

Verified by execution on 2026-09-20 in a fresh `.venv` (`uv pip install -e ".[dev]"`, Python 3.12):

| Gate | Result |
|---|---|
| `pytest -m "not slow"` | **27 failed**, 1252 passed, 4 skipped |
| `ruff check .` | **54 errors** (21 E741, 12 E501, 11 F401, 4 I001, 4 F841, 2 F821) |
| `ruff format --check .` | **36 files** would be reformatted |
| `mypy xaytune/ --ignore-missing-imports` | **cannot complete** — `python_version = "3.10"` in `pyproject.toml` vs numpy stubs requiring 3.12 syntax |

GitHub Actions run 27032496660 on `main` confirms this is not a local artifact: `lint`, `type-check`, and all three
`test` matrix jobs fail. `main` has been red since the last commit (2209c1f, 2026-06-05).

Triage of the 27 test failures:

| Class | Count | Examples |
|---|---:|---|
| Test-side rot — stale assertions | 3 | `test_packaging::test_all_exports` (expects `__all__` without `"pipeline"`); `test_example_configs::test_ten_examples_total` (hardcodes 11, there are 13); `test_all_examples_are_valid_yaml` (asserts a `model` key on `pipeline.yaml`, which is a pipeline config) |
| Test-side rot — mock drift | 10 | `test_qlora_preparation` patches `xaytune.models.peft.prepare_model_for_kbit_training`, which no longer exists there (4); fake tokenizers reject the `add_special_tokens` kwarg the code now passes (3); `test_loader` `KeyError: 'text'` (3) |
| Network-dependent | 3 | `test_model_merge_cli` downloads `model-a` from HF Hub → 404. Should be mocked or marked `slow` |
| Float precision | 1 | `test_checkpoint_portability` asserts `0.949999988 == 0.95`; needs `pytest.approx` |
| `evaluate()` robustness | 5 | `xaytune/eval/evaluate.py:67` masks `labels` assuming a tensor, two lines after the code explicitly handles non-tensor values. Test fixtures pass `labels=[1]` (a list) → `IndexError` |
| Expected-call-not-found | 2 | `test_trainer/test_distributed`, `test_logging/test_mlflow::test_log_config` |
| **Possible real behavior bugs — need investigation** | 3 | `test_integration::test_gradient_accumulation_reduces_optimizer_steps` (2 vs 4 optimizer steps); `test_scheduler::test_constant_ignores_warmup_steps` (0.0 vs 0.5); `test_alignment_edge_cases::test_both_logprobs_zero` |

**Confirmed production bug, caught by ruff and missed by the test suite:** `xaytune/pipeline.py:150-151` evaluates
`TrainerConfig()` and `LoraConfig()`, but their import sits at line 157 — *after* the use (F821). Because
`stage.trainer or TrainerConfig()` short-circuits, this only raises when a pipeline stage omits `trainer:` or `lora:`,
which is why no test caught it. `xaytune pipeline` is a compatibility-committed CLI surface (contract Rule 3).

Consequence for sequencing: contract §10 requires "existing tests pass" before any task is marked complete, and the
case for incremental refactor over a greenfield rewrite rests entirely on the test suite acting as a safety net.
Neither holds while the baseline is red. Greening the gates is a prerequisite to Phase 1, not a parallel cleanup.

### R2a — baseline greening: outcome (2026-09-20)

| Gate | Before | After |
|---|---|---|
| `pytest -m "not slow"` | 27 failed, 1252 passed | **1284 passed**, 4 skipped, 4 xfailed |
| `ruff check .` | 54 errors | **clean** |
| `ruff format --check .` | 36 files unformatted | **clean** |
| `mypy xaytune/` | could not complete | completes; **1 error**, which is finding F2 below |

Test-side repairs: stale `__all__` and example-config assertions replaced with non-brittle equivalents;
`pytest.approx` for float32 metrics; fake tokenizers taught the `add_special_tokens` kwarg; loader tests rewritten
against the current deferred-chat-template contract; `evaluate` fixtures given real tensors; `AutoConfig` mocked in
the merge-CLI tests (they were reaching the HF Hub and 404ing); `torch.cuda.is_available` mocked in the distributed
backend test, which could only ever pass on a GPU host.

Config: `python_version` 3.10 → 3.12 in `[tool.mypy]` (numpy's stubs use 3.12-only syntax, and CI already runs mypy
under 3.12 — the config was claiming a version it never checked at). **Tradeoff:** mypy no longer verifies 3.10
compatibility, while `requires-python` is `>=3.10` and CI still tests on 3.10. Worth revisiting.
`**/*.md` and the spec package excluded from ruff — ruff 0.16 formats Markdown code blocks, which rewrote README and
docs snippets with PEP8 blank lines the project never opted into.

### Findings — four real defects, none fixed pending review

**F1 — `xaytune/pipeline.py:150` `UnboundLocalError` (fixed, with regression test).** `TrainerConfig()`/`LoraConfig()`
were evaluated above their import. `stage.trainer or TrainerConfig()` short-circuits, so it only fired when a stage
omitted `trainer:` or `lora:`. `xaytune/pipeline.py` had **no test file at all** — `tests/test_pipeline.py` is new.
This one *was* fixed because it is a plain ordering mistake with no semantic question attached.

**F2 — DeepSpeed + resume crashes (`xaytune/trainer/loop.py:64,122`).** `train()` sets `optimizer = None` on the
DeepSpeed path, then calls `optimizer.load_state_dict(...)` unguarded when `resume_checkpoint_dir` is set. The
adjacent scaler and scheduler branches both check for `None`; the optimizer branch does not. This is the one
remaining mypy error. Fix: guard the branch, or restore optimizer state through the DeepSpeed engine.

**F3 — CPU-only distributed crashes (`xaytune/trainer/distributed.py:55`).** `torch.cuda.set_device(local_rank)` is
called unconditionally, so a gloo (CPU) process group raises `AttributeError: module 'torch._C' has no attribute
'_cuda_setDevice'`. Covered by a strict xfail in `tests/test_trainer/test_distributed.py`. Fix: guard with
`torch.cuda.is_available()`.

**F4 — ORPO NaN, i.e. BUG-011, still open.** `orpo_loss` computes `log1p(-logps.exp())`; at `logps == 0` that is
`log(0) = -inf`, and `(-inf) - (-inf)` is `NaN`. `logps == 0` is reachable — a fully-masked sequence sums to exactly
zero. `gap-analysis/bugs.md:19` lists BUG-011 as **Critical / OPEN**; the edge-case test was written but the fix never
landed. Fix: clamp `exp()` below 1 by an epsilon, as TRL does. Strict xfail records it.

### Two open semantic decisions (strict xfails, need a product call)

**D1 — does `constant` honour `warmup_steps`?** `scheduler.py:48-53` deliberately auto-upgrades `constant` to warmup
behaviour when `warmup_steps > 0`; `test_constant_ignores_warmup_steps` asserts the opposite. Deciding for the
implementation makes the separate `constant_with_warmup` option redundant; deciding for the test means a requested
warmup is silently dropped.

**D2 — what unit is `global_step`?** The loop counts optimizer steps; `test_gradient_accumulation_reduces_optimizer_steps`
expects micro-steps. The rest of the loop agrees with the implementation — `loop.py` derives `total_steps` as
`num_batches // gradient_accumulation`, so LR scheduling and `max_steps` are already in optimizer steps. The test is
probably wrong, but `global_step` is user-visible in checkpoints and logs, so the definition is a product call.

### R2b — scope and lineage revisions (ADR-011)

Two design changes came out of reviewing the spec against what it would take to serve
frontier pretraining. The conclusion was that it should not try to, and sharpening that
made the rest cleaner.

**Positioning.** The README now states the non-goals near the top and defines the
boundary as experiment topology rather than a GPU count: many candidates with frequent
evaluation and branching, as against one continuous months-long run. The architecture can
still submit large distributed jobs; what it does not own is second-scale in-band fault
tolerance. Xaytune owns *semantic* recovery, the runtime owns distributed-systems fault
tolerance.

**Lineage.** ADR-003's binary split has no category for a scientifically meaningful
change applied to a run that is still going — a reactive LR drop, a curriculum
transition, a reward ramp. ADR-011 adds `TrainingIntervention` as a third level,
produced by the existing `Action` path so there is still one governance route, and adds
the comparability rule that decides between an intervention and a new node. Identity
splits into `CandidateFingerprint` (declared, including pre-registered schedules) and
`RunRealizationFingerprint` (what actually happened).

ADR-011 was **accepted on 2026-09-20**, ahead of ADR-001..010, because PR-005 cannot
define its event schema without it. The rollback question it originally carried is
decided rather than deferred: an intervention *decision* belongs to the run, each
*application* is recorded separately, a restore never erases prior applications, and
re-application is governed by an explicit, immutable `InterventionReplayPolicy` rather
than by position alone — so a one-off emergency adjustment cannot silently become a permanent
schedule after a worker dies. The realization fingerprint hashes applications, which is
what distinguishes a run that double-applied after a rollback from one that did not.

Triggers are a tagged union and every intervention records one, so `REEVALUATE_TRIGGER`
has something concrete to re-evaluate. It is restricted to non-monotone conditions:
pairing it with a step or token trigger would re-match on every restore and silently
double-apply a change already present in the restored optimizer state.

### R3 — two competing plan documents in the repo

`gap-analysis/` (BUG-/GAP-/FEAT- IDs) and `implementation-plan/` (TASK-001…031, EPIC-0…11, dated 2026-06-03) predate
this spec package and are organised around fixing v0.6, not around the control plane. They are not wrong, but an agent
told to "follow the implementation plan" will pick the wrong one. Add a pointer in `implementation-plan/README.md`
stating which plan governs which workstream.

### R4 — `TrainerConfig` straddles the scientific/execution boundary

`learning_rate`, `num_epochs`, `seed`, `scheduler`, `warmup_*`, `weight_decay` are scientific (change → new node).
`batch_size`, `gradient_accumulation`, `activation_checkpointing`, `async_checkpoint`, `strategy`, `mixed_precision`
are executional (change → `ExecutionOverride`, same node). Splitting one Pydantic model that
`recipes/`, `cli.py`, `pipeline.py`, `studio/codegen.py` and ~20 example YAMLs all construct is a wide change.
Mitigation: do **not** split it in Phase 1. Build `TrainingSpec` as a new type in PR-007 and give `NativeCompiler`
a one-way `TrainingSpec → TrainConfig` mapping. `TrainConfig` stays the legacy surface until parity is proven.

### R5 — checkpoint format is implicit and torch-coupled

`trainer/checkpointing.py` writes `model.pt`/`optimizer.pt`/`scheduler.pt`/`scaler.pt` + `metadata.json` with
`torch.save`. There is no version field and no compatibility key. ADR-009 splits codec/store/manager and ADR-006 adds
`CheckpointCompatibilityKey`. Phase 4 (PR-018) must read the existing layout unchanged — add a version field going
forward, never rewrite old checkpoints in place. `tests/test_checkpoint_portability.py` is the regression anchor.

### R6 — `discover_plugins()` swallows load failures

`plugins.py` catches bare `Exception` and logs a warning. ADR-008 requires fail-closed on unknown major plugin API
versions, and contract §7 forbids swallowing. Changing this is user-visible (a previously-ignored broken plugin starts
raising) → needs a CHANGELOG entry and a deprecation window.

---

## 4. Tests to add

Per contract §5, keyed to the PR that introduces each.

| PR | Test | Assertion |
|---|---|---|
| R1 fix | `tests/test_core_import_isolation.py` | Subprocess with `torch`/`transformers`/`peft`/`ray`/`trl` blocked via a `sys.meta_path` finder: `import xaytune.core` succeeds. Also assert `"torch" not in sys.modules` after it. This is the executable form of Invariant H. |
| R1 fix | extend `tests/test_top_level_api.py` | `xaytune.finetune`, `.pretrain`, `.align`, `.evaluate`, `.pipeline`, `.lr_find`, `.JobManager` all still resolve and are callable (Rule 3 regression guard). |
| PR-001 | `tests/test_core/test_ids.py` | Prefix correctness, lexicographic sortability matches creation order, round-trip str↔ID, rejection of malformed IDs. |
| PR-002 | `tests/test_core/test_domain.py` | Pydantic round-trip (`model_dump_json` → `model_validate_json`) for every aggregate; immutability of `TrainingSpecSnapshot` (Rule 5). |
| PR-003 | `tests/test_core/test_state_machines.py` | Table-driven: every legal transition per aggregate succeeds, every illegal one raises `InvalidTransitionError`; terminal states accept no transitions; direct `obj.status = ...` is blocked (Rule 7). |
| PR-004 | `tests/test_storage/test_repository.py` | Revision CAS: a stale `expected_revision` raises `ConcurrentModificationError` and leaves the row untouched. |
| PR-005 | `tests/test_storage/test_atomicity.py` | **Crash test** — inject an exception between the aggregate `UPDATE` and the `events` `INSERT`; reopen the DB and assert state, revision, event, and outbox row all rolled back together (Rule 8 / ADR-005). Then assert the happy path commits all four. |
| PR-005 | `tests/test_storage/test_outbox.py` | Idempotent redelivery: replaying an undelivered outbox record twice produces one effect at the sink. |
| PR-006 | `tests/test_experiment/test_graph.py` | Parent/child/roots/descendants; cycle creation raises; an operational recovery adds a `RunAttempt` and creates **no** node, while an LR mutation creates one (Invariant B / ADR-003 — the single most valuable test in Phase 1). |
| PR-006 | `tests/test_architecture.py` | AST-level import-direction guard: no module under `xaytune/core/` imports `torch|transformers|peft|trl|ray|torchft|kubernetes|mlflow|wandb`; nothing under `xaytune/core/` imports from `xaytune/trainer|recipes|studio|eval`. Cheap, and it holds for every later PR. |

Existing regression anchors that must stay green throughout: `tests/test_packaging.py`, `tests/test_top_level_api.py`,
`tests/test_cli.py`, `tests/test_integration.py`, `tests/test_checkpoint_resume_integration.py`, `tests/test_plugins.py`.

---

## 5. Architecture invariant checks

Mechanical checks to run per PR, mapped to the hard rules in the review prompt.

| Invariant / rule | Check | Enforced by |
|---|---|---|
| H — core has no ML deps | `import xaytune.core` in a torch-free subprocess | `test_core_import_isolation.py` (automated) |
| §6 — dependency direction | AST scan of `xaytune/core/**` imports | `test_architecture.py` (automated) |
| Rule 4 — compile ≠ execute | No `subprocess`, `requests`, `httpx`, `ray`, `kubernetes`, or `.submit(` inside `xaytune/trainers/**` | `test_architecture.py` (automated) |
| Rule 7 — no direct state assignment | `ast-grep` for attribute assignment to `.status`/`.state` outside `core/state/` and repository internals | `test_architecture.py` (automated) |
| Rule 8 — atomic state+event | Every `INSERT INTO events` is inside the same `BEGIN IMMEDIATE` as its aggregate `UPDATE` | crash test + review |
| Rule 5 — immutable scientific lineage | Domain snapshot models declare `model_config = ConfigDict(frozen=True)` | `test_domain.py` (automated) |
| Rule 9 — LLM proposes only | No `subprocess`/`eval`/`exec` anywhere under `xaytune/agents/**` | `test_architecture.py` (automated) |
| Rule 2 — no repo rewrite | `git diff --stat` touches ≤3 subsystems; Phase 1 adds files only | review |
| Rule 3 — public compatibility | `tests/test_top_level_api.py` + `tests/test_cli.py` unchanged and green | automated |
| Invariant B — lineage split | Recovery → new `RunAttempt`; scientific mutation → new node | `test_graph.py` (automated) |

Gate command for every PR (from contract §4, matching `.github/workflows/ci.yml`):

```bash
ruff check . && ruff format --check . && mypy xaytune/ --ignore-missing-imports && pytest tests/ -q -m "not slow"
```

---

## 6. Recommended first three PRs

1. **PR-000 (new, not in the spec plan)** — lazy `__getattr__` in `xaytune/__init__.py`, move `discover_plugins()`
   off import time, add `test_core_import_isolation.py` and `test_architecture.py`. Unblocks ADR-010 and costs
   ~40 lines. Everything else in Phase 1 is untestable against Invariant H without it.
2. **PR-001** — `xaytune/core/ids.py`, `errors.py`, `refs.py`.
3. **PR-002** — domain aggregates, frozen, no controller.

Blocking question before any of this: ADR-001…ADR-006 are all `Status: Proposed`, and Phase 0's stated exit criterion
is that they be accepted. Confirm that gate, or explicitly waive it, before PR-000 lands.
