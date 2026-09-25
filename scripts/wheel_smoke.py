"""Does the built wheel carry a usable control plane? Checked without running ML.

Run inside a fresh virtualenv that has the wheel -- and nothing from this
checkout -- installed::

    python -I scripts/wheel_smoke.py

The test suite runs against the source tree, so it cannot notice a package
that builds but ships without its migrations, or whose ``import xaytune``
drags in torch. This runs the shipped boundaries a user reaches first:

    import xaytune              without torch, transformers or trl
    control-plane store         created, and migrated through every migration
    EmbeddedControllerHost      opens a store and closes
    CandidateSpec, ExperimentSpec
                                constructed, validated, round-tripped
    NativeCompiler.compile()    a TrainingExecutionSpec, deterministically
    ResolvedExecutionPlan       serialized and read back as the same request
    examples/control_plane/     import from the installed package; the one
                                that needs no model runs to completion

Nothing is submitted to a runtime. The first public ``submit() -> wait()``
smoke arrives with a packaged execution path that is product functionality,
not scaffolding written for this check.

``-I`` keeps the checkout off ``sys.path``; the script also refuses to run
against a ``xaytune`` imported from it. The checkout is read for two things
only: the list of migration files the wheel must contain, and the examples,
which a user runs from a clone against whatever they installed.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import io
import runpy
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

HEAVY = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "datasets",
    "accelerate",
    "bitsandbytes",
    "deepspeed",
    "lm_eval",
)
"""Training stacks the control plane must not import as a side effect (ADR-010)."""


class SmokeError(AssertionError):
    pass


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeError(message)


def _step(message: str) -> None:
    print(f"ok  {message}", flush=True)


def _no_training_stack(after: str) -> None:
    loaded = sorted(name for name in HEAVY if name in sys.modules)
    _check(not loaded, f"{after} imported {', '.join(loaded)}")


def import_the_installed_package() -> None:
    import xaytune

    location = Path(xaytune.__file__).resolve()
    _check(
        REPO not in location.parents,
        f"xaytune was imported from the checkout ({location}), not from the installed "
        f"wheel; run with `python -I` from a fresh virtualenv",
    )
    _no_training_stack("import xaytune")
    _step(f"import xaytune {xaytune.__version__} from {location.parent}")

    import xaytune.compilation.native
    import xaytune.compilation.trl
    import xaytune.core
    import xaytune.evaluation
    import xaytune.experiment
    import xaytune.storage  # noqa: F401

    _no_training_stack("importing the control-plane packages")
    _step("control-plane packages import without a training stack")

    from importlib.metadata import distribution

    installed = distribution("xaytune")
    expression = installed.metadata.get("License-Expression")
    _check(expression == "Apache-2.0", f"the wheel declares license {expression!r}")
    licenses = [f for f in installed.files or () if f.name == "LICENSE"]
    _check(bool(licenses), "the wheel installs no LICENSE file")
    _step(f"the wheel declares {expression} and installs {licenses[0]}")


def migrate_a_new_store(directory: Path) -> None:
    from xaytune.storage import applied_versions, available_migrations, connect, migrate

    expected = sorted(p.name for p in (REPO / "xaytune" / "storage" / "migrations").glob("*.sql"))
    shipped = [m.path.name for m in available_migrations()]
    _check(
        shipped == expected,
        f"the wheel ships migrations {shipped}, the source tree has {expected}",
    )

    connection = connect(directory / "state.db")
    try:
        applied = migrate(connection)
        versions = tuple(m.version for m in available_migrations())
        _check(applied == versions, f"a new store applied {applied}, expected {versions}")
        _check(migrate(connection) == (), "migrating an up-to-date store applied something")
        _check(applied_versions(connection) == versions, "the ledger disagrees with the files")
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    finally:
        connection.close()
    for table in ("experiments", "runtime_operations", "actions", "evaluation_results"):
        _check(table in tables, f"the migrated store has no {table} table")
    _step(f"new store migrated through {len(versions)} migrations: {', '.join(shipped)}")


def open_and_close_a_host(directory: Path) -> None:
    from xaytune.experiment import EmbeddedControllerHost

    state = directory / "host" / "state.db"
    host = EmbeddedControllerHost(state)
    asyncio.run(host.close())
    _check(state.exists(), "EmbeddedControllerHost did not create its state database")
    _no_training_stack("opening an EmbeddedControllerHost")
    _step("EmbeddedControllerHost opens a new store and closes")


def compile_a_candidate(directory: Path) -> None:
    from xaytune.compilation import CompilationContext
    from xaytune.compilation.native import NativeCompiler
    from xaytune.core import (
        CandidateSpec,
        DatasetRef,
        DataSpec,
        LRScheduleSpec,
        ModelRef,
        ModelSpec,
        Objective,
        ObjectiveMetric,
        OptimizationSpec,
        OptimizerSpec,
        PrecisionSpec,
        ResolvedExecutionPlan,
        RuntimeOperationTarget,
        TrainingExecutionSpec,
        TrainingKind,
        TrainingSpec,
    )
    from xaytune.core.execution import PythonModuleEntrypoint
    from xaytune.experiment import CompilerSpec, ExperimentSpec, RuntimeSpec

    # Compilation inspects nothing, so these paths need not exist: the plan
    # names them for a worker, which is where they would be read.
    candidate = CandidateSpec(
        model=ModelSpec(model=ModelRef(uri=str(directory / "models" / "base"))),
        data=DataSpec(
            dataset=DatasetRef(uri=str(directory / "data" / "train.jsonl")),
            format="alpaca",
            max_seq_length=512,
            packing=False,
        ),
        training=TrainingSpec(
            kind=TrainingKind.SFT,
            optimization=OptimizationSpec(
                optimizer=OptimizerSpec(name="adamw", weight_decay=0.0),
                lr_schedule=LRScheduleSpec(name="constant"),
                learning_rate=2e-5,
                micro_batch_size=4,
                gradient_accumulation=1,
                epochs=1,
                max_grad_norm=1.0,
            ),
            precision=PrecisionSpec(dtype="fp32"),
        ),
    )
    spec = ExperimentSpec(
        name="wheel-smoke",
        objective=Objective(primary=ObjectiveMetric(name="loss", direction="minimize")),
        candidate=candidate,
        seed=7,
        compiler=CompilerSpec(name="native"),
        runtime=RuntimeSpec(kind="local", config={"root": str(directory / "runtime")}),
        artifact_root=str(directory / "artifacts"),
    )
    restored_spec = ExperimentSpec.model_validate_json(spec.model_dump_json())
    _check(restored_spec == spec, "an ExperimentSpec did not survive a JSON round trip")
    _check(
        restored_spec.candidate.candidate_fingerprint() == candidate.candidate_fingerprint(),
        "a round-tripped candidate has a different fingerprint",
    )
    _step(f"ExperimentSpec round-trips; candidate {candidate.candidate_fingerprint()}")

    compiler = NativeCompiler()
    support = compiler.supports(candidate)
    _check(bool(support), f"NativeCompiler refused the candidate: {support.reasons}")
    context = CompilationContext(
        run_id="run_smoke", seed=spec.seed, output_uri=str(directory / "artifacts" / "run_smoke")
    )
    compiled = compiler.compile(candidate, context)
    _check(isinstance(compiled, TrainingExecutionSpec), "compile() returned no training spec")
    _check(
        compiled.candidate_fingerprint == candidate.candidate_fingerprint(),
        "the compiled spec names a different candidate",
    )
    _check(compiler.compile(candidate, context) == compiled, "compilation is not deterministic")
    entrypoint = compiled.entrypoint
    _check(isinstance(entrypoint, PythonModuleEntrypoint), f"unexpected entrypoint {entrypoint}")
    assert isinstance(entrypoint, PythonModuleEntrypoint)
    # Located, not imported: the worker loads the training stack, and it is the
    # worker process's to load. The plan must name a module the wheel ships.
    _check(
        importlib.util.find_spec(entrypoint.module) is not None,
        f"the plan runs {entrypoint.module}, which the wheel does not contain",
    )
    _no_training_stack("compiling a candidate")
    _step(f"NativeCompiler compiled it for {entrypoint.module}, which the wheel ships")

    plan = ResolvedExecutionPlan(
        spec=compiled,
        runtime=spec.runtime.kind,
        target=RuntimeOperationTarget(kind="training-attempt", id="attempt_smoke"),
    )
    restored = ResolvedExecutionPlan.model_validate_json(plan.model_dump_json())
    _check(restored == plan, "a ResolvedExecutionPlan did not survive a JSON round trip")
    _check(isinstance(restored.spec, TrainingExecutionSpec), "the plan read back as another kind")
    _check(
        restored.request_digest("submit") == plan.request_digest("submit"),
        "a round-tripped plan is a different request",
    )
    _step(f"ResolvedExecutionPlan round-trips as the same request {plan.request_digest('submit')}")


def run_the_examples() -> None:
    examples = REPO / "examples" / "control_plane"
    scripts = sorted(examples.glob("[0-9][0-9]_*.py"))
    _check(bool(scripts), f"no examples found in {examples}")
    # The runnable examples share one module beside them, as a script run
    # from that directory would find it.
    sys.path.insert(0, str(examples))
    try:
        for script in scripts:
            # Importing defines everything and runs nothing: each example
            # guards its work behind __main__. The first needs no model, so
            # it runs to completion.
            run_name = "__main__" if script.name.startswith("01_") else "__smoke__"
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    runpy.run_path(str(script), run_name=run_name)
            except Exception as exc:  # any failure of a public example is the finding
                raise SmokeError(f"{script.name} failed against the wheel: {exc!r}") from exc
            _no_training_stack(f"{script.name}")
            _step(f"{script.name} {'runs' if run_name == '__main__' else 'imports'}")
    finally:
        sys.path.remove(str(examples))


def main() -> int:
    try:
        import_the_installed_package()
        with tempfile.TemporaryDirectory(prefix="xaytune-smoke-") as tmp:
            directory = Path(tmp).resolve()
            migrate_a_new_store(directory)
            open_and_close_a_host(directory)
            compile_a_candidate(directory)
        run_the_examples()
        _no_training_stack("the whole smoke")
    except SmokeError as failure:
        print(f"FAIL {failure}", file=sys.stderr)
        return 1
    print("wheel smoke passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
