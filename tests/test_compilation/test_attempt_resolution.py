"""Attempt lineage changes execution while preserving restart request identity."""

from __future__ import annotations

import asyncio

import pytest

from tests.test_experiment.test_host_behaviour import _spec
from tests.training_fixtures import sft_candidate
from xaytune.compilation import CompilationContext
from xaytune.compilation.attempt_resolution import resolve_training_attempt
from xaytune.compilation.native import NativeCompiler
from xaytune.compilation.trl import TRLCompiler
from xaytune.core.domain.run import ExecutionOverride, RunAttempt
from xaytune.core.ids import CheckpointId, RunAttemptId, RunId
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import Actor, CheckpointRef
from xaytune.experiment import CompilerSpec, EmbeddedControllerHost


def _candidate(tmp_path):
    candidate = sft_candidate(tmp_path / "model", tmp_path / "data.jsonl", "text")
    optimization = candidate.training.optimization.model_copy(
        update={"micro_batch_size": 4, "gradient_accumulation": 8}
    )
    training = candidate.training.model_copy(update={"optimization": optimization})
    return candidate.model_copy(update={"training": training})


def _attempt(run_id):
    checkpoint = CheckpointRef(
        id=CheckpointId.generate(),
        uri="file:///checkpoint",
        digest="sha256:" + "a" * 64,
        compatibility_key="test-compatible",
    )
    return RunAttempt(
        id=RunAttemptId.generate(),
        run_id=run_id,
        attempt_number=2,
        checkpoint_ref=checkpoint,
        execution_overrides=(
            ExecutionOverride(
                id="oom-micro",
                kind="micro_batch_resize",
                reason="CUDA OOM",
                values=FrozenDict({"from": 4, "to": 2}),
                preserves=("effective_batch_size",),
            ),
            ExecutionOverride(
                id="oom-accumulation",
                kind="gradient_accumulation_adjustment",
                reason="preserve effective batch",
                values=FrozenDict({"from": 8, "to": 16}),
                preserves=("effective_batch_size",),
            ),
            ExecutionOverride(
                id="oom-checkpoint",
                kind="checkpoint_restore",
                reason="resume from valid checkpoint",
                values=FrozenDict({"checkpoint_id": str(checkpoint.id)}),
            ),
        ),
    )


def _cumulative_attempt(run_id):
    first = _attempt(run_id)
    assert first.checkpoint_ref is not None
    checkpoint = first.checkpoint_ref.model_copy(update={"id": CheckpointId.generate()})
    return RunAttempt(
        id=RunAttemptId.generate(),
        run_id=run_id,
        attempt_number=3,
        checkpoint_ref=checkpoint,
        execution_overrides=(
            *first.execution_overrides[:2],
            ExecutionOverride(
                id="oom-micro-again",
                kind="micro_batch_resize",
                reason="second CUDA OOM",
                values=FrozenDict({"from": 2, "to": 1}),
                preserves=("effective_batch_size",),
            ),
            ExecutionOverride(
                id="oom-accumulation-again",
                kind="gradient_accumulation_adjustment",
                reason="preserve effective batch again",
                values=FrozenDict({"from": 16, "to": 32}),
                preserves=("effective_batch_size",),
            ),
            ExecutionOverride(
                id="oom-checkpoint-again",
                kind="checkpoint_restore",
                reason="resume from the second valid checkpoint",
                values=FrozenDict({"checkpoint_id": str(checkpoint.id)}),
            ),
        ),
    )


def _with_overrides(attempt: RunAttempt, overrides: tuple[ExecutionOverride, ...]) -> RunAttempt:
    return RunAttempt.model_validate(
        {**attempt.model_dump(mode="json"), "execution_overrides": overrides}
    )


@pytest.mark.parametrize("compiler", [NativeCompiler(), TRLCompiler()])
def test_both_trainers_resolve_same_durable_resize_and_checkpoint(compiler, tmp_path):
    candidate = _candidate(tmp_path)
    compiled = compiler.compile(
        candidate,
        CompilationContext(run_id="run-contract", seed=7, output_uri=str(tmp_path / "artifacts")),
    )
    attempt = _attempt(RunId.generate())
    resolved = resolve_training_attempt(compiled, attempt, "local")
    assert resolved.spec.config["optimization"]["micro_batch_size"] == 2
    assert resolved.spec.config["optimization"]["gradient_accumulation"] == 16
    assert resolved.spec.candidate_fingerprint == compiled.candidate_fingerprint
    assert resolved.runtime_options["checkpoint_restore"]["id"] == str(attempt.checkpoint_ref.id)
    assert resolved.request_digest("submit") == resolve_training_attempt(
        compiled, attempt, "local"
    ).request_digest("submit")
    assert resolved.request_digest("submit") != resolve_training_attempt(
        compiled,
        RunAttempt(id=RunAttemptId.generate(), run_id=attempt.run_id, attempt_number=1),
        "local",
    ).request_digest("submit")


@pytest.mark.parametrize("compiler", [NativeCompiler(), TRLCompiler()])
def test_both_trainers_resolve_cumulative_second_oom_lineage(compiler, tmp_path):
    compiled = compiler.compile(
        _candidate(tmp_path),
        CompilationContext(run_id="run-contract", seed=7, output_uri=str(tmp_path / "artifacts")),
    )
    attempt = _cumulative_attempt(RunId.generate())
    resolved = resolve_training_attempt(compiled, attempt, "local")
    assert resolved.spec.config["optimization"]["micro_batch_size"] == 1
    assert resolved.spec.config["optimization"]["gradient_accumulation"] == 32
    assert resolved.runtime_options["checkpoint_restore"]["id"] == str(attempt.checkpoint_ref.id)
    assert resolved.request_digest("submit") == resolve_training_attempt(
        compiled, attempt, "local"
    ).request_digest("submit")


def test_override_resolution_fails_closed_on_inconsistent_lineage(tmp_path):
    compiled = NativeCompiler().compile(
        _candidate(tmp_path),
        CompilationContext(run_id="run-contract", seed=7, output_uri=str(tmp_path / "artifacts")),
    )
    attempt = _attempt(RunId.generate())
    with pytest.raises(ValueError, match="preserve effective batch"):
        resolve_training_attempt(
            compiled,
            _with_overrides(
                attempt,
                (
                    attempt.execution_overrides[0],
                    attempt.execution_overrides[1].model_copy(
                        update={"values": FrozenDict({"from": 8, "to": 8})}
                    ),
                    attempt.execution_overrides[2],
                ),
            ),
            "local",
        )
    with pytest.raises(ValueError, match="checkpoint reference and restore override"):
        resolve_training_attempt(
            compiled,
            _with_overrides(attempt, attempt.execution_overrides[:2]),
            "local",
        )


@pytest.mark.parametrize(
    "index,from_value,to_value,error",
    [
        (2, 3, 1, "micro_batch_size override provenance"),
        (3, 8, 32, "gradient_accumulation override provenance"),
        (3, 16, 31, "preserve effective batch"),
    ],
)
def test_second_resize_refuses_a_broken_intermediate_chain(
    tmp_path, index, from_value, to_value, error
):
    compiled = NativeCompiler().compile(
        _candidate(tmp_path),
        CompilationContext(run_id="run-contract", seed=7, output_uri=str(tmp_path / "artifacts")),
    )
    attempt = _cumulative_attempt(RunId.generate())
    overrides = list(attempt.execution_overrides)
    overrides[index] = overrides[index].model_copy(
        update={"values": FrozenDict({"from": from_value, "to": to_value})}
    )
    with pytest.raises(ValueError, match=error):
        resolve_training_attempt(compiled, _with_overrides(attempt, tuple(overrides)), "local")


def test_each_resize_pair_must_preserve_batch_even_if_final_product_matches(tmp_path):
    compiled = NativeCompiler().compile(
        _candidate(tmp_path),
        CompilationContext(run_id="run-contract", seed=7, output_uri=str(tmp_path / "artifacts")),
    )
    attempt = _cumulative_attempt(RunId.generate())
    overrides = list(attempt.execution_overrides)
    overrides[1] = overrides[1].model_copy(update={"values": FrozenDict({"from": 8, "to": 8})})
    overrides[3] = overrides[3].model_copy(update={"values": FrozenDict({"from": 8, "to": 32})})
    # The first pair changes 4×8 to 2×8; the second would change 2×8 back
    # to 1×32. A final-only product check would incorrectly accept this.
    with pytest.raises(ValueError, match="preserve effective batch"):
        resolve_training_attempt(compiled, _with_overrides(attempt, tuple(overrides)), "local")


@pytest.mark.parametrize("compiler_name", ["native", "trl"])
@pytest.mark.parametrize("attempt_factory", [_attempt, _cumulative_attempt])
def test_host_rebuilds_resized_submission_after_restart(tmp_path, compiler_name, attempt_factory):
    state = tmp_path / "state.db"
    candidate = _candidate(tmp_path)

    async def first():
        host = EmbeddedControllerHost(state)
        try:
            spec = _spec(tmp_path).model_copy(
                update={"candidate": candidate, "compiler": CompilerSpec(name=compiler_name)}
            )
            experiment = host._record_experiment(
                spec, host._compiler(spec.compiler.name), host._runtime(spec.runtime), None
            )
            run = host._record_run(host._record_node(experiment, spec), spec.seed)
            attempt = attempt_factory(run.id)
            plan = host._plan(experiment, run, attempt, host._compiler(spec.compiler.name))
            _, operation = host.repository.create_attempt_with_submit_intent(
                attempt,
                request_digest=plan.request_digest("submit"),
                actor=Actor(type="system", id="test"),
            )
            return experiment.id, run.id, attempt.id, operation.id, plan
        finally:
            await host.close()

    experiment_id, run_id, attempt_id, operation_id, original = asyncio.run(first())

    async def restart():
        host = EmbeddedControllerHost(state)
        try:
            experiment = host.repository.aggregates.load_experiment(str(experiment_id))
            operation = host.repository.operations.get(str(operation_id))
            assert operation is not None
            rebuilt = host._rebuild_plan(
                "training-attempt", experiment, run_id, attempt_id, operation
            )
            assert rebuilt == original
            assert rebuilt.request_digest("submit") == operation.request_digest
        finally:
            await host.close()

    asyncio.run(restart())
