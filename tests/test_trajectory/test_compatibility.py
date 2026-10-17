"""Recording is an opt-in extension that moves nothing already recorded (ADR-021 T9; criterion 10).

``PINNED_H03_REQUEST`` was computed on ``main`` at ``c07c4af``, before H05,
from the H03 fixture spec at ``/runs/r1``. With recording off, H05 must
compile that spec byte for byte.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from tests.test_harness.fixtures import compile_spec, context, harness
from tests.test_harness.test_compiler import PINNED_EXECUTION_V1
from tests.test_trajectory.fixtures import recorded_spec, recording, run
from xaytune.core.agent_execution import (
    HarnessRunResult,
    agent_execution_fingerprint,
    agent_run_request_digest,
)
from xaytune.core.execution import ArtifactOutput
from xaytune.core.ids import ArtifactId
from xaytune.core.immutable import FrozenDict, thaw
from xaytune.core.refs import ArtifactRef
from xaytune.core.trajectory import TRAJECTORY_SCHEMA, WriterIdentity
from xaytune.harness import (
    RUNNER,
    HarnessRunRefusedError,
    ReferenceHarnessCompiler,
    ScriptedHarnessRunner,
)

PINNED_H03_REQUEST = "sha256:ed3cf478080c2a486341001d82b20a7cd37f3716e31e4c6ccebaef73a0d94118"


def test_recording_off_compiles_the_h03_spec_byte_for_byte() -> None:
    spec = compile_spec("/runs/r1")
    assert agent_execution_fingerprint(spec) == PINNED_EXECUTION_V1
    assert agent_run_request_digest(spec) == PINNED_H03_REQUEST
    explicit = ReferenceHarnessCompiler().compile(
        harness(), context(harness(), "/runs/r1", trajectory=None)
    )
    assert explicit == spec
    assert [output.kind for output in spec.outputs] == ["xaytune.agent-run-result/v1alpha1"]
    assert set(spec.config) == {"compiler_api", "token_counter"}


def test_recording_on_is_a_different_request_and_never_a_different_harness(
    tmp_path: Path,
) -> None:
    off = compile_spec(tmp_path)
    on = recorded_spec(tmp_path)
    assert on.harness_fingerprint == off.harness_fingerprint
    assert agent_execution_fingerprint(on) != agent_execution_fingerprint(off)
    assert agent_run_request_digest(on) != agent_run_request_digest(off)
    assert [o.name for o in on.outputs] == ["result", "trajectory"]
    assert on.outputs[1].kind == TRAJECTORY_SCHEMA


def test_the_runner_and_result_contracts_are_unchanged() -> None:
    assert (RUNNER.name, RUNNER.version) == ("xaytune.scripted-harness-runner", "1")
    assert set(HarnessRunResult.model_fields) == {
        "schema_version", "request_digest", "execution_fingerprint", "harness_fingerprint",
        "compiler", "runner", "model_digest", "task_id", "seed", "replicate", "termination",
        "completed", "answer", "error", "turns", "tool_calls", "refused_tool_calls",
        "failed_tool_calls", "context_truncations", "prompt_tokens", "output_tokens",
        "total_tokens", "token_counter",
    }  # fmt: skip


def test_a_result_published_before_h05_still_replays(tmp_path: Path) -> None:
    spec = compile_spec(tmp_path)
    first = run(spec)
    assert not (tmp_path / "trajectory").exists(), "recording off writes no trajectory"
    assert run(spec) == first


def test_the_agent_trajectory_kind_changes_no_existing_serialization() -> None:
    reference = ArtifactRef(
        id=ArtifactId.generate(), kind="logs", uri="file:///runs/logs.txt", digest=None
    )
    assert ArtifactRef.model_validate_json(reference.model_dump_json()) == reference
    trajectory = ArtifactRef(
        id=ArtifactId.generate(),
        kind="agent_trajectory",
        uri="file:///runs/trajectory/attempts/x/manifest.json",
        digest="sha256:" + "0" * 64,
    )
    assert json.loads(trajectory.model_dump_json())["kind"] == "agent_trajectory"


def test_no_migration_or_table_is_added() -> None:
    """Trajectories live in files, not the database: no migration creates a table for them.

    Checked by what the migrations create, not by which one is last (H05 shipped
    on 019), so a later migration for something else does not break it.
    """
    migrations = Path(__file__).parents[2] / "xaytune" / "storage" / "migrations"
    tables = {
        name.lower()
        for path in migrations.glob("*.sql")
        for name in re.findall(r"CREATE TABLE\s+(\w+)", path.read_text(), re.IGNORECASE)
    }
    assert not {table for table in tables if "trajector" in table}


# ---- the refused shapes (T9) -----------------------------------------------------------------


def _refusals(spec: Any) -> tuple[str, ...]:
    return ScriptedHarnessRunner().supports(spec).reasons


def _rebuild(spec: Any, **changes: Any) -> Any:
    return spec.model_copy(update=changes)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        pytest.param(
            lambda s: {"outputs": (s.outputs[0], s.outputs[1], s.outputs[1])},
            "at most one trajectory",
            id="two-trajectory-outputs",
        ),
        pytest.param(
            lambda s: {
                "config": FrozenDict({k: v for k, v in s.config.items() if k != "trajectory"})
            },
            "needs config['trajectory']",
            id="output-without-block",
        ),
        pytest.param(
            lambda s: {"outputs": (s.outputs[0],)},
            "needs a trajectory output",
            id="block-without-output",
        ),
        pytest.param(
            lambda s: {"outputs": (s.outputs[1], s.outputs[0])},
            "the result, then the trajectory",
            id="outputs-reversed",
        ),
        pytest.param(
            lambda s: {
                "config": FrozenDict(
                    {
                        **thaw(s.config),
                        "trajectory": {**thaw(s.config)["trajectory"], "schema_version": "v9"},
                    }
                )
            },
            "not a valid",
            id="unknown-recording-schema",
        ),
        pytest.param(
            lambda s: {
                "config": FrozenDict(
                    {
                        **thaw(s.config),
                        "trajectory": {**thaw(s.config)["trajectory"], "max_events": 0},
                    }
                )
            },
            "not a valid",
            id="bounds-out-of-range",
        ),
        pytest.param(
            lambda s: {
                "outputs": (
                    s.outputs[0],
                    ArtifactOutput(name="trajectory", uri="relative/dir", kind=TRAJECTORY_SCHEMA),
                )
            },
            "absolute local path",
            id="relative-trajectory",
        ),
    ],
)
def test_every_other_shape_is_refused(tmp_path: Path, change: Any, reason: str) -> None:
    spec = recorded_spec(tmp_path)
    assert _refusals(spec) == ()
    refusals = _refusals(_rebuild(spec, **change(spec)))
    assert any(reason in r for r in refusals), refusals


def test_a_runner_without_the_named_writer_refuses_rather_than_running_unrecorded(
    tmp_path: Path,
) -> None:
    other = recording().model_copy(update={"writer": WriterIdentity(name="elsewhere", version="9")})
    spec = recorded_spec(tmp_path)
    changed = _rebuild(
        spec, config=FrozenDict({**thaw(spec.config), "trajectory": other.model_dump(mode="json")})
    )
    assert any("trajectory writer this runner lacks" in r for r in _refusals(changed))
    with pytest.raises(HarnessRunRefusedError):
        run(changed)
    assert not (tmp_path / "agent-run.json").exists()
