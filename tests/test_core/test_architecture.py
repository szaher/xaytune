"""Architecture invariants for the control-plane core.

These are the executable form of ADR-010 / Invariant H ("core does not require
ML runtimes") and the dependency-direction rules in the architecture spec.
They are cheap and they guard every later change, not just this one.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

CORE_DIR = Path(__file__).resolve().parents[2] / "xaytune" / "core"

# Packages xaytune.core must never pull in, directly or transitively.
FORBIDDEN_ROOTS = frozenset(
    {
        "bitsandbytes",
        "datasets",
        "deepspeed",
        "gradio",
        "kubernetes",
        "mlflow",
        "peft",
        "ray",
        "torch",
        "torchft",
        "torchtune",
        "transformers",
        "trl",
        "verl",
        "wandb",
    }
)

# Existing xaytune subpackages that sit *above* the core in the dependency
# direction. The core may not reach back into them.
FORBIDDEN_XAYTUNE_MODULES = frozenset(
    {
        "xaytune.cli",
        "xaytune.compilation",
        "xaytune.config",
        "xaytune.data",
        "xaytune.eval",
        "xaytune.export",
        "xaytune.logging",
        "xaytune.models",
        "xaytune.pipeline",
        "xaytune.plugins",
        "xaytune.recipes",
        "xaytune.runtimes",
        "xaytune.studio",
        "xaytune.trainer",
    }
)

_BLOCKER_SCRIPT = """
import sys

BLOCKED = {blocked!r}


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{{name}} must not be importable from xaytune.core")
        return None


sys.meta_path.insert(0, _Blocker())

{body}

print("OK")
"""


def _run_isolated(body: str) -> subprocess.CompletedProcess[str]:
    script = _BLOCKER_SCRIPT.format(blocked=sorted(FORBIDDEN_ROOTS), body=body)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(CORE_DIR.parents[1]),
    )


def _iter_core_modules():
    for path in sorted(CORE_DIR.rglob("*.py")):
        yield path, ast.parse(path.read_text(), filename=str(path))


def _imported_modules(tree: ast.AST) -> set[str]:
    """Every module named by an import, including imports inside functions."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import, stays inside the core
                continue
            if node.module:
                found.add(node.module)
    return found


class TestCoreImportIsolation:
    def test_core_imports_without_ml_runtimes(self):
        """The whole point of ADR-010: importable on a laptop without torch."""
        result = _run_isolated("import xaytune.core")
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_core_domain_usable_without_ml_runtimes(self):
        """Not just importable — the domain must be *usable* with no ML stack.

        Guards against a heavy import hidden inside a function body, which a
        bare ``import xaytune.core`` would not reach.
        """
        body = """
import xaytune.core as core

experiment = core.Experiment(
    id=core.ExperimentId.generate(),
    name="no-torch",
    objective=core.Objective(
        primary=core.ObjectiveMetric(name="task_success", direction="maximize"),
    ),
    controller_host=core.ControllerHostRef(kind="embedded"),
)
active = experiment.with_status(core.ExperimentStatus.ACTIVE)
assert core.Experiment.model_validate_json(active.model_dump_json()) == active
assert "torch" not in sys.modules
"""
        result = _run_isolated(body)
        assert result.returncode == 0, result.stderr

    def test_importing_xaytune_does_not_import_torch(self):
        """Top-level ``import xaytune`` stays lazy.

        ``import xaytune.core`` executes the package ``__init__`` first, so the
        core boundary only holds while the public API resolves lazily.
        """
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, xaytune; print('torch' in sys.modules)",
            ],
            capture_output=True,
            text=True,
            cwd=str(CORE_DIR.parents[1]),
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "False"


class TestCoreImportFootprint:
    """Pin what ``import xaytune.core`` costs, so it cannot creep.

    ADR-010 allows lightweight schema/utility packages. The package
    ``__init__`` binds ``pipeline`` eagerly (it collides with the
    ``xaytune/pipeline.py`` submodule), which is what drags in
    ``xaytune.config`` and therefore PyYAML.
    """

    ALLOWED_THIRD_PARTY = frozenset(
        {
            "annotated_types",
            "cython_runtime",
            "pydantic",
            "pydantic_core",
            "typing_extensions",
            "typing_inspection",
            "xaytune",
            "yaml",
        }
    )

    def test_third_party_footprint_is_pinned(self):
        script = """
import sys

before = set(sys.modules)
import xaytune.core
roots = {m.split(".")[0] for m in set(sys.modules) - before}
third = sorted(r for r in roots if r not in sys.stdlib_module_names and not r.startswith("_"))
print(",".join(third))
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(CORE_DIR.parents[1]),
        )
        assert result.returncode == 0, result.stderr
        pulled = set(result.stdout.strip().split(","))
        unexpected = pulled - self.ALLOWED_THIRD_PARTY
        assert not unexpected, (
            f"import xaytune.core pulled in unexpected packages: {sorted(unexpected)}. "
            f"Add them here deliberately or keep them out of the core import path."
        )


class TestCoreDependencyDirection:
    def test_core_declares_no_forbidden_imports(self):
        offenders: list[str] = []
        for path, tree in _iter_core_modules():
            for module in _imported_modules(tree):
                if module.split(".")[0] in FORBIDDEN_ROOTS:
                    offenders.append(f"{path.name}: {module}")
        assert not offenders, f"xaytune.core must not import ML runtimes: {offenders}"

    def test_core_does_not_reach_back_into_upper_layers(self):
        offenders: list[str] = []
        for path, tree in _iter_core_modules():
            for module in _imported_modules(tree):
                if not module.startswith("xaytune."):
                    continue
                if module.startswith("xaytune.core"):
                    continue
                root = ".".join(module.split(".")[:2])
                if root in FORBIDDEN_XAYTUNE_MODULES:
                    offenders.append(f"{path.name}: {module}")
        assert not offenders, f"xaytune.core must not depend on upper layers: {offenders}"

    def test_scan_actually_found_modules(self):
        """Guard against the scan silently passing because it found nothing."""
        modules = list(_iter_core_modules())
        assert len(modules) >= 8, f"only found {len(modules)} core modules"


class TestForbiddenImportDetection:
    """The scan must actually fail when something forbidden appears."""

    @pytest.mark.parametrize(
        "source",
        [
            "import torch",
            "from transformers import AutoModel",
            "def f():\n    import ray\n",
        ],
    )
    def test_detects_forbidden_import(self, source):
        tree = ast.parse(source)
        found = _imported_modules(tree)
        assert any(m.split(".")[0] in FORBIDDEN_ROOTS for m in found)
