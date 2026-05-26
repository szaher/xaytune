from __future__ import annotations

import io
import threading
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CodeResult:
    """Result of executing user-provided Python code."""

    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    duration: float = 0.0


def _build_namespace(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the execution namespace with xaytune pre-imported."""
    ns: dict[str, Any] = {"__builtins__": __builtins__}
    try:
        import xaytune

        ns["xaytune"] = xaytune
    except ImportError:
        pass
    if extra:
        ns.update(extra)
    return ns


@dataclass
class _ExecState:
    result: CodeResult = field(default_factory=CodeResult)
    done: bool = False


def run_code(
    code: str,
    namespace: dict[str, Any] | None = None,
    timeout: float = 3600,
) -> CodeResult:
    """Execute Python code with stdout/stderr capture and error handling.

    The code runs in the current process with ``xaytune`` pre-imported
    in the namespace. Output and errors are captured and returned in a
    :class:`CodeResult`.

    Args:
        code: Python source code to execute.
        namespace: Extra names to inject into the execution namespace.
        timeout: Maximum wall-clock seconds before the run is considered
            timed out.  The thread cannot be forcibly killed, but the
            result will report a timeout error.
    """
    ns = _build_namespace(namespace)
    state = _ExecState()

    def _exec() -> None:
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        t0 = time.monotonic()
        try:
            compiled = compile(code, "<studio>", "exec")
            with redirect_stdout(out_buf), redirect_stderr(err_buf):
                exec(compiled, ns)  # noqa: S102
        except SyntaxError as exc:
            state.result.error = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
        except Exception as exc:
            state.result.error = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
        finally:
            state.result.stdout = out_buf.getvalue()
            state.result.stderr = err_buf.getvalue()
            state.result.duration = time.monotonic() - t0
            state.done = True

    thread = threading.Thread(target=_exec, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if not state.done:
        state.result.error = (
            f"Execution timed out after {timeout:.0f}s. "
            "The code may still be running in the background."
        )
        state.result.duration = timeout

    return state.result


CODE_TEMPLATES: dict[str, str] = {
    "Fine-tuning": (
        "import xaytune\n"
        "\n"
        "state = xaytune.finetune(\n"
        '    model="meta-llama/Llama-3-8B",\n'
        '    dataset="data/train.jsonl",\n'
        '    format="alpaca",\n'
        "    num_epochs=3,\n"
        "    learning_rate=2e-4,\n"
        ")\n"
        "\n"
        "print(f\"Final loss: {state.metrics.get('loss', 'N/A')}\")\n"
    ),
    "LoRA Fine-tuning": (
        "import xaytune\n"
        "\n"
        "state = xaytune.finetune(\n"
        '    model="meta-llama/Llama-3-8B",\n'
        '    dataset="data/train.jsonl",\n'
        '    method="lora",\n'
        '    format="alpaca",\n'
        "    num_epochs=3,\n"
        "    learning_rate=2e-4,\n"
        ")\n"
        "\n"
        "print(f\"Final loss: {state.metrics.get('loss', 'N/A')}\")\n"
    ),
    "Alignment (DPO)": (
        "import xaytune\n"
        "\n"
        "state = xaytune.align(\n"
        '    model="meta-llama/Llama-3.1-8B-Instruct",\n'
        '    dataset="data/preferences.jsonl",\n'
        '    method="dpo",\n'
        '    format="preference",\n'
        "    num_epochs=1,\n"
        "    learning_rate=5e-6,\n"
        ")\n"
        "\n"
        "print(f\"Final loss: {state.metrics.get('loss', 'N/A')}\")\n"
    ),
    "Alignment (GRPO)": (
        "import xaytune\n"
        "\n"
        "state = xaytune.align(\n"
        '    model="meta-llama/Llama-3.1-8B-Instruct",\n'
        '    dataset="data/prompts.jsonl",\n'
        '    method="grpo",\n'
        "    num_epochs=1,\n"
        "    learning_rate=1e-6,\n"
        '    reward_name="format_check",\n'
        "    max_new_tokens=256,\n"
        "    temperature=0.7,\n"
        "    group_size=4,\n"
        ")\n"
        "\n"
        "print(f\"Final loss: {state.metrics.get('loss', 'N/A')}\")\n"
    ),
    "Pre-training": (
        "import xaytune\n"
        "\n"
        "state = xaytune.pretrain(\n"
        '    model="meta-llama/Llama-3-8B",\n'
        '    dataset="data/corpus/",\n'
        '    format="text",\n'
        "    num_epochs=1,\n"
        "    learning_rate=3e-4,\n"
        "    max_steps=10000,\n"
        ")\n"
        "\n"
        "print(f\"Final loss: {state.metrics.get('loss', 'N/A')}\")\n"
    ),
    "Custom": (
        "import xaytune\n"
        "\n"
        "# Write your training code here.\n"
        "# The xaytune module is pre-imported.\n"
        "#\n"
        "# Available functions:\n"
        "#   xaytune.finetune(model=..., dataset=..., ...)\n"
        '#   xaytune.align(model=..., dataset=..., method="dpo", ...)\n'
        "#   xaytune.pretrain(model=..., dataset=..., ...)\n"
        "#   xaytune.evaluate(model=..., benchmarks=[...], ...)\n"
        "\n"
        'print("Hello from xaytune Studio!")\n'
    ),
}
