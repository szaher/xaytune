from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def to_gguf(
    model_path: str,
    *,
    output: str,
    quantization: str = "Q4_K_M",
) -> None:
    """Convert a HuggingFace model to GGUF format via llama.cpp.

    Args:
        model_path: Path to a saved HuggingFace model directory.
        output: Destination path for the ``.gguf`` file.
        quantization: Quantization scheme (default ``"Q4_K_M"``).

    Raises:
        FileNotFoundError: If *model_path* does not exist.
        RuntimeError: If the conversion subprocess fails.
    """
    model_dir = Path(model_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "llama_cpp.convert",
        str(model_dir),
        "--outfile",
        str(output_path),
        "--outtype",
        quantization,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"GGUF conversion failed: {result.stderr}")
