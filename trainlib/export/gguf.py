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
