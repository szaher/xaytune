# Control-plane examples

These use the experiment control plane, first released in `1.0.0a1` (a
pre-release; the `0.6.0` package has no control plane). Install it, or install
from a clone to match CI exactly:

```bash
pip install "xaytune[trl]==1.0.0a1"   # --compiler trl needs the trl extra
uv sync --locked                 # or: pip install -e .
uv sync --locked --extra trl     # for --compiler trl; or: pip install -e ".[trl]"
```

| Script | What it shows | Needs |
|---|---|---|
| `01_compile_a_candidate.py` | Describe a candidate, see it refused with every reason, compile it into a serializable plan | Nothing: no model, no GPU |
| `02_train.py` | Submit, follow events, wait, and read the result; `--compiler native` or `trl` | A local model and dataset |
| `03_cancel.py` | Cancel while training; the record says `CANCELLED` only once the workload has stopped | As above |
| `04_restart_and_attach.py` | `start` submits and exits while training continues; `attach` adopts it from a new process | As above |
| `05_train_and_evaluate.py` | Train, evaluate with the built-in `native` evaluator, and decide against a loss `--target`: the experiment ends `SUCCEEDED` or `FAILED` | As above, plus a held-out JSONL file |

`sft_experiment.py` is the experiment 02 to 05 share.

**The model** is a local Hugging Face directory (`save_pretrained` output).
A hub name is refused, because without a pinned revision it names whatever
the hub serves on the day the worker starts. **The dataset** is local JSONL
with a `text` field per line:

```json
{"text": "The quick brown fox jumps over the lazy dog."}
```

Everything an experiment writes goes under `--workdir` (default
`xaytune-workdir/`): the control-plane database `state.db`, the local runtime's
registry and logs in `runtime/`, and trained models and evaluation reports in
`artifacts/`.

[Control-plane getting started](https://szaher.github.io/xaytune/control-plane/getting-started/)
walks through all of this.
