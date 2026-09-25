# Control-plane examples

These use the experiment control plane on `main`. It is not in the `0.6.0`
package on PyPI yet, so install from a clone first:

```bash
uv sync --locked                 # add --extra trl to try --compiler trl
```

| Script | What it shows | Needs |
|---|---|---|
| `01_compile_a_candidate.py` | Describe a candidate, see it refused with every reason, compile it into a serializable plan | Nothing: no model, no GPU |
| `02_train.py` | Submit, follow events, wait, and read the result; `--compiler native` or `trl` | A local model and dataset |
| `03_cancel.py` | Cancel while training; the record says `CANCELLED` only once the workload has stopped | As above |
| `04_restart_and_attach.py` | `start` submits and exits while training continues; `attach` adopts it from a new process | As above |

`sft_experiment.py` is the experiment the last three share.

**The model** is a local Hugging Face directory (`save_pretrained` output).
A hub name is refused, because without a pinned revision it names whatever
the hub serves on the day the worker starts. **The dataset** is local JSONL
with a `text` field per line:

```json
{"text": "The quick brown fox jumps over the lazy dog."}
```

Everything an experiment writes goes under `--workdir` (default
`xaytune-workdir/`): the control-plane database `state.db`, the local runtime's
registry and logs in `runtime/`, and trained models in `artifacts/`.

Evaluation is not shown yet. The lifecycle exists, but no evaluator is built
in until the next release step. [Control-plane getting started](https://szaher.github.io/xaytune/control-plane/getting-started/)
walks through all of this.
