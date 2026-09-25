# Examples

- **[`control_plane/`](control_plane/)**: the experiment control plane on
  `main`. Compile a candidate, submit it, follow it, cancel it, and attach to
  it from another process.
- **The notebooks and `end_to_end.py` in this directory** use the **legacy
  trainer API** (`xaytune.finetune()`, `align()`, `evaluate()`, the CLI and
  pipelines), the library in the `0.6.0` package on PyPI. It remains
  supported, and each notebook opens with a note saying so.
