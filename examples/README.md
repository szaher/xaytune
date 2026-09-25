# Examples

- **[`control_plane/`](control_plane/)**: the experiment control plane,
  first released in `1.0.0a1`. Compile a candidate, submit it, follow it,
  cancel it, attach to it from another process, evaluate it and decide.
- **The notebooks and `end_to_end.py` in this directory** use the **legacy
  trainer API** (`xaytune.finetune()`, `align()`, `evaluate()`, the CLI and
  pipelines), the library in the `0.6.0` package on PyPI, still included in
  `1.0.0a1`. It remains
  supported, and each notebook opens with a note saying so.
