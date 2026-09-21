# Architecture images

These self-contained SVGs render directly in GitHub Markdown and scale without
losing label clarity. Each includes an accessible title and description.

| Image | Source of truth |
|---|---|
| [Architecture overview](architecture-overview.svg) | Spec README and chapter 02 |
| [Execution path](execution-path.svg) | Chapter 02, ADR-013 |
| [Evaluation path](evaluation-path.svg) | Chapter 10, ADR-015 |
| [Dependency boundaries](dependency-boundaries.svg) | Chapter 02 |
| [Implementation order](implementation-order.svg) | Chapter 15 |

Edit `render.py`, then regenerate from the repository root:

```bash
python xaytune-training-harness-spec/assets/diagrams/render.py
```

The generator uses only the Python standard library. Commit both the generator
and regenerated SVGs. Check the images in a browser after changing labels or
layout. The diagrams describe target contracts; the implementation plan defines
which integrations exist at each phase.
