# Dependencies

Last updated: 2026-06-03 14:00

## Task Dependency Graph

```
=== EPIC-0 (Foundational) ===

TASK-025 (SFT prompt masking)      ── no deps
│
└── TASK-026 (preference log-prob masking) ── blocked-by TASK-025

TASK-027 (ORPO crash fix)          ── no deps
│
└── TASK-030 (Studio alignment wiring) ── blocked-by TASK-027

TASK-028 (QLoRA preparation)       ── no deps
TASK-029 (DeepSpeed loop fix)      ── no deps
TASK-031 (PPO rename/mark experimental) ── no deps

=== EPIC-1–11 (Original backlog) ===

TASK-001 (ORPO stability)          ── blocked-by TASK-027 (ORPO must work first)
TASK-002 (SimPO clamp)             ── no deps
TASK-003 (eval_callback metrics)   ── no deps
TASK-004 (torch.load map_location) ── no deps
TASK-005 (checkpoint serialization)── no deps
TASK-006 (reinforce in validation) ── no deps
│
├── TASK-007 (validate from setup_training) ── blocked-by TASK-006
│   │
│   └── TASK-009 (pretrain validation) ── blocked-by TASK-007
│
TASK-008 (override key validation) ── no deps
TASK-010 (MLflow flatten)          ── no deps
TASK-011 (logging isolation)       ── no deps
TASK-012 (import guards)           ── no deps
TASK-013 (model_merge config.json) ── no deps
TASK-014 (GGUF command)            ── no deps
TASK-015 (hub tokenizer warning)   ── no deps
TASK-016 (format_text warning)     ── no deps
TASK-017 (preference shuffle)      ── no deps
TASK-018 (studio dropdown)         ── no deps
TASK-019 (CLI eval default)        ── no deps
TASK-020 (numpy seed)              ── no deps
TASK-021 (gloo fallback)           ── no deps
TASK-022 (lr_finder device)        ── no deps
TASK-023 (constant warmup)         ── no deps
TASK-024 (agent BOS)               ── no deps
```

## Critical Path

```
TASK-025 → TASK-026  (SFT masking enables preference masking)
TASK-027 → TASK-001  (ORPO must work before stabilizing its numerics)
TASK-027 → TASK-030  (ORPO must work before Studio can wire it)
TASK-006 → TASK-007 → TASK-009  (validation chain)
```

The longest critical path is: **TASK-025 → TASK-026** (SFT + preference masking). Estimated: M + M = ~2 days.

## Parallelization Strategy

**EPIC-0 NOW (7 tasks):**
- TASK-025, TASK-027, TASK-028, TASK-029, TASK-031 can start simultaneously.
- TASK-026 waits for TASK-025.
- TASK-030 waits for TASK-027.

**Original NOW (6 tasks):** All independent except TASK-001 waits for TASK-027.

**NEXT (13 tasks):** TASK-009 waits for TASK-007 waits for TASK-006. Everything else independent.

**LATER (5 tasks):** All independent.
