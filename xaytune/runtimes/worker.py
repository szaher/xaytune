"""What passes between a runtime and the worker it starts.

A worker is told where things are and reports what happened. This module is
the part of that exchange every runtime and every worker agrees on, kept out of
any one backend so that a Ray or Training Hub runtime can supervise the same
worker without either side changing.

Torch-free: both the controller-side launcher and the worker import it.
"""

from __future__ import annotations

__all__ = ["TOPOLOGY_VARIABLES"]

TOPOLOGY_VARIABLES: frozenset[str] = frozenset(
    {
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "GROUP_WORLD_SIZE",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "ROLE_NAME",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_USE_AGENT_STORE",
        "TORCHELASTIC_ERROR_FILE",
    }
)
"""The variables through which torch.distributed learns where a process sits.

**Placement topology belongs to the runtime, never to whatever shell started
the controller.** A controller launched under ``torchrun`` has these set; a
worker that inherited them would believe it was rank 3 of 8, try to join a
process group at the controller's ``MASTER_ADDR``, and either fail or hang
waiting for seven ranks that were never started. Nothing in the plan would say
so, because nothing in the plan caused it.

So a runtime removes them from what a worker inherits, and sets them only when
it has deliberately placed the worker in a group of its own.

Behavioural knobs that happen to share the prefix -- ``NCCL_*`` tuning,
``TORCH_NCCL_ASYNC_ERROR_HANDLING`` -- are not placement and are left alone.
So is ``CUDA_VISIBLE_DEVICES``: an operator restricting which devices a
controller may use is a constraint a worker should keep, and removing it would
hand the worker every device on the machine.
"""
