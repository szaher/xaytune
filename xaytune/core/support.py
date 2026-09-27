"""Whether a plugin can handle a request, and if not, why.

Shared by every extension point that answers "can you do this?" before doing
it -- compilers, evaluators, action plugins -- so a refusal reads the same
wherever it comes from.
"""

from __future__ import annotations

from pydantic import Field

from xaytune.core.immutable import FrozenDomainModel

__all__ = ["SupportResult"]


class SupportResult(FrozenDomainModel):
    """Whether a plugin can handle a request, and if not, why.

    A bare ``False`` is not actionable: a planner that learns only "no" cannot
    tell a missing algorithm from an unsupported adapter, and cannot propose
    anything better. So refusal carries reasons.
    """

    supported: bool
    reasons: tuple[str, ...] = Field(default_factory=tuple)

    def __bool__(self) -> bool:
        return self.supported
