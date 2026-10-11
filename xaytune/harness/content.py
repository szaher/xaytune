"""Resolving a harness's content references before anything uses them (ADR-019 E2).

A :class:`~xaytune.core.domain.harness.ContentRef` names text by digest and
says where to find it. Resolution fetches the text a harness can **reach** and
checks that it is what the harness was fingerprinted against: the right
SHA-256, a ``text/*`` media type, and valid UTF-8. Anything else is refused
before compilation, with every reason.

Nothing here opens a connection. A :class:`ContentSource` is supplied by the
host. The only one H03 ships is :class:`MappingContentSource`, which serves
bytes the host already holds.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from xaytune.core.agent_execution import (
    ResolvedContent,
    ResolvedHarnessContent,
    reachable_content,
)
from xaytune.core.domain.harness import AgentHarnessSpec, ContentRef

__all__ = [
    "ContentSource",
    "MappingContentSource",
    "UnverifiedContentError",
    "resolve_harness_content",
]


class UnverifiedContentError(ValueError):
    """Reachable content could not be fetched or did not verify. Carries every reason.

    Reasons name the role of the content, such as ``instructions.system``, and
    never the locator or the bytes.
    """

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("harness content is not verified: " + "; ".join(reasons))


@runtime_checkable
class ContentSource(Protocol):
    """Where a reference's bytes come from. Returns ``None`` for a reference it cannot serve."""

    def fetch(self, ref: ContentRef) -> bytes | None: ...


class MappingContentSource:
    """Bytes the host already holds, by URI. No network, no filesystem."""

    def __init__(self, blobs: Mapping[str, bytes]) -> None:
        self._blobs = dict(blobs)

    def fetch(self, ref: ContentRef) -> bytes | None:
        return self._blobs.get(ref.uri)


def resolve_harness_content(
    harness: AgentHarnessSpec, source: ContentSource
) -> ResolvedHarnessContent:
    """Fetch and verify every content reference *harness* can reach.

    A disabled tool's references are not fetched. Each reachable reference is
    fetched once, even if several roles share it.

    Raises:
        UnverifiedContentError: Naming each role whose content is missing, has
            the wrong digest, is not ``text/*`` or is not UTF-8.
    """
    reasons: list[str] = []
    resolved: dict[str, str] = {}
    for role, content in reachable_content(harness):
        if not isinstance(content, ContentRef):
            continue
        if not content.media_type.lower().startswith("text/"):
            reasons.append(f"{role}: media type is not text/*; a harness reads text")
            continue
        if content.digest in resolved:
            continue
        data = source.fetch(content)
        if data is None:
            reasons.append(f"{role}: the content source does not hold it")
            continue
        if "sha256:" + hashlib.sha256(data).hexdigest() != content.digest:
            reasons.append(f"{role}: the fetched bytes do not have the pinned digest")
            continue
        try:
            resolved[content.digest] = data.decode("utf-8")
        except UnicodeDecodeError:
            reasons.append(f"{role}: the content is not valid UTF-8")
    if reasons:
        raise UnverifiedContentError(tuple(reasons))
    return ResolvedHarnessContent(
        entries=tuple(
            ResolvedContent(digest=digest, text=text) for digest, text in sorted(resolved.items())
        )
    )
