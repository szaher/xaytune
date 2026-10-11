"""One envelope for every kind of candidate, without moving a single training identity (H02).

```text
ExperimentCandidate  envelope_version = "xaytune.candidate/v1alpha1"
  payload, by kind
  ├─ TrainingCandidate      kind="training"       spec: CandidateSpec
  │                                                 → candidate v2 (and v1 for lookup)
  └─ AgentHarnessCandidate  kind="agent-harness"  harness: AgentHarnessSpec
                                                    → agent-harness v1
```

**Training is unchanged, and stays the default nowhere.** ``CandidateSpec`` is
carried as it is -- its serialization inside the envelope is byte-for-byte its
serialization outside -- and its scientific fingerprint is its own
``candidate_fingerprint()``. The envelope adds no hash of its own: a third
identity over both kinds would be one nobody needs and everyone must keep
stable.

**Historical records read as what they were.** Every candidate stored before
H02 is an unwrapped ``CandidateSpec`` (inside a ``CandidateSpecSnapshot``).
:func:`read_candidate` recognizes one by the absence of an envelope and reads it
as training through the existing class, rewriting nothing. Anything else is
refused: an unknown envelope version or kind, a harness without its envelope,
a snapshot of a shape this build does not know. Nothing is coerced from one
kind to the other.

**Not yet durable, on purpose.** Nodes, snapshots, specs, proposals and
planning contexts still hold ``CandidateSpec`` (ADR-018 D10). The envelope is
the contract H03 persists through, behind its own migration review.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Any, Final, Literal

from pydantic import ConfigDict, Field, ValidationError

from xaytune.core.domain.candidate import CandidateSpec
from xaytune.core.domain.experiment import CandidateSpecSnapshot
from xaytune.core.domain.harness import AGENT_HARNESS_SCHEMA, AgentHarnessSpec
from xaytune.core.errors import XaytuneError
from xaytune.core.immutable import FrozenDomainModel

__all__ = [
    "CANDIDATE_ENVELOPE_VERSION",
    "AgentHarnessCandidate",
    "CandidateEnvelopeError",
    "CandidateKind",
    "ExperimentCandidate",
    "TrainingCandidate",
    "candidate_of_snapshot",
    "read_candidate",
]

CANDIDATE_ENVELOPE_VERSION: Final = "xaytune.candidate/v1alpha1"


class CandidateEnvelopeError(XaytuneError):
    """A record is not a candidate this build can read: unknown version or kind, or ambiguous."""


class CandidateKind(str, Enum):
    """What a candidate optimizes. Not the training algorithm: SFT and GRPO are both training."""

    TRAINING = "training"
    AGENT_HARNESS = "agent-harness"


class TrainingCandidate(FrozenDomainModel):
    """A model-training proposition: the existing ``CandidateSpec``, unchanged."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    kind: Literal["training"] = "training"
    spec: CandidateSpec


class AgentHarnessCandidate(FrozenDomainModel):
    """A harness proposition: a pinned model operated one way, weights fixed."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    kind: Literal["agent-harness"] = "agent-harness"
    harness: AgentHarnessSpec


CandidatePayload = Annotated[TrainingCandidate | AgentHarnessCandidate, Field(discriminator="kind")]


class ExperimentCandidate(FrozenDomainModel):
    """A candidate of any kind, versioned for interchange.

    Construct one with :meth:`training` or :meth:`agent_harness`, which state
    the version; read one with :func:`read_candidate`; write one with
    ``model_dump(mode="json")``. Parsing never supplies a missing version.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    envelope_version: Literal["xaytune.candidate/v1alpha1"]
    """Required when parsing: a record that does not say which envelope it is, is not one."""
    payload: CandidatePayload

    @classmethod
    def training(cls, spec: CandidateSpec) -> ExperimentCandidate:
        return cls(
            envelope_version=CANDIDATE_ENVELOPE_VERSION, payload=TrainingCandidate(spec=spec)
        )

    @classmethod
    def agent_harness(cls, harness: AgentHarnessSpec) -> ExperimentCandidate:
        return cls(
            envelope_version=CANDIDATE_ENVELOPE_VERSION,
            payload=AgentHarnessCandidate(harness=harness),
        )

    @property
    def kind(self) -> CandidateKind:
        return CandidateKind(self.payload.kind)

    def scientific_fingerprint(self) -> str:
        """The payload's own identity: training v2, or harness v1. Never a combined hash."""
        payload = self.payload
        if isinstance(payload, TrainingCandidate):
            return payload.spec.candidate_fingerprint()
        return payload.harness.harness_fingerprint()

    def fingerprints_for_lookup(self) -> tuple[str, ...]:
        """Every identity this candidate may be recorded under, newest first.

        Training keeps its ``(v2, v1)`` lookup exactly; a harness has only v1.
        """
        payload = self.payload
        if isinstance(payload, TrainingCandidate):
            return payload.spec.candidate_fingerprints_for_lookup()
        return (payload.harness.harness_fingerprint(),)


def read_candidate(record: Mapping[str, Any]) -> ExperimentCandidate:
    """Read *record* as a candidate: an envelope, or a historical unwrapped ``CandidateSpec``.

    The envelope is read only at the version this build implements, and its
    payload only as the kinds it knows. A mapping without an envelope is a
    record written before H02, which was always a training candidate; it is
    read through ``CandidateSpec`` and wrapped, not rewritten.

    Raises:
        CandidateEnvelopeError: For an unknown envelope version or kind, a
            harness spec without its envelope, or a record that is neither.
    """
    if not isinstance(record, Mapping):
        raise CandidateEnvelopeError(
            f"a candidate record is a mapping, not {type(record).__name__}"
        )
    if "envelope_version" in record:
        version = record["envelope_version"]
        if version != CANDIDATE_ENVELOPE_VERSION:
            raise CandidateEnvelopeError(
                f"candidate envelope {version!r} is not one this build reads; it reads "
                f"{CANDIDATE_ENVELOPE_VERSION!r}"
            )
        payload = record.get("payload")
        kind = payload.get("kind") if isinstance(payload, Mapping) else None
        if kind not in {member.value for member in CandidateKind}:
            raise CandidateEnvelopeError(
                f"candidate kind {kind!r} is not one this build reads; it reads "
                f"{sorted(member.value for member in CandidateKind)}"
            )
        try:
            return ExperimentCandidate.model_validate(record)
        except ValidationError as error:
            raise CandidateEnvelopeError(f"invalid {kind} candidate: {error}") from error
    if str(record.get("schema_version", "")).startswith("xaytune.agent-harness/"):
        raise CandidateEnvelopeError(
            f"an unwrapped {AGENT_HARNESS_SCHEMA} spec is not a historical record: a harness "
            f"candidate is read only inside its envelope, never guessed from its fields"
        )
    try:
        spec = CandidateSpec.model_validate(record)
    except ValidationError as error:
        raise CandidateEnvelopeError(
            f"the record has no candidate envelope and is not a historical CandidateSpec: {error}"
        ) from error
    return ExperimentCandidate.training(spec)


def candidate_of_snapshot(snapshot: CandidateSpecSnapshot) -> ExperimentCandidate:
    """A node's stored snapshot as a candidate: training, at the one shape it has had.

    Raises:
        CandidateEnvelopeError: If the snapshot records a ``spec_version``
            this build does not know.
    """
    if snapshot.spec_version != "1":
        raise CandidateEnvelopeError(
            f"candidate snapshot version {snapshot.spec_version!r} is not one this build reads"
        )
    return ExperimentCandidate.training(snapshot.candidate)
