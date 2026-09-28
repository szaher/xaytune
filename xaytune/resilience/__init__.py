"""Structured incident detection and deterministic classification (PR-017).

Consumes the existing core telemetry observations. It proposes no response,
changes no attempt, and calls no runtime. Free-form detail is evidence only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from xaytune.core.domain.incident import (
    AttemptContext,
    DetectorProvenance,
    Incident,
    IncidentCandidate,
    IncidentCategory,
)
from xaytune.core.fingerprint import fingerprint
from xaytune.core.immutable import FrozenDict
from xaytune.core.telemetry import (
    EvaluationFailedPayload,
    EvaluationObservation,
    IncidentObservedPayload,
    NumericalInstabilityObserved,
    TrainingFailedPayload,
    TrainingObservation,
)

__all__ = [
    "CheckpointFailureDetector",
    "CudaOOMDetector",
    "IncidentClassifier",
    "IncidentDetector",
    "NaNInfDetector",
    "ProcessFailureDetector",
]

_FAILURES = (IncidentObservedPayload, TrainingFailedPayload, EvaluationFailedPayload)


class IncidentDetector(Protocol):
    name: str
    version: str

    def inspect(
        self, signal: TrainingObservation | EvaluationObservation, context: AttemptContext
    ) -> IncidentCandidate | None: ...


class _ReasonDetector:
    """Exact structured reason codes; no interpretation of exception text."""

    name: str
    version = "1"
    reasons: Mapping[str, IncidentCategory]

    def inspect(
        self, signal: TrainingObservation | EvaluationObservation, context: AttemptContext
    ) -> IncidentCandidate | None:
        if not isinstance(signal, _FAILURES):
            return None
        category = self.reasons.get(signal.reason)
        if category is None:
            return None
        return IncidentCandidate(
            category=category,
            detector=DetectorProvenance(name=self.name, version=self.version),
            reason=f"structured reason: {signal.reason}",
        )


class ProcessFailureDetector(_ReasonDetector):
    name = "process-failure"
    reasons = {
        reason: IncidentCategory.PROCESS_FAILURE
        for reason in ("process-failure", "nonzero-exit", "signalled", "spawn-failed")
    }

    def inspect(
        self, signal: TrainingObservation | EvaluationObservation, context: AttemptContext
    ) -> IncidentCandidate | None:
        if isinstance(signal, IncidentObservedPayload):
            code = signal.exit_code
            if (
                (signal.reason == "nonzero-exit" and code is not None and code <= 0)
                or (signal.reason == "signalled" and code is not None and code >= 0)
                or (signal.reason == "spawn-failed" and code is not None)
            ):
                return None
        return super().inspect(signal, context)


class CudaOOMDetector(_ReasonDetector):
    name = "cuda-oom"
    reasons = {"cuda-oom": IncidentCategory.CUDA_OOM}


class CheckpointFailureDetector(_ReasonDetector):
    name = "checkpoint-failure"
    reasons = {
        "checkpoint-write-failure": IncidentCategory.CHECKPOINT_WRITE_FAILURE,
        "checkpoint-corruption": IncidentCategory.CHECKPOINT_CORRUPTION,
        "checkpoint-incompatible": IncidentCategory.CHECKPOINT_INCOMPATIBLE,
    }


class NaNInfDetector:
    name = "nan-inf"
    version = "1"

    def inspect(
        self, signal: TrainingObservation | EvaluationObservation, context: AttemptContext
    ) -> IncidentCandidate | None:
        if not isinstance(signal, NumericalInstabilityObserved):
            return None
        if signal.observation == "nan":
            category = IncidentCategory.NUMERICAL_NAN
        elif signal.observation in ("positive-infinity", "negative-infinity"):
            category = IncidentCategory.NUMERICAL_INF
        else:
            return None
        return IncidentCandidate(
            category=category,
            detector=DetectorProvenance(name=self.name, version=self.version),
            reason=f"{signal.quantity} reported {signal.observation}",
        )


class IncidentClassifier:
    """Matching diagnoses agree, or the category is UNKNOWN. No guessed priority."""

    name = "structured-incidents"
    version = "1"

    def __init__(self, detectors: Sequence[IncidentDetector] | None = None) -> None:
        self.detectors = tuple(
            detectors
            if detectors is not None
            else (
                ProcessFailureDetector(),
                CudaOOMDetector(),
                NaNInfDetector(),
                CheckpointFailureDetector(),
            )
        )

    def inspect(
        self,
        signal: TrainingObservation | EvaluationObservation,
        context: AttemptContext,
        *,
        evidence: Mapping[str, Any],
    ) -> Incident | None:
        if not isinstance(signal, (*_FAILURES, NumericalInstabilityObserved)):
            return None
        if fingerprint(evidence["payload"]["data"]) != fingerprint(signal.model_dump(mode="json")):
            raise ValueError("signal and evidence disagree")
        if (
            isinstance(signal, IncidentObservedPayload)
            and signal.reason in ("nonzero-exit", "signalled")
            and signal.metadata.get("cancelled") is True
        ):
            return None
        candidates = tuple(
            candidate
            for detector in self.detectors
            if (candidate := detector.inspect(signal, context)) is not None
        )
        return self.classify(candidates, context, evidence=evidence)

    def classify(
        self,
        candidates: Sequence[IncidentCandidate],
        context: AttemptContext,
        *,
        evidence: Mapping[str, Any],
    ) -> Incident:
        categories = {candidate.category for candidate in candidates}
        category = next(iter(categories)) if len(categories) == 1 else IncidentCategory.UNKNOWN
        return Incident(
            context=context,
            stream_generation=evidence["stream_generation"],
            sequence=evidence["sequence"],
            category=category,
            evidence=FrozenDict(evidence),
            candidates=tuple(candidates),
            detectors=tuple(
                DetectorProvenance(name=detector.name, version=detector.version)
                for detector in self.detectors
            ),
            classifier=DetectorProvenance(name=self.name, version=self.version),
        )
