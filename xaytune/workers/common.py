"""What every worker does the same way, whichever trainer it drives.

A worker's trainer-specific half is translation: a wire config into the
trainer's own configuration, and the trainer's events into observations. The
rest is the execution contract, and it must not vary by trainer, because the
controller reads it without knowing which trainer ran:

- **artifact publication** -- written, *confirmed* to exist, and only then
  reported, with a publication failure kept distinct from a training failure;
- **observation reporting** -- a value the vocabulary refuses is one dropped
  observation, a broken channel is a failed run;
- **failure reasons** and the environment a worker requires.

Kept in one place so that "ArtifactProduced means the model exists" is one
implementation, not a property each worker has to remember to re-establish.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from xaytune.core.ids import ArtifactId
from xaytune.core.refs import ArtifactRef
from xaytune.core.telemetry import ArtifactProducedPayload, IncidentObservedPayload
from xaytune.runtimes.worker import ObservationWriter

__all__ = [
    "failure_reason",
    "nonfinite",
    "publish_model",
    "report",
    "require_environment",
    "require_model_artifact",
]


def report(writer: ObservationWriter, build: Any) -> None:
    """Build one observation and write it, keeping the two failures apart.

    A value the vocabulary refuses is one bad observation: it is reported
    on stderr and skipped, and training continues, because the observation
    channel is not authoritative and one malformed metric is no reason to
    lose a run. A failure to *write* is not caught: that is the runtime's
    channel broken, which is an execution-contract failure.
    """
    try:
        observation = build()
    except ValidationError as exc:
        print(f"xaytune: dropped an observation the vocabulary refused: {exc}", file=sys.stderr)
        return
    writer.write(observation)


_WEIGHT_PATTERNS = ("*.safetensors", "*.bin", "*.safetensors.index.json", "*.bin.index.json")


def require_model_artifact(target: Path, *, with_tokenizer: bool) -> None:
    """Confirm the artifact exists, rather than inferring it from silence.

    ``save_pretrained`` does not always raise when it fails. Given a path that
    is a file, it logs "should be a directory" and returns -- and a worker that
    read *no exception* as *written* would announce a model that does not
    exist, under a run reporting success. So the result is checked for the
    files a loader needs: a config and the weights, and the tokenizer's
    config when one was saved.

    Structural rather than a full load. Loading the model back would prove
    more, at the cost of a second copy of the weights in memory on the
    training host.

    Raises:
        FileNotFoundError: Naming what is missing.
    """
    if not target.is_dir():
        raise FileNotFoundError(f"{target} is not a directory; nothing was written there")

    missing = []
    if not (target / "config.json").is_file():
        missing.append("config.json")
    if not any(any(target.glob(pattern)) for pattern in _WEIGHT_PATTERNS):
        missing.append("model weights")
    if with_tokenizer and not (target / "tokenizer_config.json").is_file():
        missing.append("tokenizer_config.json")
    if missing:
        raise FileNotFoundError(f"{target} is missing {', '.join(missing)}")


def nonfinite(value: float) -> Literal["nan", "positive-infinity", "negative-infinity"]:
    if math.isnan(value):
        return "nan"
    return "positive-infinity" if value > 0 else "negative-infinity"


def publish_model(model: Any, tokenizer: Any, target: Path, writer: ObservationWriter) -> None:
    """Produce the model the plan declared, and say so.

    Whatever a trainer leaves behind on its own -- the native trainer's raw
    ``state_dict`` checkpoint, a ``Trainer``'s ``checkpoint-*`` directories --
    is that trainer's format, not the run's output. The plan's declared
    ``model`` output is a loadable artifact, so it is written with
    ``save_pretrained``, the same form a candidate names when it refers to a
    model: what a run produces is something the next candidate can train from,
    whichever trainer produced it.

    Reported after it is **confirmed** to exist, not merely after the save
    returned. An ``ArtifactProduced`` for a write that silently did nothing
    would be a claim with nothing behind it -- which is not hypothetical:
    ``save_pretrained`` returns without raising when handed a file.

    No content digest yet. Hashing multi-gigabyte weights on the training host
    is a real cost, and whether it happens here, asynchronously, or in an
    artifact store is a decision worth making deliberately rather than by
    default.

    Raises:
        Exception: Whatever the save raised, after reporting it -- the run did
            train, but did not produce what the plan declared, so the process
            must not exit as though it had.
    """
    try:
        # Unwrapped if distributed wrapping ever applies; a plain model here.
        getattr(model, "module", model).save_pretrained(target)
        if tokenizer is not None:
            tokenizer.save_pretrained(target)
        require_model_artifact(target, with_tokenizer=tokenizer is not None)
    except Exception as exc:
        writer.write(
            IncidentObservedPayload(
                reason="artifact-publication-failed",
                detail=f"training completed but the model could not be written: {exc}",
            )
        )
        raise

    writer.write(
        ArtifactProducedPayload(
            # Attribution travels on the envelope's target, which names the
            # attempt, so it is not repeated as producer_attempt_id here.
            artifact_ref=ArtifactRef(id=ArtifactId.generate(), kind="model", uri=str(target))
        )
    )


def require_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set; a worker runs under a runtime that provides it, "
            f"and without it there is no config to run or channel to report on"
        )
    return value


def failure_reason(exc: BaseException) -> str:
    """A ``Name``-shaped reason: the exception's type, lower-kebab-cased."""
    name = type(exc).__name__
    kebab = "".join(f"-{c.lower()}" if c.isupper() else c for c in name).lstrip("-")
    return kebab or "error"
