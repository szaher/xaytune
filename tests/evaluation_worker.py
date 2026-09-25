"""The scripted evaluation worker, run by LocalRuntime as ``python -m tests.evaluation_worker``.

See ``tests/evaluation_fixtures.py``. It reports what an evaluation reports --
started, a streamed metric, and a completion carrying the final result --
under telemetry v1alpha3, and otherwise does what its config says.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from xaytune.core.domain.evaluation import MetricResult
from xaytune.core.ids import ArtifactId
from xaytune.core.refs import ArtifactRef
from xaytune.core.telemetry import (
    EvaluationCompletedPayload,
    EvaluationFailedPayload,
    EvaluationStartedPayload,
    MetricObservedPayload,
)
from xaytune.runtimes.worker import WORKER_CONFIG_PATH_ENV, ObservationWriter

_HOLD_LIMIT_SECONDS = 300


def main() -> int:
    config = json.loads(Path(os.environ[WORKER_CONFIG_PATH_ENV]).read_text(encoding="utf-8"))
    writer = ObservationWriter.from_environment()
    assert writer is not None, "run by a runtime, which provides the observation channel"
    value = float(config.get("value", 0.75))

    writer.write(EvaluationStartedPayload())
    writer.write(MetricObservedPayload(name="accuracy", value=value / 2))

    hold = config.get("hold")
    if hold is not None:
        deadline = time.monotonic() + _HOLD_LIMIT_SECONDS
        while not Path(hold).exists():
            if time.monotonic() > deadline:
                return 3
            time.sleep(0.05)

    mode = config.get("mode", "complete")
    if mode == "fail":
        writer.write(EvaluationFailedPayload(reason="scripted-failure"))
        return 1
    if mode == "no-result":
        return 0

    report = Path(config["output_uri"]) / "report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"subject": config["subject_uri"], "accuracy": value}))
    writer.write(
        EvaluationCompletedPayload(
            metrics=(
                MetricResult(
                    name="accuracy",
                    value=value,
                    sample_count=4,
                    evaluator_name="scripted",
                    evaluator_version=config["evaluator_version"],
                    seed=config["seed"],
                ),
            ),
            result_ref=ArtifactRef(
                id=ArtifactId.generate(), kind="evaluation_report", uri=str(report)
            ),
        )
    )
    # Reported a result, then failed on the way out: the exit is the outcome.
    return 1 if mode == "complete-then-fail" else 0


if __name__ == "__main__":
    sys.exit(main())
