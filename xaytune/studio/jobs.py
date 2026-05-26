from __future__ import annotations

import collections
import dataclasses
import enum
import json
import logging
import threading
import time
import traceback as tb_mod
import uuid
from pathlib import Path
from typing import Any

from xaytune.config.schema import TrainConfig
from xaytune.recipes.base import setup_training
from xaytune.studio.events import EventBus, register_event_callbacks
from xaytune.trainer.callbacks import CallbackManager, TrainState


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LogBuffer:
    """Thread-safe ring buffer for training log lines."""

    def __init__(self, maxlen: int = 5000) -> None:
        self._buffer: collections.deque[str] = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, line: str) -> None:
        with self._lock:
            self._buffer.append(line)

    def get_all(self) -> list[str]:
        with self._lock:
            return list(self._buffer)

    def get_since(self, offset: int) -> list[str]:
        with self._lock:
            buf = list(self._buffer)
        return buf[offset:]

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


class _LogBufferHandler(logging.Handler):
    """Logging handler that writes records to a LogBuffer."""

    def __init__(self, log_buffer: LogBuffer) -> None:
        super().__init__()
        self._log_buffer = log_buffer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._log_buffer.append(msg)
        except Exception:
            self.handleError(record)


@dataclasses.dataclass
class JobInfo:
    job_id: str
    status: JobStatus
    recipe: str
    created_at: float
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    state: dict[str, Any] | None = None
    tags: list[str] = dataclasses.field(default_factory=list)
    log_buffer: LogBuffer = dataclasses.field(default_factory=LogBuffer)
    metrics_history: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "recipe": self.recipe,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "state": self.state,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        metrics_history: list[dict[str, Any]] | None = None,
    ) -> JobInfo:
        return cls(
            job_id=d["job_id"],
            status=JobStatus(d["status"]),
            recipe=d["recipe"],
            created_at=d["created_at"],
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
            error=d.get("error"),
            state=d.get("state"),
            tags=d.get("tags", []),
            metrics_history=metrics_history or [],
        )


class JobManager:
    def __init__(
        self,
        event_bus: EventBus | None = None,
        persist_dir: Path | str | None = None,
    ) -> None:
        self._jobs: dict[str, JobInfo] = {}
        self._event_bus = event_bus or EventBus()
        self._lock = threading.Lock()
        if persist_dir is not None:
            self._persist_dir: Path | None = Path(persist_dir).expanduser()
            self._persist_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._persist_dir = None
        self._load_persisted_jobs()

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    def submit(
        self,
        config: TrainConfig,
        **kwargs: Any,
    ) -> str:
        job_id = str(uuid.uuid4())
        job = JobInfo(
            job_id=job_id,
            status=JobStatus.PENDING,
            recipe=config.recipe,
            created_at=time.time(),
        )
        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, config, kwargs),
            daemon=True,
        )
        thread.start()
        return job_id

    def get_status(self, job_id: str) -> JobInfo:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"Unknown job: {job_id}")
            return self._jobs[job_id]

    def list_jobs(self) -> list[JobInfo]:
        with self._lock:
            return list(self._jobs.values())

    def cancel(self, job_id: str) -> None:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"Unknown job: {job_id}")
            self._jobs[job_id].status = JobStatus.CANCELLED
            self._save_metadata(job_id)

    def add_tag(self, job_id: str, tag: str) -> None:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"Unknown job: {job_id}")
            if tag not in self._jobs[job_id].tags:
                self._jobs[job_id].tags.append(tag)
            self._save_metadata(job_id)

    def remove_tag(self, job_id: str, tag: str) -> None:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"Unknown job: {job_id}")
            try:
                self._jobs[job_id].tags.remove(tag)
            except ValueError:
                pass
            self._save_metadata(job_id)

    def get_metrics_history(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"Unknown job: {job_id}")
            return list(self._jobs[job_id].metrics_history)

    def _save_metadata(self, job_id: str) -> None:
        if self._persist_dir is None:
            return
        job = self._jobs[job_id]
        job_dir = self._persist_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "metadata.json").write_text(json.dumps(job.to_dict(), indent=2))

    def _append_metrics(self, job_id: str, entry: dict[str, Any]) -> None:
        if self._persist_dir is None:
            return
        job_dir = self._persist_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        with (job_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def _save_logs(self, job_id: str) -> None:
        if self._persist_dir is None:
            return
        job = self._jobs[job_id]
        lines = job.log_buffer.get_all()[-2000:]
        job_dir = self._persist_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "logs.txt").write_text("\n".join(lines))

    def _load_persisted_jobs(self) -> None:
        if self._persist_dir is None or not self._persist_dir.exists():
            return
        for job_dir in self._persist_dir.iterdir():
            if not job_dir.is_dir():
                continue
            meta_path = job_dir / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                d = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            status = d.get("status", "")
            if status in ("running", "pending"):
                continue
            if d.get("job_id") in self._jobs:
                continue
            metrics: list[dict[str, Any]] = []
            metrics_path = job_dir / "metrics.jsonl"
            if metrics_path.exists():
                for line in metrics_path.read_text().splitlines():
                    try:
                        metrics.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            self._jobs[d["job_id"]] = JobInfo.from_dict(d, metrics_history=metrics)

    def get_logs(self, job_id: str) -> str:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"Unknown job: {job_id}")
            job = self._jobs[job_id]
        lines = job.log_buffer.get_all()
        if lines:
            return "\n".join(lines)
        if self._persist_dir is not None:
            logs_path = self._persist_dir / job_id / "logs.txt"
            if logs_path.exists():
                return logs_path.read_text()
        return ""

    def _run_job(
        self,
        job_id: str,
        config: TrainConfig,
        kwargs: dict[str, Any],
    ) -> None:
        job = self._jobs[job_id]
        cb = CallbackManager()

        last_step_time: list[float] = []

        @cb.on("step_end")
        def _track_state(state: TrainState) -> None:
            now = time.time()
            with self._lock:
                job.state = state.to_dict()
                if job.status == JobStatus.CANCELLED:
                    state.stop_training()

                metrics = state.to_dict().get("metrics", {})
                entry: dict[str, Any] = {
                    "step": state.global_step,
                    "timestamp": now,
                    **metrics,
                }
                if last_step_time:
                    dt = now - last_step_time[0]
                    if dt > 0:
                        entry["samples_per_sec"] = config.trainer.batch_size / dt

                job.metrics_history.append(entry)
                self._append_metrics(job_id, entry)

            last_step_time.clear()
            last_step_time.append(now)

        register_event_callbacks(
            callback_manager=cb,
            event_bus=self._event_bus,
            job_id=job_id,
        )

        handler = _LogBufferHandler(job.log_buffer)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        try:
            with self._lock:
                job.status = JobStatus.RUNNING
                job.started_at = time.time()
                self._save_metadata(job_id)

            last_step_time.append(time.time())

            components = setup_training(config, callback_manager=cb)
            final_state = components.trainer.train(
                model=components.model,
                train_dataloader=components.train_dataloader,
                resume_state=components.resume_state,
            )

            with self._lock:
                job.state = final_state.to_dict()
                if job.status == JobStatus.CANCELLED:
                    pass
                else:
                    job.status = JobStatus.COMPLETED
                job.completed_at = time.time()
                self._save_metadata(job_id)
                self._save_logs(job_id)
        except Exception as exc:
            with self._lock:
                job.status = JobStatus.FAILED
                job.error = "".join(tb_mod.format_exception(type(exc), exc, exc.__traceback__))
                job.completed_at = time.time()
                self._save_metadata(job_id)
                self._save_logs(job_id)
        finally:
            root_logger.removeHandler(handler)
