from __future__ import annotations

import dataclasses
import enum
import threading
import time
import uuid
from typing import Any

from trainlib.config.schema import TrainConfig
from trainlib.recipes.base import setup_training
from trainlib.studio.events import EventBus, register_event_callbacks
from trainlib.trainer.callbacks import CallbackManager, TrainState


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


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
        }


class JobManager:
    def __init__(self, event_bus: EventBus | None = None) -> None:
        self._jobs: dict[str, JobInfo] = {}
        self._event_bus = event_bus or EventBus()
        self._lock = threading.Lock()

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

    def _run_job(
        self,
        job_id: str,
        config: TrainConfig,
        kwargs: dict[str, Any],
    ) -> None:
        job = self._jobs[job_id]
        cb = CallbackManager()

        @cb.on("step_end")
        def _track_state(state: TrainState) -> None:
            with self._lock:
                job.state = state.to_dict()
                if job.status == JobStatus.CANCELLED:
                    state.stop_training()

        register_event_callbacks(
            callback_manager=cb,
            event_bus=self._event_bus,
            job_id=job_id,
        )

        try:
            with self._lock:
                job.status = JobStatus.RUNNING
                job.started_at = time.time()

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
        except Exception as e:
            with self._lock:
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.completed_at = time.time()
