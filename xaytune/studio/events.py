from __future__ import annotations

import dataclasses
import queue
import threading
import time
from typing import Any

from xaytune.trainer.callbacks import CallbackManager, TrainState


@dataclasses.dataclass
class TrainingEvent:
    event_type: str
    job_id: str
    timestamp: float
    data: dict[str, Any]


class EventBus:
    def __init__(self, maxsize: int = 1000) -> None:
        self._maxsize = maxsize
        self._subscribers: list[tuple[queue.Queue[TrainingEvent], str | None]] = []
        self._lock = threading.Lock()

    def subscribe(self, job_id: str | None = None) -> queue.Queue[TrainingEvent]:
        q: queue.Queue[TrainingEvent] = queue.Queue(maxsize=self._maxsize)
        with self._lock:
            self._subscribers.append((q, job_id))
        return q

    def unsubscribe(self, q: queue.Queue[TrainingEvent]) -> None:
        with self._lock:
            self._subscribers = [(sq, jid) for sq, jid in self._subscribers if sq is not q]

    def publish(self, event: TrainingEvent) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for q, job_filter in subscribers:
            if job_filter is not None and job_filter != event.job_id:
                continue
            try:
                q.put_nowait(event)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass


def register_event_callbacks(
    *,
    callback_manager: CallbackManager,
    event_bus: EventBus,
    job_id: str,
    events: tuple[str, ...] = ("train_start", "step_end", "eval_end", "train_end"),
) -> None:
    for event_name in events:

        def _make_cb(evt: str):
            def _cb(state: TrainState) -> None:
                event_bus.publish(
                    TrainingEvent(
                        event_type=evt,
                        job_id=job_id,
                        timestamp=time.time(),
                        data=state.to_dict(),
                    )
                )

            return _cb

        callback_manager.on(event_name)(_make_cb(event_name))
