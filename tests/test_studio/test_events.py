import time

from trainlib.studio.events import EventBus, TrainingEvent, register_event_callbacks
from trainlib.trainer.callbacks import CallbackManager, TrainState


def _make_event(job_id="job-1", event_type="step_end"):
    return TrainingEvent(
        event_type=event_type,
        job_id=job_id,
        timestamp=time.time(),
        data={"global_step": 1},
    )


class TestTrainingEvent:
    def test_fields(self):
        evt = _make_event()
        assert evt.event_type == "step_end"
        assert evt.job_id == "job-1"
        assert isinstance(evt.timestamp, float)
        assert evt.data == {"global_step": 1}


class TestEventBus:
    def test_publish_delivers_to_subscriber(self):
        bus = EventBus()
        q = bus.subscribe()
        evt = _make_event()
        bus.publish(evt)
        received = q.get_nowait()
        assert received is evt

    def test_multiple_subscribers(self):
        bus = EventBus()
        q1 = bus.subscribe()
        q2 = bus.subscribe()
        evt = _make_event()
        bus.publish(evt)
        assert q1.get_nowait() is evt
        assert q2.get_nowait() is evt

    def test_job_id_filtering(self):
        bus = EventBus()
        q_a = bus.subscribe(job_id="job-a")
        q_b = bus.subscribe(job_id="job-b")
        bus.publish(_make_event(job_id="job-a"))
        assert q_b.qsize() == 0
        assert q_a.get_nowait().job_id == "job-a"
        assert q_b.empty()

    def test_unsubscribe(self):
        bus = EventBus()
        q = bus.subscribe()
        bus.unsubscribe(q)
        bus.publish(_make_event())
        assert q.empty()

    def test_bounded_queue_drops_oldest(self):
        bus = EventBus(maxsize=2)
        q = bus.subscribe()
        bus.publish(_make_event(event_type="evt1"))
        bus.publish(_make_event(event_type="evt2"))
        bus.publish(_make_event(event_type="evt3"))
        items = []
        while not q.empty():
            items.append(q.get_nowait())
        assert len(items) == 2
        assert items[-1].event_type == "evt3"

    def test_subscribe_all_jobs(self):
        bus = EventBus()
        q = bus.subscribe(job_id=None)
        bus.publish(_make_event(job_id="job-a"))
        bus.publish(_make_event(job_id="job-b"))
        assert q.qsize() == 2


class TestRegisterEventCallbacks:
    def test_fires_on_step_end(self):
        bus = EventBus()
        cm = CallbackManager()
        q = bus.subscribe()

        register_event_callbacks(
            callback_manager=cm, event_bus=bus, job_id="test-job"
        )

        state = TrainState(global_step=5, metrics={"loss": 0.5})
        cm.fire("step_end", state)

        evt = q.get_nowait()
        assert evt.event_type == "step_end"
        assert evt.job_id == "test-job"
        assert evt.data["global_step"] == 5

    def test_fires_on_train_start(self):
        bus = EventBus()
        cm = CallbackManager()
        q = bus.subscribe()

        register_event_callbacks(
            callback_manager=cm, event_bus=bus, job_id="j1"
        )

        cm.fire("train_start", TrainState())
        evt = q.get_nowait()
        assert evt.event_type == "train_start"
