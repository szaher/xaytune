import time
from unittest.mock import MagicMock, patch

import torch

from trainlib.config.schema import DataConfig, ModelConfig, TrainConfig, TrainerConfig
from trainlib.studio.events import EventBus
from trainlib.studio.jobs import JobInfo, JobManager, JobStatus
from trainlib.trainer.callbacks import CallbackManager
from trainlib.trainer.loop import Trainer


def _make_config():
    return TrainConfig(
        recipe="finetune",
        method="full",
        model=ModelConfig(name="test-model"),
        data=DataConfig(path="data.jsonl", format="alpaca"),
        trainer=TrainerConfig(num_epochs=1, max_steps=3),
    )


def _mock_setup(config, callback_manager=None, resume_from=None):
    """Mock setup_training that creates a real Trainer with mock model."""
    cb = callback_manager or CallbackManager()
    trainer = Trainer(config=config.trainer, callback_manager=cb)

    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.loss = torch.tensor(0.5, requires_grad=True)
    mock_model.return_value = mock_output
    mock_model.parameters.return_value = [torch.randn(10, requires_grad=True)]

    dataloader = [
        {"input_ids": torch.tensor([i]), "labels": torch.tensor([i])}
        for i in range(5)
    ]

    from trainlib.recipes.base import TrainingComponents

    return TrainingComponents(
        model=mock_model,
        tokenizer=MagicMock(),
        train_dataloader=dataloader,
        eval_dataloader=None,
        trainer=trainer,
    )


def _wait_for_status(manager, job_id, status, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = manager.get_status(job_id)
        if job.status == status:
            return job
        time.sleep(0.05)
    return manager.get_status(job_id)


class TestJobStatus:
    def test_enum_values(self):
        assert JobStatus.PENDING == "pending"
        assert JobStatus.RUNNING == "running"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"
        assert JobStatus.CANCELLED == "cancelled"


class TestJobInfo:
    def test_to_dict(self):
        job = JobInfo(
            job_id="abc",
            status=JobStatus.COMPLETED,
            recipe="finetune",
            created_at=1000.0,
            completed_at=1010.0,
            state={"global_step": 5},
        )
        d = job.to_dict()
        assert d["job_id"] == "abc"
        assert d["status"] == "completed"
        assert d["state"]["global_step"] == 5


class TestJobManager:
    @patch("trainlib.studio.jobs.setup_training", side_effect=_mock_setup)
    def test_submit_returns_job_id(self, mock_setup):
        manager = JobManager()
        job_id = manager.submit(_make_config())
        assert isinstance(job_id, str)
        assert len(job_id) > 0

    @patch("trainlib.studio.jobs.setup_training", side_effect=_mock_setup)
    def test_job_completes(self, mock_setup):
        manager = JobManager()
        job_id = manager.submit(_make_config())
        job = _wait_for_status(manager, job_id, JobStatus.COMPLETED)
        assert job.status == JobStatus.COMPLETED

    @patch("trainlib.studio.jobs.setup_training", side_effect=_mock_setup)
    def test_job_captures_final_state(self, mock_setup):
        manager = JobManager()
        job_id = manager.submit(_make_config())
        job = _wait_for_status(manager, job_id, JobStatus.COMPLETED)
        assert job.state is not None
        assert "global_step" in job.state
        assert job.state["global_step"] == 3

    @patch(
        "trainlib.studio.jobs.setup_training",
        side_effect=RuntimeError("model load failed"),
    )
    def test_job_failure_captured(self, mock_setup):
        manager = JobManager()
        job_id = manager.submit(_make_config())
        job = _wait_for_status(manager, job_id, JobStatus.FAILED)
        assert job.status == JobStatus.FAILED
        assert "model load failed" in job.error

    @patch("trainlib.studio.jobs.setup_training", side_effect=_mock_setup)
    def test_cancel_stops_training(self, mock_setup):
        config = _make_config()
        config.trainer.max_steps = 10000
        config.trainer.num_epochs = 100

        manager = JobManager()
        job_id = manager.submit(config)
        _wait_for_status(manager, job_id, JobStatus.RUNNING, timeout=2.0)
        manager.cancel(job_id)
        job = _wait_for_status(manager, job_id, JobStatus.CANCELLED, timeout=5.0)
        assert job.status == JobStatus.CANCELLED

    @patch("trainlib.studio.jobs.setup_training", side_effect=_mock_setup)
    def test_list_jobs(self, mock_setup):
        manager = JobManager()
        id1 = manager.submit(_make_config())
        id2 = manager.submit(_make_config())
        jobs = manager.list_jobs()
        job_ids = {j.job_id for j in jobs}
        assert id1 in job_ids
        assert id2 in job_ids

    def test_get_status_unknown_job(self):
        manager = JobManager()
        try:
            manager.get_status("nonexistent")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass

    @patch("trainlib.studio.jobs.setup_training", side_effect=_mock_setup)
    def test_event_bus_receives_events(self, mock_setup):
        bus = EventBus()
        manager = JobManager(event_bus=bus)
        q = bus.subscribe()
        job_id = manager.submit(_make_config())
        _wait_for_status(manager, job_id, JobStatus.COMPLETED)
        events = []
        while not q.empty():
            events.append(q.get_nowait())
        event_types = {e.event_type for e in events}
        assert "train_start" in event_types or "step_end" in event_types
        assert all(e.job_id == job_id for e in events)

    @patch("trainlib.studio.jobs.setup_training", side_effect=_mock_setup)
    def test_concurrent_jobs(self, mock_setup):
        manager = JobManager()
        id1 = manager.submit(_make_config())
        id2 = manager.submit(_make_config())
        job1 = _wait_for_status(manager, id1, JobStatus.COMPLETED)
        job2 = _wait_for_status(manager, id2, JobStatus.COMPLETED)
        assert job1.status == JobStatus.COMPLETED
        assert job2.status == JobStatus.COMPLETED
