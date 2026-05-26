import json
import time
from unittest.mock import MagicMock, patch

import torch

from xaytune.config.schema import DataConfig, ModelConfig, TrainConfig, TrainerConfig
from xaytune.studio.events import EventBus
from xaytune.studio.jobs import JobInfo, JobManager, JobStatus
from xaytune.trainer.callbacks import CallbackManager
from xaytune.trainer.loop import Trainer


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

    dataloader = [{"input_ids": torch.tensor([i]), "labels": torch.tensor([i])} for i in range(5)]

    from xaytune.recipes.base import TrainingComponents

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
    @patch("xaytune.studio.jobs.setup_training", side_effect=_mock_setup)
    def test_submit_returns_job_id(self, mock_setup):
        manager = JobManager()
        job_id = manager.submit(_make_config())
        assert isinstance(job_id, str)
        assert len(job_id) > 0

    @patch("xaytune.studio.jobs.setup_training", side_effect=_mock_setup)
    def test_job_completes(self, mock_setup):
        manager = JobManager()
        job_id = manager.submit(_make_config())
        job = _wait_for_status(manager, job_id, JobStatus.COMPLETED)
        assert job.status == JobStatus.COMPLETED

    @patch("xaytune.studio.jobs.setup_training", side_effect=_mock_setup)
    def test_job_captures_final_state(self, mock_setup):
        manager = JobManager()
        job_id = manager.submit(_make_config())
        job = _wait_for_status(manager, job_id, JobStatus.COMPLETED)
        assert job.state is not None
        assert "global_step" in job.state
        assert job.state["global_step"] == 3

    @patch(
        "xaytune.studio.jobs.setup_training",
        side_effect=RuntimeError("model load failed"),
    )
    def test_job_failure_captured(self, mock_setup):
        manager = JobManager()
        job_id = manager.submit(_make_config())
        job = _wait_for_status(manager, job_id, JobStatus.FAILED)
        assert job.status == JobStatus.FAILED
        assert "model load failed" in job.error

    @patch("xaytune.studio.jobs.setup_training", side_effect=_mock_setup)
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

    @patch("xaytune.studio.jobs.setup_training", side_effect=_mock_setup)
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

    @patch("xaytune.studio.jobs.setup_training", side_effect=_mock_setup)
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

    @patch("xaytune.studio.jobs.setup_training", side_effect=_mock_setup)
    def test_concurrent_jobs(self, mock_setup):
        manager = JobManager()
        id1 = manager.submit(_make_config())
        id2 = manager.submit(_make_config())
        job1 = _wait_for_status(manager, id1, JobStatus.COMPLETED)
        job2 = _wait_for_status(manager, id2, JobStatus.COMPLETED)
        assert job1.status == JobStatus.COMPLETED
        assert job2.status == JobStatus.COMPLETED


class TestJobInfoFromDict:
    def test_round_trip(self):
        job = JobInfo(
            job_id="test-123",
            status=JobStatus.COMPLETED,
            recipe="finetune",
            created_at=1000.0,
            started_at=1001.0,
            completed_at=1010.0,
            error=None,
            state={"global_step": 5, "metrics": {"loss": 0.3}},
            tags=["best", "v1"],
        )
        d = job.to_dict()
        restored = JobInfo.from_dict(d, metrics_history=[{"step": 1, "loss": 0.5}])
        assert restored.job_id == "test-123"
        assert restored.status == JobStatus.COMPLETED
        assert restored.recipe == "finetune"
        assert restored.created_at == 1000.0
        assert restored.started_at == 1001.0
        assert restored.completed_at == 1010.0
        assert restored.tags == ["best", "v1"]
        assert restored.state == {"global_step": 5, "metrics": {"loss": 0.3}}
        assert len(restored.metrics_history) == 1

    def test_minimal_dict(self):
        d = {
            "job_id": "x",
            "status": "failed",
            "recipe": "align",
            "created_at": 0.0,
        }
        restored = JobInfo.from_dict(d)
        assert restored.status == JobStatus.FAILED
        assert restored.tags == []
        assert restored.metrics_history == []
        assert restored.state is None


class TestJobPersistence:
    def test_persist_dir_none_disables_persistence(self):
        mgr = JobManager(persist_dir=None)
        assert mgr._persist_dir is None

    def test_save_and_load_metadata(self, tmp_path):
        mgr = JobManager(persist_dir=tmp_path)
        job = JobInfo(
            job_id="persist-1",
            status=JobStatus.COMPLETED,
            recipe="finetune",
            created_at=1000.0,
            completed_at=1010.0,
            state={"global_step": 10},
            tags=["run1"],
        )
        mgr._jobs["persist-1"] = job
        mgr._save_metadata("persist-1")

        meta_path = tmp_path / "persist-1" / "metadata.json"
        assert meta_path.exists()
        d = json.loads(meta_path.read_text())
        assert d["job_id"] == "persist-1"
        assert d["status"] == "completed"
        assert d["tags"] == ["run1"]

        mgr2 = JobManager(persist_dir=tmp_path)
        jobs = mgr2.list_jobs()
        assert len(jobs) == 1
        assert jobs[0].job_id == "persist-1"
        assert jobs[0].tags == ["run1"]

    def test_metrics_jsonl_written(self, tmp_path):
        mgr = JobManager(persist_dir=tmp_path)
        job = JobInfo(
            job_id="metrics-1",
            status=JobStatus.RUNNING,
            recipe="finetune",
            created_at=1000.0,
        )
        mgr._jobs["metrics-1"] = job

        entries = [
            {"step": 1, "loss": 0.8, "timestamp": 1001.0},
            {"step": 2, "loss": 0.6, "timestamp": 1002.0},
            {"step": 3, "loss": 0.4, "timestamp": 1003.0},
        ]
        for e in entries:
            mgr._append_metrics("metrics-1", e)

        metrics_path = tmp_path / "metrics-1" / "metrics.jsonl"
        assert metrics_path.exists()
        lines = metrics_path.read_text().strip().splitlines()
        assert len(lines) == 3
        assert json.loads(lines[0])["step"] == 1
        assert json.loads(lines[2])["loss"] == 0.4

    def test_logs_written_on_save(self, tmp_path):
        mgr = JobManager(persist_dir=tmp_path)
        job = JobInfo(
            job_id="logs-1",
            status=JobStatus.COMPLETED,
            recipe="finetune",
            created_at=1000.0,
        )
        for i in range(10):
            job.log_buffer.append(f"log line {i}")
        mgr._jobs["logs-1"] = job
        mgr._save_logs("logs-1")

        logs_path = tmp_path / "logs-1" / "logs.txt"
        assert logs_path.exists()
        content = logs_path.read_text()
        assert "log line 0" in content
        assert "log line 9" in content

    def test_stale_running_jobs_not_loaded(self, tmp_path):
        job_dir = tmp_path / "stale-1"
        job_dir.mkdir()
        (job_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "job_id": "stale-1",
                    "status": "running",
                    "recipe": "finetune",
                    "created_at": 1000.0,
                }
            )
        )

        mgr = JobManager(persist_dir=tmp_path)
        assert len(mgr.list_jobs()) == 0

    def test_tags_persisted(self, tmp_path):
        mgr = JobManager(persist_dir=tmp_path)
        job = JobInfo(
            job_id="tag-1",
            status=JobStatus.COMPLETED,
            recipe="finetune",
            created_at=1000.0,
        )
        mgr._jobs["tag-1"] = job
        mgr._save_metadata("tag-1")

        mgr.add_tag("tag-1", "experiment-a")
        mgr.add_tag("tag-1", "best")

        mgr2 = JobManager(persist_dir=tmp_path)
        restored = mgr2.get_status("tag-1")
        assert "experiment-a" in restored.tags
        assert "best" in restored.tags

    def test_get_logs_from_disk(self, tmp_path):
        mgr = JobManager(persist_dir=tmp_path)
        job = JobInfo(
            job_id="disklog-1",
            status=JobStatus.COMPLETED,
            recipe="finetune",
            created_at=1000.0,
        )
        job.log_buffer.append("live log line")
        mgr._jobs["disklog-1"] = job
        mgr._save_logs("disklog-1")

        mgr2 = JobManager(persist_dir=tmp_path)
        job2 = JobInfo(
            job_id="disklog-1",
            status=JobStatus.COMPLETED,
            recipe="finetune",
            created_at=1000.0,
        )
        mgr2._jobs["disklog-1"] = job2
        logs = mgr2.get_logs("disklog-1")
        assert "live log line" in logs

    def test_metrics_loaded_on_startup(self, tmp_path):
        job_dir = tmp_path / "mload-1"
        job_dir.mkdir()
        (job_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "job_id": "mload-1",
                    "status": "completed",
                    "recipe": "finetune",
                    "created_at": 1000.0,
                }
            )
        )
        with (job_dir / "metrics.jsonl").open("w") as f:
            f.write(json.dumps({"step": 1, "loss": 0.9}) + "\n")
            f.write(json.dumps({"step": 2, "loss": 0.7}) + "\n")

        mgr = JobManager(persist_dir=tmp_path)
        jobs = mgr.list_jobs()
        assert len(jobs) == 1
        assert len(jobs[0].metrics_history) == 2
        assert jobs[0].metrics_history[0]["loss"] == 0.9

    def test_malformed_metrics_skipped(self, tmp_path):
        job_dir = tmp_path / "bad-1"
        job_dir.mkdir()
        (job_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "job_id": "bad-1",
                    "status": "completed",
                    "recipe": "finetune",
                    "created_at": 1000.0,
                }
            )
        )
        with (job_dir / "metrics.jsonl").open("w") as f:
            f.write(json.dumps({"step": 1, "loss": 0.5}) + "\n")
            f.write("not valid json\n")
            f.write(json.dumps({"step": 3, "loss": 0.3}) + "\n")

        mgr = JobManager(persist_dir=tmp_path)
        jobs = mgr.list_jobs()
        assert len(jobs[0].metrics_history) == 2
