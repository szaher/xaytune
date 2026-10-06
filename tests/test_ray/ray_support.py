"""Two ways to give RayJobsRuntime a Ray: a faithful fake, and a real local head.

``ProcessJobs`` implements :class:`~xaytune.ray.submission.RaySubmissionBackend` the way
Ray's job manager behaves where the runtime depends on it -- one job per
submission id, the entrypoint run as a shell command in its own process
group, ``PENDING``/``RUNNING``/terminal states, ``stop`` -- and can be made
unreachable or forgetful. It runs real supervisors and real workers; only the
cluster is simulated, so the matrix exercises every path without Ray.

``ray_head`` starts a real ``ray start --head --block`` on free ports and owns
it: teardown stops that process tree and nothing else -- never ``ray stop``,
which would take down any other Ray on the machine. It skips without Ray,
unless ``XAYTUNE_REQUIRE_RAY`` is set (the ``ray`` CI job), when it fails
instead.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from xaytune.ray.runtime.train import TRAIN_DRIVER
from xaytune.ray.submission import RayJob, RayJobStatus, RayUnavailableError

FAKE_TRAIN_DRIVER = "tests.test_ray.fake_train_driver"
_REPOSITORY = Path(__file__).resolve().parents[2]


@dataclass
class _Job:
    entrypoint: str
    metadata: dict[str, str]
    resources: dict[str, Any]
    runtime_env: dict[str, Any]
    held: bool
    process: subprocess.Popen[bytes] | None = None
    stopped: bool = False
    timed_out: bool = False
    submitted_at: float = field(default_factory=time.monotonic)


class ProcessJobs:
    """A fake Ray job manager that runs entrypoints as real processes."""

    def __init__(self, *, hold: bool = False) -> None:
        """*hold* keeps every new job pending until :meth:`release`."""
        self.jobs: dict[str, _Job] = {}
        self.submissions = 0
        self.hold = hold
        self.unreachable = False
        self.stop_ignores_pending = False

    def submit(
        self,
        submission_id: str,
        entrypoint: str,
        *,
        metadata: Mapping[str, str],
        resources: Mapping[str, Any],
        runtime_env: Mapping[str, Any],
    ) -> None:
        self._reachable()
        self.submissions += 1
        if submission_id in self.jobs:
            raise RayUnavailableError(f"Ray did not accept job {submission_id}: RuntimeError")
        job = _Job(entrypoint, dict(metadata), dict(resources), dict(runtime_env), held=self.hold)
        self.jobs[submission_id] = job
        if not job.held:
            self._start(job)

    def release(self, submission_id: str) -> None:
        """Let a held job start, as a cluster does once resources free up."""
        job = self.jobs[submission_id]
        job.held = False
        if not job.stopped:
            self._start(job)

    def time_out(self, submission_id: str) -> None:
        """End a job that never started as failed, as Ray's start timeout does."""
        job = self.jobs[submission_id]
        assert job.process is None
        job.timed_out = True

    def forget(self, submission_id: str) -> None:
        """Lose the job, as a restarted cluster without persistent GCS does."""
        del self.jobs[submission_id]

    def info(self, submission_id: str) -> RayJob | None:
        self._reachable()
        job = self.jobs.get(submission_id)
        if job is None:
            return None
        return RayJob(
            submission_id=submission_id,
            status=self._status(job),
            metadata=job.metadata,
            exit_code=None if job.process is None else job.process.poll(),
        )

    def stop(self, submission_id: str) -> None:
        self._reachable()
        job = self.jobs.get(submission_id)
        if job is None or self._status(job) not in ("PENDING", "RUNNING"):
            return
        if job.process is None and self.stop_ignores_pending:
            return  # what Ray does for a job it cannot schedule
        job.stopped = True
        if job.process is not None and job.process.poll() is None:
            try:
                os.killpg(job.process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):  # pragma: no cover
                pass

    def close(self) -> None:
        for job in self.jobs.values():
            if job.process is not None and job.process.poll() is None:
                try:
                    os.killpg(job.process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):  # pragma: no cover
                    pass
                job.process.wait()

    def _start(self, job: _Job) -> None:
        job.process = subprocess.Popen(  # noqa: S602 -- Ray runs entrypoints through a shell too
            # This "cluster's image" has no Ray, so its Ray Train driver is the
            # one whose worker group is simulated (tests/test_ray/fake_train_driver).
            job.entrypoint.replace(TRAIN_DRIVER, FAKE_TRAIN_DRIVER),
            cwd=str(_REPOSITORY),
            shell=True,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # The entrypoint's ``python`` is the job environment's; here, this one.
            env={
                **os.environ,
                "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
            },
        )

    def _status(self, job: _Job) -> RayJobStatus:
        if job.process is None:
            if job.timed_out:
                return "FAILED"
            return "STOPPED" if job.stopped else "PENDING"
        code = job.process.poll()
        if code is None:
            return "RUNNING"
        if job.stopped:
            return "STOPPED"
        return "SUCCEEDED" if code == 0 else "FAILED"

    def _reachable(self) -> None:
        if self.unreachable:
            raise RayUnavailableError("cannot reach Ray at http://fake: ConnectionError")


def _free_ports(count: int) -> list[int]:
    """*count* distinct ports free now; held open together so none repeats."""
    probes = [socket.socket() for _ in range(count)]
    try:
        for probe in probes:
            probe.bind(("127.0.0.1", 0))
        return [int(probe.getsockname()[1]) for probe in probes]
    finally:
        for probe in probes:
            probe.close()


@dataclass(frozen=True)
class RayHead:
    address: str


@pytest.fixture(scope="session")
def ray_head() -> Iterator[RayHead]:
    """A real local Ray head with the job-submission API, owned by this session."""
    try:
        from ray.job_submission import JobSubmissionClient
    except ImportError:
        if os.environ.get("XAYTUNE_REQUIRE_RAY"):
            raise
        pytest.skip("Ray is not installed (pip install xaytune[ray])")

    # Ray's Unix sockets live under the temp dir, and their paths may not
    # exceed ~104 bytes, so this one is short and outside pytest's tmp_path.
    temp = tempfile.mkdtemp(prefix="xr", dir="/tmp")
    # Every port this head listens on, not only the two a client dials: Ray's
    # defaults (the dashboard agent's 52365, the client server's 10001, ...)
    # belong to whichever Ray on the machine took them first, and a job sent
    # to this head's dashboard must reach this head's agent.
    ports = _free_ports(9)
    port, dashboard = ports[0], ports[1]
    address = f"http://127.0.0.1:{dashboard}"
    environment = {**os.environ, "RAY_USAGE_STATS_ENABLED": "0"}
    # The job's entrypoint shell must find this interpreter's xaytune.
    environment["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{environment['PATH']}"
    # A job Ray cannot schedule fails after this long, rather than 15 minutes.
    environment["RAY_JOB_START_TIMEOUT_SECONDS"] = "15"
    # Under ``uv run`` Ray would otherwise give every job a runtime_env of
    # its own -- this repository uploaded as working_dir, ``uv run python`` as
    # the interpreter -- in place of the one the runtime asked for.
    environment["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    log = open(Path(temp) / "head.log", "wb")  # noqa: SIM115 -- closed at teardown
    head = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "ray.scripts.scripts",
            "start",
            "--head",
            "--block",
            f"--port={port}",
            f"--dashboard-port={dashboard}",
            f"--dashboard-agent-listen-port={ports[2]}",
            f"--dashboard-agent-grpc-port={ports[3]}",
            f"--ray-client-server-port={ports[4]}",
            f"--metrics-export-port={ports[5]}",
            f"--node-manager-port={ports[6]}",
            f"--object-manager-port={ports[7]}",
            f"--runtime-env-agent-port={ports[8]}",
            "--dashboard-host=127.0.0.1",
            "--include-dashboard=true",
            "--num-cpus=4",
            "--num-gpus=0",
            f"--temp-dir={temp}",
            "--disable-usage-stats",
        ],
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        _wait_for_jobs_api(JobSubmissionClient, address, head, Path(temp) / "head.log")
        yield RayHead(address=address)
    finally:
        _stop_owned(head, temp)
        log.close()
        shutil.rmtree(temp, ignore_errors=True)


def _wait_for_jobs_api(client: Any, address: str, head: subprocess.Popen[bytes], log: Path) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if head.poll() is not None:
            raise RuntimeError(f"ray start exited {head.returncode}:\n{log.read_text()[-4000:]}")
        try:
            client(address).list_jobs()
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"Ray's job API never answered at {address}:\n{log.read_text()[-4000:]}")


def _stop_owned(head: subprocess.Popen[bytes], temp: str) -> None:
    """Stop the head this fixture started, and only it.

    Owned means: a descendant of the ``ray start --block`` process, or a
    process naming this session's private temp dir (one Ray reparented).
    Another Ray on the machine is neither, and is left alone.
    """
    import psutil

    owned: dict[int, psutil.Process] = {}
    try:
        for process in psutil.Process(head.pid).children(recursive=True):
            owned[process.pid] = process
    except psutil.NoSuchProcess:
        pass
    for process in psutil.process_iter(["cmdline"]):
        try:
            if any(temp in part for part in process.info["cmdline"] or ()):
                owned[process.pid] = process
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    owned.pop(head.pid, None)

    head.send_signal(signal.SIGTERM)  # --block stops its own processes on SIGTERM
    try:
        head.wait(timeout=30)
    except subprocess.TimeoutExpired:
        head.kill()
        head.wait()
    _, alive = psutil.wait_procs(list(owned.values()), timeout=10)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=10)
