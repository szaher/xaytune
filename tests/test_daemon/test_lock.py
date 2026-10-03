"""The state database's singleton lock (ADR-004 §5), within one process.

flock locks belong to an open file description, so two locks opened
separately conflict even in one process -- which lets the contract be checked
here; :mod:`.test_daemon_process` checks it across processes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xaytune.daemon import DaemonAlreadyRunningError, StateDatabaseLock, lock_path


def test_a_second_holder_is_refused_and_told_who_holds_it(tmp_path: Path) -> None:
    first = StateDatabaseLock(tmp_path / "state.db")
    first.acquire({"pid": 1, "instance_id": "first"})
    try:
        with pytest.raises(DaemonAlreadyRunningError) as refused:
            StateDatabaseLock(tmp_path / "state.db").acquire({"instance_id": "second"})
        assert refused.value.holder == {"pid": 1, "instance_id": "first"}
        assert StateDatabaseLock.holder(first.path) == {"pid": 1, "instance_id": "first"}, (
            "a refused acquire does not erase the holder's diagnostics"
        )
    finally:
        first.release()


def test_aliases_of_one_database_share_one_lock(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "sub")
    assert lock_path(tmp_path / "link" / "state.db") == lock_path(tmp_path / "sub" / "state.db")
    assert lock_path(tmp_path / "sub" / ".." / "state.db") == lock_path(tmp_path / "state.db")


def test_release_frees_it_and_is_idempotent(tmp_path: Path) -> None:
    lock = StateDatabaseLock(tmp_path / "state.db")
    lock.acquire({"pid": 1})
    lock.release()
    lock.release()
    assert not lock.held
    assert StateDatabaseLock.holder(lock.path) is None
    again = StateDatabaseLock(tmp_path / "state.db")
    again.acquire({"pid": 2})
    again.release()


def test_an_in_memory_database_has_nothing_to_lock() -> None:
    with pytest.raises(ValueError, match="in-memory"):
        StateDatabaseLock(":memory:")


def test_the_entrypoint_refuses_an_unlockable_platform_before_anything_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Before the config, the database or the event loop's signal handlers."""
    import xaytune.daemon.lock as lock_module
    from xaytune.daemon.__main__ import EXIT_UNSUPPORTED_PLATFORM, main

    monkeypatch.setattr(lock_module, "fcntl", None)
    status = main(
        ["--state", str(tmp_path / "state.db"), "--config", "nowhere.at_all:create_config"]
    )
    assert status == EXIT_UNSUPPORTED_PLATFORM
    assert "fcntl" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []
