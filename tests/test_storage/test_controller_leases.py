"""The durable controller lease and the write fences (ADR-004 §8; PR-028).

```text
acquire        epoch 1 first; every new owner the next epoch -- after a clean
               release as after a crash; a live lease is never taken
renew          the owner, its epoch, unexpired: extends, keeps the epoch
release        expires in place; another owner's lease is left alone
fence          every repository write proves owner + epoch + unexpired inside
               its transaction, or writes nothing
no-live-lease  an embedded host writes only while no daemon holds a live lease
migration 017  one row, never deleted; the epoch keeps or moves on by one
```
"""

from __future__ import annotations

import ast
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.test_storage.conftest import make_experiment
from xaytune.core.refs import Actor
from xaytune.storage import (
    ControllerLease,
    ControllerLeaseFence,
    ControllerLeaseHeldError,
    ControllerLeaseStore,
    ControlPlaneRepository,
    LeaseLostError,
    NoLiveLeaseFence,
)

_ACTOR = Actor(type="system", id="test")
_TTL = timedelta(seconds=30)
_XAYTUNE = Path(__file__).resolve().parents[2] / "xaytune"


class _Clock:
    """A clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture
def leases(connection: sqlite3.Connection, clock: _Clock) -> ControllerLeaseStore:
    return ControllerLeaseStore(connection, clock=clock)


def _acquired(leases: ControllerLeaseStore, controller_id: str) -> ControllerLease:
    lease = leases.acquire(controller_id, _TTL)
    assert lease is not None
    return lease


def _experiments(connection: sqlite3.Connection) -> int:
    return int(connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0])


def _events(connection: sqlite3.Connection) -> int:
    return int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])


# ---- the lease lifecycle -------------------------------------------------------------


def test_the_first_owner_gets_epoch_one(leases: ControllerLeaseStore, clock: _Clock) -> None:
    assert leases.current() is None
    lease = _acquired(leases, "daemon-a")
    assert (lease.controller_id, lease.epoch) == ("daemon-a", 1)
    assert lease.heartbeat_at == clock.now
    assert lease.lease_expires_at == clock.now + _TTL
    assert leases.current() == lease


def test_renewal_extends_the_lease_and_keeps_its_epoch(
    leases: ControllerLeaseStore, clock: _Clock
) -> None:
    lease = _acquired(leases, "daemon-a")
    clock.advance(10)
    renewed = leases.renew(lease, _TTL)
    assert renewed.epoch == lease.epoch == 1
    assert renewed.lease_expires_at == clock.now + _TTL
    assert leases.current() == renewed


def test_a_clean_release_expires_at_once_and_the_next_owner_is_the_next_epoch(
    leases: ControllerLeaseStore,
) -> None:
    first = _acquired(leases, "daemon-a")
    assert leases.release(first) is True
    released = leases.current()
    assert released is not None and released.epoch == 1, "the row stays, with its epoch"
    second = _acquired(leases, "daemon-b")
    assert (second.controller_id, second.epoch) == ("daemon-b", 2)


def test_the_same_controller_id_acquiring_again_is_still_a_new_epoch(
    leases: ControllerLeaseStore,
) -> None:
    """No ABA: a reused id never makes an earlier generation current again."""
    first = _acquired(leases, "daemon-a")
    leases.release(first)
    again = _acquired(leases, "daemon-a")
    assert again.epoch == 2
    with pytest.raises(LeaseLostError):
        leases.renew(first, _TTL)


def test_a_crashed_owners_lease_holds_until_it_expires(
    leases: ControllerLeaseStore, clock: _Clock
) -> None:
    crashed = _acquired(leases, "daemon-a")
    clock.advance(29.9)
    assert leases.acquire("daemon-b", _TTL) is None, "a live lease is never taken"
    assert leases.current() == crashed
    clock.advance(0.1)
    taken = _acquired(leases, "daemon-b")
    assert (taken.controller_id, taken.epoch) == ("daemon-b", crashed.epoch + 1)


def test_a_stale_owner_cannot_renew(leases: ControllerLeaseStore, clock: _Clock) -> None:
    old = _acquired(leases, "daemon-a")
    clock.advance(31)
    new = _acquired(leases, "daemon-b")
    with pytest.raises(LeaseLostError, match="owned by daemon-b at epoch 2"):
        leases.renew(old, _TTL)
    assert leases.current() == new


def test_an_expired_lease_is_not_revived_by_renewing_it(
    leases: ControllerLeaseStore, clock: _Clock
) -> None:
    lease = _acquired(leases, "daemon-a")
    clock.advance(30)
    with pytest.raises(LeaseLostError, match="expired"):
        leases.renew(lease, _TTL)
    assert leases.current() == lease, "nothing written"


def test_a_stale_owner_cannot_release_the_new_owners_lease(
    leases: ControllerLeaseStore, clock: _Clock
) -> None:
    old = _acquired(leases, "daemon-a")
    clock.advance(31)
    new = _acquired(leases, "daemon-b")
    assert leases.release(old) is False
    assert leases.current() == new


@pytest.mark.parametrize("seconds", [0, -1])
def test_a_ttl_must_be_positive(leases: ControllerLeaseStore, seconds: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        leases.acquire("daemon-a", timedelta(seconds=seconds))


# ---- migration 017 -------------------------------------------------------------------


def test_the_lease_row_is_never_deleted(
    leases: ControllerLeaseStore, connection: sqlite3.Connection
) -> None:
    _acquired(leases, "daemon-a")
    with pytest.raises(sqlite3.IntegrityError, match="never deleted"):
        connection.execute("DELETE FROM controller_leases")


def test_the_epoch_keeps_or_moves_on_by_one(
    leases: ControllerLeaseStore, connection: sqlite3.Connection
) -> None:
    _acquired(leases, "daemon-a")
    with pytest.raises(sqlite3.IntegrityError, match="epoch"):
        connection.execute("UPDATE controller_leases SET epoch = 3")
    with pytest.raises(sqlite3.IntegrityError, match="epoch"):
        connection.execute("UPDATE controller_leases SET controller_id = 'daemon-b'")
    with pytest.raises(sqlite3.IntegrityError, match="epoch"):
        connection.execute("UPDATE controller_leases SET epoch = 0")


def test_the_first_lease_is_epoch_one_and_there_is_only_one(
    connection: sqlite3.Connection,
) -> None:
    insert = (
        "INSERT INTO controller_leases (singleton_key, controller_id, epoch, heartbeat_at, "
        "lease_expires_at) VALUES (?, 'daemon-a', ?, 'x', 'x')"
    )
    with pytest.raises(sqlite3.IntegrityError, match="epoch 1"):
        connection.execute(insert, (1, 2))
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(insert, (2, 1))


# ---- the daemon's fence --------------------------------------------------------------


def _fenced(
    connection: sqlite3.Connection, clock: _Clock, lease: ControllerLease, **fence: object
) -> ControlPlaneRepository:
    return ControlPlaneRepository(
        connection,
        fence=ControllerLeaseFence(lease.controller_id, lease.epoch, clock=clock, **fence),  # type: ignore[arg-type]
    )


def test_the_current_epoch_may_write(
    connection: sqlite3.Connection, leases: ControllerLeaseStore, clock: _Clock
) -> None:
    repository = _fenced(connection, clock, _acquired(leases, "daemon-a"))
    repository.create_experiment(make_experiment(), actor=_ACTOR)
    assert _experiments(connection) == 1


@pytest.mark.parametrize("who", ["another controller", "an old epoch", "an expired owner"])
def test_a_writer_that_does_not_own_the_database_writes_nothing(
    connection: sqlite3.Connection, leases: ControllerLeaseStore, clock: _Clock, who: str
) -> None:
    lease = _acquired(leases, "daemon-a")
    if who == "another controller":
        writer = ControllerLease("daemon-x", lease.epoch, clock.now, lease.lease_expires_at)
    elif who == "an old epoch":
        leases.release(lease)
        _acquired(leases, "daemon-a")
        writer = lease
    else:
        clock.advance(30)
        writer = lease
    lost: list[LeaseLostError] = []
    repository = _fenced(connection, clock, writer, on_lost=lost.append)
    before = (_experiments(connection), _events(connection))

    with pytest.raises(LeaseLostError) as refused:
        repository.create_experiment(make_experiment(), actor=_ACTOR)

    assert (_experiments(connection), _events(connection)) == before
    assert not connection.in_transaction, "the refused transaction rolled back"
    assert lost == [refused.value], "the process is told before the error is raised"


def test_a_takeover_between_two_writes_fences_the_second(
    connection: sqlite3.Connection, leases: ControllerLeaseStore, clock: _Clock
) -> None:
    old = _acquired(leases, "daemon-a")
    repository = _fenced(connection, clock, old)
    repository.create_experiment(make_experiment(), actor=_ACTOR)
    clock.advance(31)
    _acquired(leases, "daemon-b")
    with pytest.raises(LeaseLostError, match="daemon-b at epoch 2"):
        repository.create_experiment(make_experiment(), actor=_ACTOR)
    assert _experiments(connection) == 1


# ---- the embedded host's fence -------------------------------------------------------


def test_an_embedded_writer_is_refused_while_a_daemon_lease_is_live(
    connection: sqlite3.Connection, leases: ControllerLeaseStore, clock: _Clock
) -> None:
    repository = ControlPlaneRepository(connection, fence=NoLiveLeaseFence(clock=clock))
    experiment = repository.create_experiment(make_experiment(), actor=_ACTOR)

    lease = _acquired(leases, "daemon-a")
    with pytest.raises(ControllerLeaseHeldError, match="daemon-a"):
        repository.create_experiment(make_experiment(), actor=_ACTOR)
    assert _experiments(connection) == 1
    assert repository.aggregates.load_experiment(str(experiment.id)) == experiment, "reads stay"

    leases.release(lease)
    repository.create_experiment(make_experiment(), actor=_ACTOR)
    assert _experiments(connection) == 2


def test_an_embedded_writer_may_write_once_a_crashed_daemons_lease_expires(
    connection: sqlite3.Connection, leases: ControllerLeaseStore, clock: _Clock
) -> None:
    repository = ControlPlaneRepository(connection, fence=NoLiveLeaseFence(clock=clock))
    _acquired(leases, "daemon-a")
    with pytest.raises(ControllerLeaseHeldError):
        repository.create_experiment(make_experiment(), actor=_ACTOR)
    clock.advance(30)
    repository.create_experiment(make_experiment(), actor=_ACTOR)


# ---- the architecture guard ----------------------------------------------------------


def _calls_to(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
        )
    ]


def test_every_repository_write_goes_through_the_fenced_transaction() -> None:
    """``ControlPlaneRepository`` opens a transaction in one place: ``_write()``.

    A method that opened its own ``write_transaction`` would write unfenced:
    a stale daemon or an embedded host beside a live daemon would commit.
    """
    tree = ast.parse((_XAYTUNE / "storage" / "control_plane.py").read_text())
    (repository,) = (
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ControlPlaneRepository"
    )
    owners = [
        method.name
        for method in repository.body
        if isinstance(method, ast.FunctionDef) and _calls_to(method, "write_transaction")
    ]
    assert owners == ["_write"]
    assert len(_calls_to(tree, "write_transaction")) == 1, "nothing else in the module opens one"
    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.lstrip().upper().startswith("BEGIN")
    ], "nor a transaction by hand"


def test_no_other_module_opens_a_control_plane_write_transaction() -> None:
    """Outside the repository, only the lease store and the runtime's own database may.

    The lease store establishes ownership, so it checks it itself; the local
    runtime's registry is a different database.
    """
    allowed = {
        Path("storage/control_plane.py"),
        Path("storage/leases.py"),
        Path("runtimes/local/registry.py"),
    }
    opening = {
        path.relative_to(_XAYTUNE)
        for path in _XAYTUNE.rglob("*.py")
        if _calls_to(ast.parse(path.read_text()), "write_transaction")
    }
    assert opening <= allowed, opening - allowed
