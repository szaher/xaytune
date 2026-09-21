"""Cross-process write contention (ADR-005 §11, test 3).

``test_separate_connections_contend_without_losing_an_update`` proves that two
*connections* contend correctly. That is not the same claim. Both connections
live in one interpreter, so it would still pass if correctness quietly depended
on a shared connection object, a module-level lock, or any other process-local
state.

The frozen claim is stronger:

    Repository correctness must not depend on ADR-004's controller lease.
    Multiple processes may contend for writes.

so the test has to cross a process boundary. One deterministic case is enough --
this is not a fuzzing suite. Determinism comes from sequencing the two processes
through files rather than from sleeping and hoping.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

from xaytune.core import ExperimentStatus
from xaytune.storage import AggregateStore, write_transaction

from .conftest import make_experiment

# Runs in a separate interpreter: reads the experiment, transitions it, and
# reports whether its write won or lost the revision CAS.
_CHILD = textwrap.dedent(
    """
    import json, sys
    from xaytune.core import ExperimentStatus
    from xaytune.core.errors import ConcurrentModificationError
    from xaytune.storage import AggregateStore, connect, write_transaction

    db_path, experiment_id, target, ready_path, go_path = sys.argv[1:6]

    connection = connect(db_path)
    store = AggregateStore(connection)

    # Read before signalling, so both processes have the same revision in hand.
    experiment = store.load_experiment(experiment_id)
    open(ready_path, "w").write(str(experiment.revision))

    # Wait for the parent to release us; both writes are then in flight against
    # the same starting revision.
    import os, time
    while not os.path.exists(go_path):
        time.sleep(0.01)

    try:
        with write_transaction(connection):
            store._update_experiment(experiment.with_status(ExperimentStatus(target)))
        outcome = "won"
    except ConcurrentModificationError:
        outcome = "lost"
    finally:
        connection.close()

    print(json.dumps({"outcome": outcome, "read_revision": experiment.revision}))
    """
)


def _run_child(db_path: Path, experiment_id: str, target: str, tmp_path: Path) -> Any:
    ready = tmp_path / "child-ready"
    go = tmp_path / "child-go"
    script = tmp_path / "child.py"
    script.write_text(_CHILD, encoding="utf-8")

    process = subprocess.Popen(
        [sys.executable, str(script), str(db_path), experiment_id, target, str(ready), str(go)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return process, ready, go


def test_two_processes_contend_and_exactly_one_transition_survives(
    db_path: Path, connection: sqlite3.Connection, store: AggregateStore, tmp_path: Path
) -> None:
    """Both read revision 0; one commits, the other's CAS is refused.

    No lease, no lock file, no coordination beyond SQLite and the revision
    guard -- which is exactly the guarantee ADR-005 §8 puts on the repository.
    """
    experiment = make_experiment()
    with write_transaction(connection):
        store._insert_experiment(experiment)
    experiment_id = str(experiment.id)

    process, ready, go = _run_child(db_path, experiment_id, "cancelled", tmp_path)
    try:
        # Both sides now hold the same revision.
        deadline = time.monotonic() + 30
        while not ready.exists():
            assert process.poll() is None, f"child died early: {process.communicate()}"
            assert time.monotonic() < deadline, "child never signalled that it had read"
            time.sleep(0.01)
        child_read_revision = int(ready.read_text())

        parent_copy = store.load_experiment(experiment_id)
        assert parent_copy.revision == child_read_revision

        # Parent commits first, then releases the child.
        with write_transaction(connection):
            store._update_experiment(parent_copy.with_status(ExperimentStatus.ACTIVE))
        go.write_text("go", encoding="utf-8")

        stdout, stderr = process.communicate(timeout=60)
        assert process.returncode == 0, stderr
        result = json.loads(stdout.strip().splitlines()[-1])
    finally:
        if process.poll() is None:  # pragma: no cover - only on an early failure
            process.kill()

    assert result["read_revision"] == child_read_revision
    assert result["outcome"] == "lost", "the second writer must not overwrite the first"

    # Exactly one transition survived, and it is the one that committed first.
    final = store.load_experiment(experiment_id)
    assert final.status is ExperimentStatus.ACTIVE
    assert final.revision == experiment.revision + 1
