"""A client that hands a submission to the daemon and dies at once.

``python -m tests.test_daemon.client_process STATE SPEC_JSON`` commits a
submit request for the ``ExperimentSpec`` in SPEC_JSON, prints its request id,
then kills itself with SIGKILL: no clean exit, no chance to wait for anything.
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

from xaytune.daemon import DaemonClient
from xaytune.experiment import ExperimentSpec


def main() -> None:
    state, spec_path = sys.argv[1:3]
    spec = ExperimentSpec.model_validate_json(Path(spec_path).read_text())
    request = DaemonClient(state).submit(spec)
    print(request.id, flush=True)
    os.kill(os.getpid(), signal.SIGKILL)


if __name__ == "__main__":
    main()
