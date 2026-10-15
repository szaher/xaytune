"""Record a real experiment into a state database, for the Studio browser tests (UX-02).

    python -m tests.studio_record DIRECTORY        → prints DIRECTORY/state.db

The spec 18 adaptive scenario, run by a real controller host, exactly as
``tests/test_api`` records it: training with a CUDA OOM and a recovered
attempt, evaluation, a decision, a planner branch and the child's training.
Nothing here writes records by hand; the host is closed before it returns.
"""

from __future__ import annotations

import sys
from pathlib import Path

from tests.test_api.conftest import _record


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m tests.studio_record DIRECTORY", file=sys.stderr)
        return 2
    directory = Path(argv[0])
    directory.mkdir(parents=True, exist_ok=True)
    recorded = _record(directory)
    print(recorded.state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
