"""Run the H07 local harness search on the pinned support suite and explain it (ADR-024 §P12).

    python -m tests.harness_search_demo DIRECTORY   → DIRECTORY/search-report.json, explained

Baseline harness → validated mutation → new candidate → verified execution →
H06 evaluation → provider-safe comparison → next candidate, three rounds,
under a spending envelope. The model is the scripted one, so the run is
deterministic and needs nothing installed; every decision and what it spent
is printed.
"""

from __future__ import annotations

import sys
from pathlib import Path

from tests.test_harness_search import fixtures
from xaytune.harness.search import explain_local_search, replay_local_harness_search


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m tests.harness_search_demo DIRECTORY", file=sys.stderr)
        return 2
    report = fixtures.search(Path(argv[0]).resolve())
    print(explain_local_search(report))
    replayed = replay_local_harness_search(report, fixtures.planner())
    print(f"\nReplayed {replayed} planning round(s) from the report: identical proposals.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
