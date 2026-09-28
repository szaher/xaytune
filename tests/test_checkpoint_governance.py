"""The implemented checkpoint layers require an accepted Phase 0 decision."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

SPEC = Path(__file__).resolve().parents[1] / "xaytune-training-harness-spec"


def test_checkpoint_layers_governance_gate_is_accepted():
    adr = (SPEC / "adrs/ADR-009-checkpoint-layers.md").read_text()
    status = adr.split("## Status", 1)[1].split("## ", 1)[0].strip()
    assert re.fullmatch(r"Accepted — \d{4}-\d{2}-\d{2}", status), status
    date.fromisoformat(status.removeprefix("Accepted — "))


@pytest.mark.parametrize("document", ["README.md", "15-implementation-plan.md"])
def test_checkpoint_layers_status_tables_agree(document):
    text = (SPEC / document).read_text()
    if document == "README.md":
        accepted = next(line for line in text.splitlines() if "| Accepted by decision |" in line)
        proposed = next(line for line in text.splitlines() if "| Still `Proposed` |" in line)
    else:
        accepted = text.split("### Accepted by decision", 1)[1].split("### ", 1)[0]
        proposed = text.split("### Still Proposed", 1)[1].split("### ", 1)[0]
    assert "ADR-009" in accepted
    assert "ADR-009" not in proposed
