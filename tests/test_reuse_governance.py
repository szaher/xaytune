"""Band G (the planner) requires ADR-017's reuse policy to be an accepted decision."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

SPEC = Path(__file__).resolve().parents[1] / "xaytune-training-harness-spec"


def test_reuse_policy_governance_gate_is_accepted():
    adr = (SPEC / "adrs/ADR-017-reuse-policy.md").read_text()
    status = adr.split("## Status", 1)[1].split("## ", 1)[0].strip().splitlines()[0]
    assert re.fullmatch(r"Accepted — \d{4}-\d{2}-\d{2}", status), status
    date.fromisoformat(status.removeprefix("Accepted — "))


def test_v1_reuse_policy_disables_training_artifact_reuse():
    decision = (SPEC / "adrs/ADR-017-reuse-policy.md").read_text().split("## Decision", 1)[1]
    assert "training artifact reuse is disabled" in decision
    assert "necessary evidence, never sufficient authority" in decision


def test_adr_006_defers_to_the_decided_reuse_policy():
    """ADR-006 split its reuse half out; it must not still call that half open."""
    adr = (SPEC / "adrs/ADR-006-fingerprints-and-reuse.md").read_text()
    flat = " ".join(adr.split())
    assert "Reuse policy is decided in ADR-017" in flat
    assert "decided by ADR-017" in flat
    assert "still open" not in flat
    assert "(`Proposed`)" not in flat


@pytest.mark.parametrize("document", ["README.md", "15-implementation-plan.md"])
def test_reuse_policy_status_tables_agree(document):
    text = (SPEC / document).read_text()
    if document == "README.md":
        accepted = next(line for line in text.splitlines() if "| Accepted by decision |" in line)
        proposed = next(line for line in text.splitlines() if "| Still `Proposed` |" in line)
        assert "ADR-011 – ADR-017" in accepted
    else:
        accepted = text.split("### Accepted by decision", 1)[1].split("### ", 1)[0]
        proposed = text.split("### Still Proposed", 1)[1].split("### ", 1)[0]
        assert "ADR-017" in accepted
    assert "ADR-017" not in proposed
