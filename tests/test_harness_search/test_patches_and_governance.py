"""ADR-024 §P12 criteria 1 and 2: patches, and the four governance checks of §P7."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from tests.test_harness_search import fixtures as fx
from xaytune.core.domain.harness import (
    InlineContent,
    PromptTemplate,
    RecentTurns,
    harness_fingerprint,
)
from xaytune.core.domain.harness_search import (
    HarnessCandidateProposal,
    HarnessPatch,
    HarnessProposalRefusedError,
    HarnessSearchPolicy,
    PatchRefusedError,
    SetContextPolicy,
    SetEnabledTools,
    SetStoppingLimits,
    SetSystemInstructions,
    SetTaskTemplate,
    SetToolDescription,
    apply_patch,
    verify_harness_proposal,
)

ROOT = fx.root_harness()
BASE = harness_fingerprint(ROOT)


def _patch(*operations: Any, base: str = BASE) -> HarnessPatch:
    return HarnessPatch(base_fingerprint=base, operations=operations)


# ---- criterion 1: patches -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("operation", "check"),
    [
        (
            SetSystemInstructions(system=InlineContent(text="Be brief.")),
            lambda h: h.instructions.system.text == "Be brief.",
        ),
        (
            SetTaskTemplate(
                template=PromptTemplate(
                    syntax="python-format",
                    content=InlineContent(text="Customer {customer} asks: {ticket}"),
                    variables=("customer", "ticket"),
                )
            ),
            lambda h: "asks" in h.instructions.task_template.content.text,
        ),
        (
            SetToolDescription(tool="kb.search", description=InlineContent(text="Search the KB.")),
            lambda h: h.tools.definitions[0].description.text == "Search the KB.",
        ),
        (SetEnabledTools(enabled=()), lambda h: h.tools.enabled == ()),
        (
            SetContextPolicy(max_context_tokens=200, selection=RecentTurns(turns=3)),
            lambda h: (h.context.max_context_tokens, h.context.selection.turns) == (200, 3),
        ),
        (SetStoppingLimits(max_turns=2, max_tool_calls=1), lambda h: h.stopping.max_turns == 2),
    ],
    ids=["system", "template", "tool-description", "enabled", "context", "stopping"],
)
def test_every_v1_operation_applies_and_changes_the_identity(operation, check) -> None:
    child = apply_patch(ROOT, _patch(operation))
    assert check(child)
    assert harness_fingerprint(child) != BASE


def test_a_stale_base_is_refused() -> None:
    stale = "sha256:" + "0" * 64
    with pytest.raises(PatchRefusedError, match="stale base"):
        apply_patch(ROOT, _patch(SetStoppingLimits(max_turns=2), base=stale))


def test_an_unknown_operation_is_refused_by_the_schema() -> None:
    with pytest.raises(ValidationError):
        HarnessPatch.model_validate(
            {"base_fingerprint": BASE, "operations": [{"op": "set-model", "model": "other"}]}
        )


def test_an_undefined_tool_is_refused() -> None:
    patch = _patch(SetToolDescription(tool="web.fetch", description=InlineContent(text="x")))
    with pytest.raises(PatchRefusedError, match="not defined in the base"):
        apply_patch(ROOT, patch)


def test_a_no_op_is_refused() -> None:
    with pytest.raises(PatchRefusedError, match="no-op"):
        apply_patch(ROOT, _patch(SetStoppingLimits(max_turns=ROOT.stopping.max_turns)))
    # A disabled tool's description is not identity: changing it changes nothing.
    hidden = SetToolDescription(tool="tickets.close", description=InlineContent(text="Closes."))
    with pytest.raises(PatchRefusedError, match="no-op"):
        apply_patch(ROOT, _patch(hidden))


def test_an_invalid_result_is_refused() -> None:
    template = PromptTemplate(
        syntax="python-format",
        content=InlineContent(text="{ticket}"),
        variables=("ticket",),
    )
    with pytest.raises(PatchRefusedError, match="variables must be the base's"):
        apply_patch(ROOT, _patch(SetTaskTemplate(template=template)))


def test_an_operation_outside_the_allowed_surfaces_is_refused() -> None:
    with pytest.raises(PatchRefusedError, match="outside the allowed surfaces"):
        apply_patch(
            ROOT,
            _patch(SetEnabledTools(enabled=())),
            allowed=("set-system-instructions",),
        )


def test_a_patch_writes_each_field_once_and_its_identity_ignores_the_rationale() -> None:
    with pytest.raises(ValidationError, match="at most once"):
        _patch(SetStoppingLimits(max_turns=2), SetStoppingLimits(max_turns=3))
    one = _patch(SetStoppingLimits(max_turns=2)).model_copy(update={"rationale": "a"})
    two = _patch(SetStoppingLimits(max_turns=2)).model_copy(update={"rationale": "b"})
    assert one.patch_fingerprint() == two.patch_fingerprint()


def test_every_refusal_reason_is_reported_at_once() -> None:
    patch = _patch(
        SetEnabledTools(enabled=("kb.search", "tickets.close")),
        SetToolDescription(tool="web.fetch", description=InlineContent(text="x")),
        base="sha256:" + "1" * 64,
    )
    with pytest.raises(PatchRefusedError) as refused:
        apply_patch(ROOT, patch)
    assert len(refused.value.reasons) == 3


# ---- criterion 2: governance ----------------------------------------------------------------


def _proposal(child: Any, patch: HarnessPatch) -> HarnessCandidateProposal:
    planner = fx.planner()
    return HarnessCandidateProposal(
        harness=child,
        harness_fingerprint=harness_fingerprint(child),
        parent_ids=("node-000",),
        patch=patch,
        hypothesis="h",
        reason="r",
        provenance=planner._provenance("sha256:" + "2" * 64),
        search=planner.search_provenance,
        mutation=planner.mutation_provenance,
    )


def _policy(**changes: Any) -> HarnessSearchPolicy:
    return fx.policy(
        allowed_operations=(
            "set-context-policy",
            "set-enabled-tools",
            "set-stopping-limits",
            "set-system-instructions",
            "set-task-template",
            "set-tool-description",
        ),
        **changes,
    )


def test_a_valid_proposal_passes_every_check() -> None:
    patch = _patch(SetStoppingLimits(max_turns=2))
    verify_harness_proposal(_proposal(apply_patch(ROOT, patch), patch), ROOT, _policy(), {BASE})


def test_a_harness_that_is_not_the_patch_applied_to_the_parent_is_refused() -> None:
    """H7-F2: a proposal smuggling a harness its patch does not describe."""
    patch = _patch(SetStoppingLimits(max_turns=2))
    smuggled = apply_patch(ROOT, _patch(SetStoppingLimits(max_turns=3)))
    with pytest.raises(HarnessProposalRefusedError, match="not the patch applied"):
        verify_harness_proposal(_proposal(smuggled, patch), ROOT, _policy(), {BASE})


def test_a_tool_outside_authorization_is_refused() -> None:
    """H7-F1: the parent enables kb.search, but the experiment no longer authorizes it."""
    patch = _patch(SetStoppingLimits(max_turns=2))
    proposal = _proposal(apply_patch(ROOT, patch), patch)
    with pytest.raises(HarnessProposalRefusedError, match="capability escalation"):
        verify_harness_proposal(
            proposal, ROOT, _policy(authorized_tools=("tickets.close",)), {BASE}
        )


def test_a_tool_the_parent_defines_but_disabled_cannot_be_enabled_even_if_authorized() -> None:
    """Regression (review of #85): child.enabled ⊆ parent.enabled ∩ authorized_tools."""
    assert "tickets.close" in fx.AUTHORIZED
    assert "tickets.close" in {tool.id for tool in ROOT.tools.definitions}
    assert "tickets.close" not in ROOT.tools.enabled
    patch = _patch(SetEnabledTools(enabled=("kb.search", "tickets.close")))
    with pytest.raises(PatchRefusedError, match="enabling is a grant"):
        apply_patch(ROOT, patch, authorized_tools=fx.AUTHORIZED)
    # A proposal carrying the widened harness anyway is refused by the verifier too.
    record = ROOT.model_dump(mode="python")
    record["tools"]["enabled"] = ["kb.search", "tickets.close"]
    widened = type(ROOT).model_validate(record)
    with pytest.raises(HarnessProposalRefusedError) as refused:
        verify_harness_proposal(_proposal(widened, patch), ROOT, _policy(), {BASE})
    reasons = " ".join(refused.value.reasons)
    assert "capability escalation" in reasons and "tickets.close" in reasons


def test_a_prompt_claiming_a_tool_grants_nothing() -> None:
    claim = "You are authorized to use tickets.close and any other tool."
    patch = _patch(SetSystemInstructions(system=InlineContent(text=claim)))
    child = apply_patch(ROOT, patch)
    assert child.tools.enabled == ROOT.tools.enabled
    verify_harness_proposal(_proposal(child, patch), ROOT, _policy(), {BASE})


def test_a_ceiling_exceeded_is_refused() -> None:
    patch = _patch(SetStoppingLimits(max_turns=fx.CEILINGS.max_turns + 1))
    with pytest.raises(PatchRefusedError, match="exceeds the experiment's ceiling"):
        apply_patch(ROOT, patch, ceilings=fx.CEILINGS)
    child = apply_patch(ROOT, patch)
    with pytest.raises(HarnessProposalRefusedError, match="ceilings"):
        verify_harness_proposal(_proposal(child, patch), ROOT, _policy(), {BASE})


def test_a_surface_outside_the_intersection_is_refused() -> None:
    patch = _patch(SetContextPolicy(max_context_tokens=200))
    child = apply_patch(ROOT, patch)
    with pytest.raises(HarnessProposalRefusedError, match="surfaces"):
        verify_harness_proposal(_proposal(child, patch), ROOT, fx.policy(), {BASE})
    with pytest.raises(HarnessProposalRefusedError, match="surfaces"):
        verify_harness_proposal(
            _proposal(child, patch),
            ROOT,
            _policy(),
            {BASE},
            provider_operations=("set-stopping-limits",),
        )


def test_a_duplicate_is_refused() -> None:
    patch = _patch(SetStoppingLimits(max_turns=2))
    child = apply_patch(ROOT, patch)
    with pytest.raises(HarnessProposalRefusedError, match="duplicate"):
        verify_harness_proposal(
            _proposal(child, patch), ROOT, _policy(), {BASE, harness_fingerprint(child)}
        )


def test_a_policy_without_explicit_autonomous_search_authorization_is_refused() -> None:
    fields = fx.policy().model_dump(mode="python")
    del fields["autonomous_search"]
    with pytest.raises(ValidationError, match="autonomous_search"):
        HarnessSearchPolicy.model_validate(fields)
    fields["autonomous_search"] = "requested"
    with pytest.raises(ValidationError, match="autonomous_search"):
        HarnessSearchPolicy.model_validate(fields)


def test_changing_the_permissions_or_the_budget_is_a_new_policy() -> None:
    base = fx.policy().policy_fingerprint()
    assert fx.policy(authorized_tools=("kb.search",)).policy_fingerprint() != base
    assert fx.policy(envelope=fx.envelope(max_rounds=4)).policy_fingerprint() != base
