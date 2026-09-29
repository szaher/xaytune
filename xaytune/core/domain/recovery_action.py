"""Immutable provenance between an OOM recovery decision and its governed Action."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from xaytune.core.checkpoint import Digest
from xaytune.core.clock import utc_now
from xaytune.core.domain.oom_recovery import OOMResizeProposal
from xaytune.core.fingerprint import fingerprint
from xaytune.core.ids import ActionId, RecoveryEpisodeId, RecoveryPlanId
from xaytune.core.immutable import FrozenDomainModel


class RecoveryActionBinding(FrozenDomainModel):
    """An append-only, decision-bound Action proposal; no execution authority."""

    schema_version: Literal["xaytune.recovery-action-binding/v1alpha1"] = (
        "xaytune.recovery-action-binding/v1alpha1"
    )
    action_id: ActionId
    episode_id: RecoveryEpisodeId
    plan_id: RecoveryPlanId
    plan_sequence: int = Field(ge=1, strict=True)
    proposal: OOMResizeProposal
    proposal_fingerprint: Digest
    input_fingerprint: Digest
    source_execution_state_fingerprint: Digest
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _bound(self) -> RecoveryActionBinding:
        proposal = self.proposal
        if (
            self.episode_id != proposal.episode_id
            or self.plan_id != proposal.plan_id
            or self.plan_sequence != proposal.plan_sequence
            or self.proposal_fingerprint != fingerprint(proposal)
            or self.input_fingerprint != proposal.input_fingerprint
            or self.source_execution_state_fingerprint
            != proposal.source_execution_state_fingerprint
        ):
            raise ValueError("recovery Action binding disagrees with its OOM proposal")
        return self

    @classmethod
    def for_proposal(
        cls, action_id: ActionId, proposal: OOMResizeProposal
    ) -> RecoveryActionBinding:
        return cls(
            action_id=action_id,
            episode_id=proposal.episode_id,
            plan_id=proposal.plan_id,
            plan_sequence=proposal.plan_sequence,
            proposal=proposal,
            proposal_fingerprint=fingerprint(proposal),
            input_fingerprint=proposal.input_fingerprint,
            source_execution_state_fingerprint=proposal.source_execution_state_fingerprint,
        )
