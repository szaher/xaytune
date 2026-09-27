"""Whether a proposed action is permitted (PR-023).

```text
ActionSpec + PolicyContext  ──evaluate──>  PolicyProposal  ──recorded──>  PolicyDecision
```

A :class:`PolicyEngine` is the sibling of a :class:`~xaytune.decision.DecisionEngine`,
held to the same rule: **it judges; it applies nothing.** It reads only the spec
and its context -- no clock, no id, no database, no runtime -- so the same
inputs give the same proposal, byte for byte, and the repository can tell that
the decision it records is about the state the engine actually saw.

Two are built in:

- :class:`DenyAllPolicy`: what a host uses when no policy is configured. Every
  governed proposal is denied, with a durable decision saying why. Fail closed.
- :class:`RulePolicyEngine`: ordered rules on action type or mutation class;
  the first that matches decides, and otherwise the default does.
  ``RulePolicyEngine(default=PolicyVerdict.ALLOW)`` is the permissive choice,
  made explicitly.

Cancellation is never governed here: it is controller-owned and always
possible (ADR-013), and goes through the cancellation API.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import Field, model_validator

from xaytune.core.domain.actions import ActionSpec, MutationClass
from xaytune.core.domain.policy import PolicyContext, PolicyProposal, PolicyVerdict
from xaytune.core.fingerprint import fingerprint
from xaytune.core.immutable import FrozenDomainModel

__all__ = [
    "DenyAllPolicy",
    "PolicyEngine",
    "PolicyRule",
    "PolicyVerdict",
    "RulePolicyEngine",
]


@runtime_checkable
class PolicyEngine(Protocol):
    """Decides whether one proposed action is permitted, from its spec and context alone."""

    name: str
    version: str

    def evaluate(self, spec: ActionSpec, context: PolicyContext) -> PolicyProposal:
        """The verdict *spec* in *context* gets.

        Pure: nothing but the arguments is read, and nothing is minted. The
        proposal's ``input_fingerprint`` must be ``context.input_fingerprint()``.
        """
        ...


class DenyAllPolicy:
    """Every governed action is denied: what no configured policy means."""

    name = "deny-all"
    version = "1"

    def evaluate(self, spec: ActionSpec, context: PolicyContext) -> PolicyProposal:
        return PolicyProposal(
            verdict=PolicyVerdict.DENY,
            reasons=(
                "no policy is configured, and without one every governed action is "
                "denied; configure a PolicyEngine to permit actions",
            ),
            rule_ids=("deny-all",),
            engine_name=self.name,
            engine_version=self.version,
            input_fingerprint=context.input_fingerprint(),
        )


class PolicyRule(FrozenDomainModel):
    """One rule: which actions it matches, and what it says about them.

    Matches an action whose type is in ``action_types`` (if any are given)
    and whose mutation class is in ``mutation_classes`` (if any are given).
    A rule names at least one of the two: a rule matching everything is what
    the engine's default is for.
    """

    id: str = Field(min_length=1)
    verdict: PolicyVerdict
    reason: str = Field(min_length=1)
    action_types: tuple[str, ...] = Field(default_factory=tuple)
    mutation_classes: tuple[MutationClass, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _matches_something_specific(self) -> PolicyRule:
        if not self.action_types and not self.mutation_classes:
            raise ValueError(
                f"rule {self.id!r} names no action type and no mutation class; "
                f"use the engine's default for what every action gets"
            )
        return self

    def matches(self, context: PolicyContext) -> bool:
        return (not self.action_types or context.action_type in self.action_types) and (
            not self.mutation_classes or context.mutation_class in self.mutation_classes
        )


class RulePolicyEngine:
    """Ordered rules; the first match decides, and the default decides the rest.

    ``version`` names the rule set as well as the engine -- ``1+sha256:...`` --
    so a recorded decision says exactly which rules made it.
    """

    name = "rules"

    def __init__(
        self,
        rules: tuple[PolicyRule, ...] | list[PolicyRule] = (),
        *,
        default: PolicyVerdict = PolicyVerdict.DENY,
        default_reason: str | None = None,
    ) -> None:
        ids = [rule.id for rule in rules]
        if len(set(ids)) != len(ids):
            raise ValueError(f"rule ids repeat: {sorted({i for i in ids if ids.count(i) > 1})}")
        if "default" in ids:
            raise ValueError("'default' is the id of the engine's default, not of a rule")
        self.rules = tuple(rules)
        self.default = default
        self.default_reason = default_reason or f"no rule matched; the default is {default.value}"
        rule_set = {
            "rules": [rule.model_dump(mode="json") for rule in self.rules],
            "default": default.value,
            "default_reason": self.default_reason,
        }
        self.version = f"1+{fingerprint(rule_set)}"

    def evaluate(self, spec: ActionSpec, context: PolicyContext) -> PolicyProposal:
        for rule in self.rules:
            if rule.matches(context):
                verdict, reason, rule_id = rule.verdict, rule.reason, rule.id
                break
        else:
            verdict, reason, rule_id = self.default, self.default_reason, "default"
        return PolicyProposal(
            verdict=verdict,
            reasons=(reason,),
            rule_ids=(rule_id,),
            engine_name=self.name,
            engine_version=self.version,
            input_fingerprint=context.input_fingerprint(),
        )
