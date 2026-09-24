"""What a caller submits: one candidate, and everything needed to run it.

An :class:`ExperimentSpec` is data, all the way down, because everything in it
is persisted and read back by processes that did not create it (ADR-016). A
field that held a live object -- a compiler instance, a runtime, a callback --
would be refused at construction, naming the field: rejecting it at submission
is the point, since an experiment that runs for six hours and then cannot be
recovered is worse than one that refuses to start.

One candidate per experiment in this phase. A planner that proposes more is
band G; the shape here does not have to change for it, because the durable
record already has nodes and runs.
"""

from __future__ import annotations

import os

from pydantic import Field, field_validator

from xaytune.core.domain.candidate import CandidateSpec
from xaytune.core.domain.objective import Objective
from xaytune.core.domain.specs import CompilerSpec, RuntimeSpec
from xaytune.core.immutable import FrozenDomainModel

__all__ = ["CompilerSpec", "ExperimentSpec", "RuntimeSpec"]


class ExperimentSpec(FrozenDomainModel):
    """One candidate to train, and the specs that say how.

    Attributes:
        seed: The first run's seed. It belongs to the run, not the candidate
            (ADR-011), and is required rather than defaulted: a run with an
            invented seed is not reproducible, whatever the seed is.
        compiler: Which compiler compiles the candidate. Named explicitly in
            this phase; capability resolution chooses one later.
        runtime: Which backend executes it, and its configuration.
        artifact_root: Where each run's model is published, as an absolute
            local directory; a run writes to ``<artifact_root>/<run_id>``.
    """

    name: str = Field(min_length=1)
    objective: Objective
    candidate: CandidateSpec
    seed: int
    compiler: CompilerSpec
    runtime: RuntimeSpec
    artifact_root: str
    hypothesis: str | None = None

    @field_validator("compiler", "runtime")
    @classmethod
    def _unbound(cls, spec: CompilerSpec | RuntimeSpec) -> CompilerSpec | RuntimeSpec:
        # The version is the host's to resolve and record. Accepting one from
        # the caller would let the record claim an implementation version
        # nothing checked.
        if spec.version is not None:
            raise ValueError(
                "version is resolved by the host at submission and recorded; do not supply it"
            )
        return spec

    @field_validator("artifact_root")
    @classmethod
    def _absolute(cls, value: str) -> str:
        if not os.path.isabs(value):
            raise ValueError(
                f"artifact_root {value!r} is not an absolute path; it is resolved by "
                f"worker processes whose working directory is not the caller's"
            )
        return value
