"""Ray Tune as a search provider -- and nothing more (PR-034).

```text
SearchSpace ──tune_search_space()──▶ Tune domains ──OptunaSearch──▶ suggestion
                                                                    │ apply_parameters()
                                                                    ▼
                                             CandidateSpec ──▶ CandidateProposal
CandidateObservation ──replay──▶ on_trial_complete()
```

Ray Tune searches; Xaytune owns the experiment. This module is the only place
Ray Tune is imported, and it imports it only when a search is replayed: the
provider uses Tune's :class:`~ray.tune.search.Searcher` interface --
``suggest(trial_id)`` and ``on_trial_complete(trial_id, result, error)`` --
and nothing else of Tune. There is no ``Tuner``, no trial runner, no
``ray.init()``: no Ray cluster, runtime or job is touched, so the same search
serves an experiment whose candidates run locally, as Ray jobs or under Ray
Train.

**The algorithm** is Tune's ``OptunaSearch`` with an explicitly seeded
sampler -- ``tpe`` (``TPESampler``, the default) or ``random``
(``RandomSampler``) -- over the search space converted to Tune's domains and,
by Tune, to Optuna's distributions. Its mode is the objective's direction.
Everything else is :class:`~xaytune.search.SequentialSearchProvider`'s: a
fresh searcher for every suggestion, the record replayed into it, one
suggestion at a time.

**What Tune is told**, per observation:

```text
measured                          on_trial_complete(id, {metric: value})   COMPLETE
unmeasured, rejected, failed,     on_trial_complete(id, None, error=True)  FAIL
  cancelled -- no value
a repeat of the base candidate    on_trial_complete(id, None, error=True)  FAIL
```

A point that was not measured is never given a value; ``FAIL`` is Tune's way
to say a trial taught nothing about the objective. The outcome itself stays
in Xaytune's record, and in the history fingerprint.

``pip install xaytune[ray-tune]`` -- Ray Tune and Optuna, each pinned to one
minor. The exact releases are the search's **engine**: binding records them
(``config.engine``), every proposal's search record names them, and a
recorded search is refused on any other release -- even one that would
reproduce every recorded suggestion, since the next one is what a restart
must reproduce.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import metadata
from typing import Any, Literal

from pydantic import field_validator

from xaytune._version import __version__
from xaytune.core.capabilities import PLUGIN_API_VERSIONS, PluginDescriptor
from xaytune.core.domain.objective import Objective
from xaytune.core.domain.search import (
    CandidateObservation,
    ChoiceParameter,
    FloatParameter,
    IntParameter,
    SearchSpace,
)
from xaytune.search import (
    SearchProviderConfigurationError,
    SequentialSearchConfig,
    SequentialSearchProvider,
)

__all__ = [
    "RayTuneSearchConfig",
    "RayTuneSearchProvider",
    "tune_search_space",
]

_ENGINE = ("ray", "optuna")
"""The distributions whose exact releases are this search's engine."""

_METRIC = "objective"
"""What Tune's searcher calls the objective's primary metric; the name is internal."""


class RayTuneSearchConfig(SequentialSearchConfig):
    """The search space, the seed, and which seeded Optuna sampler Tune drives.

    A ``choice`` whose values are distinct to Xaytune but equal to Python --
    ``1``, ``1.0`` and ``True``; ``0`` and ``False`` -- is refused: Optuna
    keeps categorical choices by Python equality, so it could not tell them
    apart, and would search a different space than the one recorded.
    """

    algorithm: Literal["tpe", "random"] = "tpe"

    @field_validator("search_space")
    @classmethod
    def _choices_optuna_can_tell_apart(cls, space: SearchSpace) -> SearchSpace:
        aliased = []
        for parameter in space.parameters:
            if not isinstance(parameter, ChoiceParameter):
                continue
            values = parameter.values
            for i, a in enumerate(values):
                for b in values[i + 1 :]:
                    if a == b:
                        aliased.append(f"{parameter.name}: {a!r} and {b!r}")
        if aliased:
            raise ValueError(
                "choices Optuna cannot tell apart (equal under Python equality): "
                + "; ".join(aliased)
            )
        return space


class RayTuneSearchProvider(SequentialSearchProvider):
    """A sequential search driven by Ray Tune's ``OptunaSearch``, replayed from the record.

    Bind it as ``SearchProviderSpec(kind="ray-tune", config={...})`` through
    :func:`~xaytune.search.search_planner_factory`::

        planners={**PLANNERS, "search": search_planner_factory(
            {"ray-tune": RayTuneSearchProvider.from_spec}
        )}
    """

    descriptor = PluginDescriptor(
        api_version=PLUGIN_API_VERSIONS[0],
        name="ray-tune",
        plugin_version="1.0.0",
        provider="xaytune",
        xaytune_version=__version__,
    )
    config_type = RayTuneSearchConfig
    config: RayTuneSearchConfig

    @classmethod
    def engine_versions(cls) -> Mapping[str, str]:
        """The installed Ray and Optuna releases, from their distributions' metadata.

        Read without importing either, so binding a recorded search on a
        machine with other releases is refused before anything is run.
        """
        versions = {}
        missing = []
        for distribution in _ENGINE:
            try:
                versions[distribution] = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                missing.append(distribution)
        if missing:
            raise SearchProviderConfigurationError(
                cls.descriptor.name,
                (f"{', '.join(missing)} not installed: pip install xaytune[ray-tune]",),
            )
        return versions

    def searcher(self, objective: Objective) -> _TuneSearcher:
        return _TuneSearcher(self.config, objective)


def tune_search_space(space: SearchSpace) -> dict[str, Any]:
    """*space* as Ray Tune's search domains, by parameter name.

    ``int`` ranges are inclusive in Xaytune and exclusive at the top in Tune;
    ``choice`` values keep their order and their types.
    """
    from ray import tune

    domains: dict[str, Any] = {}
    for parameter in space.parameters:
        if isinstance(parameter, FloatParameter):
            real = tune.loguniform if parameter.log else tune.uniform
            domains[parameter.name] = real(parameter.low, parameter.high)
        elif isinstance(parameter, IntParameter):
            integer = tune.lograndint if parameter.log else tune.randint
            domains[parameter.name] = integer(parameter.low, parameter.high + 1)
        else:
            assert isinstance(parameter, ChoiceParameter)
            domains[parameter.name] = tune.choice(list(parameter.values))
    return domains


class _TuneSearcher:
    """One fresh, seeded ``OptunaSearch``, addressed by suggestion index."""

    def __init__(self, config: RayTuneSearchConfig, objective: Objective) -> None:
        import optuna
        from ray.tune.search.optuna import OptunaSearch

        sampler: optuna.samplers.BaseSampler = (
            optuna.samplers.TPESampler(seed=config.seed)
            if config.algorithm == "tpe"
            else optuna.samplers.RandomSampler(seed=config.seed)
        )
        self._search = OptunaSearch(
            space=OptunaSearch.convert_search_space(tune_search_space(config.search_space)),
            metric=_METRIC,
            mode="max" if objective.primary.direction == "maximize" else "min",
            sampler=sampler,
        )

    def suggest(self, index: int) -> Mapping[str, Any] | None:
        from ray.tune.search import Searcher

        suggestion = self._search.suggest(str(index))
        if suggestion is None or suggestion == Searcher.FINISHED:
            return None
        return dict(suggestion)

    def complete(self, index: int, observation: CandidateObservation | None) -> None:
        if observation is not None and observation.value is not None:
            self._search.on_trial_complete(str(index), {_METRIC: observation.value})
        else:
            self._search.on_trial_complete(str(index), None, error=True)
