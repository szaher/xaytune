"""LMEvalEvaluator: a task name resolved once, into a binding, and judged before and after.

```text
supports(declared) → resolve() → supports(resolved) → recorded → prepare()
                     the only step that
                     reads the Hub
```

lm-eval-free, as the evaluator is when it prepares: resolution goes through a
:class:`~xaytune.evaluation.lmeval.TaskResolver`, and these tests hand it one
that answers without lm-eval or the network. What the real resolver binds is
``test_lmeval_tasks.py``; the host's two checks around ``resolve()`` are
``tests/test_experiment/test_evaluation_resolution.py``.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

from xaytune.core.domain.evaluation import EvaluationSpec, EvaluatorDeterminism, EvaluatorSpec
from xaytune.core.ids import ArtifactId
from xaytune.core.refs import ArtifactRef, DatasetRef
from xaytune.evaluation import EvaluationContext, Evaluator, UnsupportedEvaluationError
from xaytune.evaluation.lmeval import LMEvalEvaluator, UnresolvableTaskError, _portable
from xaytune.workers.eval_lmeval import measurement_from
from xaytune.workers.eval_lmeval_schema import (
    LM_EVAL_VERSION,
    LMEvalTaskBinding,
    LMEvalWorkerConfig,
    task_config_digest,
)

CONFIG: dict[str, Any] = {
    "task": "arc_easy",
    "num_fewshot": 0,
    "batch_size": 4,
    "precision": "fp32",
    "limit": 20,
}


def _binding(**changes: Any) -> LMEvalTaskBinding:
    fields: dict[str, Any] = {
        "task": "arc_easy",
        "task_version": "1.0",
        "task_config_digest": "sha256:" + "d" * 64,
        "lm_eval_version": LM_EVAL_VERSION,
        "dataset_path": "allenai/ai2_arc",
        "dataset_name": "ARC-Easy",
        "dataset_revision": "2" * 40,
        "output_type": "multiple_choice",
        "num_fewshot": 0,
        "metrics": ("acc", "acc_norm"),
    }
    fields.update(changes)
    return LMEvalTaskBinding(**fields)


class _Resolver:
    """Answers with a fixed binding, or refuses; counts what it was asked."""

    def __init__(self, binding: LMEvalTaskBinding | None = None, *reasons: str) -> None:
        self.binding = binding or _binding()
        self.reasons = reasons
        self.asked: list[tuple[str, int]] = []

    def bind(self, task: str, num_fewshot: int) -> LMEvalTaskBinding:
        self.asked.append((task, num_fewshot))
        if self.reasons:
            raise UnresolvableTaskError(task, self.reasons)
        return self.binding


def _spec(**config: Any) -> EvaluationSpec:
    values = {**CONFIG, **config}
    return EvaluationSpec(
        evaluator=EvaluatorSpec(
            name="lm-eval", config={k: v for k, v in values.items() if v is not None}
        )
    )


def _resolved(binding: LMEvalTaskBinding | None = None, **config: Any) -> EvaluationSpec:
    return LMEvalEvaluator(_Resolver(binding)).resolve(_spec(**config))


def _subject(**fields: object) -> ArtifactRef:
    values: dict[str, object] = {"id": ArtifactId.generate(), "kind": "model", "uri": "/m/run_1"}
    values.update(fields)
    return ArtifactRef(**values)  # type: ignore[arg-type]


def _context(**fields: object) -> EvaluationContext:
    values: dict[str, object] = {
        "experiment_id": "exp_1",
        "node_id": "node_1",
        "evaluation_run_id": "evalrun_1",
        "seed": 7,
        "replicate": 1,
        "output_uri": "/out/evalrun_1",
    }
    values.update(fields)
    return EvaluationContext(**values)  # type: ignore[arg-type]


def _reasons(spec: EvaluationSpec) -> str:
    return " | ".join(LMEvalEvaluator(_Resolver()).supports(spec).reasons)


# ---- what it is --------------------------------------------------------------------


def test_it_is_an_evaluator_that_declares_itself_seeded() -> None:
    evaluator = LMEvalEvaluator(_Resolver())
    assert isinstance(evaluator, Evaluator)
    assert evaluator.descriptor.name == "lm-eval"
    assert evaluator.determinism is EvaluatorDeterminism.SEEDED


def test_the_host_offers_it_by_default() -> None:
    from xaytune.experiment.host import _default_evaluators

    assert _default_evaluators()["lm-eval"] is LMEvalEvaluator


def test_the_evaluator_needs_neither_lm_eval_nor_the_training_stack() -> None:
    """It prepares in the controller (ADR-010); only resolving and running need lm-eval."""
    code = (
        "import sys, xaytune.evaluation.lmeval, xaytune.workers.eval_lmeval\n"
        "from xaytune.evaluation.lmeval import LMEvalEvaluator\n"
        "LMEvalEvaluator()\n"
        "heavy = ('torch', 'transformers', 'datasets', 'lm_eval')\n"
        "heavy = [m for m in heavy if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


# ---- judged as declared --------------------------------------------------------------


def test_a_declared_task_is_supported_before_it_is_resolved() -> None:
    assert LMEvalEvaluator(_Resolver()).supports(_spec())


def test_limit_is_optional() -> None:
    assert LMEvalEvaluator(_Resolver()).supports(_spec(limit=None))


@pytest.mark.parametrize("missing", ["task", "num_fewshot", "batch_size", "precision"])
def test_nothing_that_changes_the_numbers_is_defaulted(missing: str) -> None:
    config = {k: v for k, v in CONFIG.items() if k != missing}
    spec = EvaluationSpec(evaluator=EvaluatorSpec(name="lm-eval", config=config))
    assert f"evaluator.config.{missing}" in _reasons(spec)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"precision": "bf16"}, "evaluator.config.precision"),
        ({"num_fewshot": -1}, "evaluator.config.num_fewshot"),
        ({"limit": 0}, "evaluator.config.limit"),
        ({"batch_size": 0}, "evaluator.config.batch_size"),
        ({"generation": {"temperature": 0}}, "evaluator.config.generation"),
    ],
    ids=["precision", "fewshot", "limit", "batch", "unknown-field"],
)
def test_a_config_it_cannot_honour_exactly_is_refused(change: dict, reason: str) -> None:
    assert reason in _reasons(_spec(**change))


def test_a_dataset_and_slices_are_refused() -> None:
    spec = _spec().model_copy(
        update={"dataset": DatasetRef(uri="/data/x.jsonl"), "slices": ("easy",)}
    )
    reasons = _reasons(spec)
    assert "names its own dataset" in reasons
    assert "no slices to select" in reasons


# ---- resolved once, into a binding ---------------------------------------------------


def test_resolving_asks_for_the_task_and_records_its_binding() -> None:
    resolver = _Resolver()
    resolved = LMEvalEvaluator(resolver).resolve(_spec())

    assert resolver.asked == [("arc_easy", 0)]
    config = dict(resolved.evaluator.config)
    assert LMEvalTaskBinding.model_validate(dict(config.pop("binding"))) == _binding()
    assert config == CONFIG, "what the caller declared is kept as declared"


def test_a_binding_is_never_taken_from_the_caller() -> None:
    forged = _spec(binding=_binding().model_dump(mode="json"))
    assert LMEvalEvaluator(_Resolver()).supports(forged), "it describes a supported task"

    with pytest.raises(UnsupportedEvaluationError, match="not taken from the caller"):
        LMEvalEvaluator(_Resolver()).resolve(forged)


def test_a_task_that_cannot_be_bound_is_refused_with_every_reason() -> None:
    resolver = _Resolver(None, "not registered", "no dataset")
    with pytest.raises(UnsupportedEvaluationError) as refused:
        LMEvalEvaluator(resolver).resolve(_spec())
    assert refused.value.reasons == ("not registered", "no dataset")


# ---- judged again, as resolved -------------------------------------------------------


def test_a_resolved_supported_task_is_supported() -> None:
    assert LMEvalEvaluator(_Resolver()).supports(_resolved())


@pytest.mark.parametrize(
    ("binding", "reason"),
    [
        (
            _binding(task="gsm8k", output_type="generate_until", metrics=("exact_match",)),
            "decoding settings, stop sequences, filters and answer extraction",
        ),
        (_binding(output_type="loglikelihood_rolling"), "'loglikelihood_rolling'"),
        (_binding(metrics=("perplexity", "acc")), "reports ['perplexity']"),
        (_binding(lm_eval_version="0.4.12"), "bound under lm-eval 0.4.12"),
        (_binding(task="hellaswag"), "config names 'arc_easy'"),
        (_binding(num_fewshot=5), "5-shot"),
    ],
    ids=["generation", "rolling", "metric", "release", "task", "fewshot"],
)
def test_what_a_task_resolves_to_can_still_be_refused(
    binding: LMEvalTaskBinding, reason: str
) -> None:
    """Passing the declared check says nothing about what the name turns out to be."""
    resolved = _resolved(binding)
    assert reason in _reasons(resolved)


# ---- prepared from the record alone ------------------------------------------------


def test_an_unresolved_spec_is_never_prepared() -> None:
    with pytest.raises(UnsupportedEvaluationError, match="never resolved"):
        LMEvalEvaluator(_Resolver()).prepare(_subject(), _spec(), _context())


def test_prepare_refuses_a_subject_or_run_it_cannot_measure() -> None:
    with pytest.raises(UnsupportedEvaluationError) as refused:
        LMEvalEvaluator(_Resolver()).prepare(
            _subject(kind="checkpoint", uri="relative/model"),
            _resolved(),
            _context(seed=None, output_uri="relative/out"),
        )
    reasons = " | ".join(refused.value.reasons)
    for expected in ("not a model", "not an absolute local path", "no seed", "report to"):
        assert expected in reasons


def test_prepare_does_not_resolve() -> None:
    resolver = _Resolver()
    evaluator = LMEvalEvaluator(resolver)
    resolved = _resolved()
    evaluator.prepare(_subject(), resolved, _context())
    assert resolver.asked == []


def test_the_plan_carries_the_binding_the_seed_and_the_limit() -> None:
    plan = LMEvalEvaluator(_Resolver()).prepare(_subject(), _resolved(), _context())

    config = LMEvalWorkerConfig.model_validate(dict(plan.config))
    assert config.binding == _binding()
    assert config.realization.seed == 7
    assert config.measure.limit == 20
    assert config.model_uri == "/m/run_1"
    assert plan.entrypoint.module == "xaytune.workers.eval_lmeval"


def test_preparing_is_deterministic() -> None:
    subject, resolved, context = _subject(), _resolved(), _context()
    first = LMEvalEvaluator(_Resolver()).prepare(subject, resolved, context)
    second = LMEvalEvaluator(_Resolver()).prepare(subject, resolved, context)
    assert first.model_dump() == second.model_dump()


@pytest.mark.parametrize(
    "change",
    [
        {"dataset_revision": "3" * 40},
        {"task_config_digest": "sha256:" + "e" * 64},
        {"task_version": "2.0"},
    ],
    ids=["dataset-commit", "definition", "task-version"],
)
def test_what_the_task_resolved_to_is_part_of_the_fingerprint(change: dict) -> None:
    """Same name, same shots: another dataset commit or definition is another evaluation."""
    assert (
        _resolved(_binding(**change)).evaluation_fingerprint()
        != _resolved().evaluation_fingerprint()
    )
    assert _resolved().evaluation_fingerprint() == _resolved().evaluation_fingerprint()


def test_the_worker_refuses_a_binding_it_cannot_run() -> None:
    plan = LMEvalEvaluator(_Resolver()).prepare(_subject(), _resolved(), _context())
    config = dict(plan.config)
    config["binding"] = {**config["binding"], "output_type": "generate_until"}
    with pytest.raises(ValueError, match="not supported"):
        LMEvalWorkerConfig.model_validate(config)


# ---- what the worker reads out of lm-eval -------------------------------------------


def _results(**scored: Any) -> dict[str, Any]:
    values = {"acc,none": 0.55, "acc_stderr,none": 0.114, "acc_norm,none": 0.4}
    values.update(scored)
    values.setdefault("acc_norm_stderr,none", 0.112)
    return {
        "results": {"arc_easy": values},
        "n-samples": {"arc_easy": {"original": 2376, "effective": 20}},
    }


def test_sample_count_is_what_lm_eval_scored_not_the_dataset() -> None:
    measurements, counts = measurement_from(_results(), _binding())

    assert counts == {"original": 2376, "effective": 20}
    assert [(m.name, m.value, m.standard_error, m.sample_count) for m in measurements] == [
        ("acc", 0.55, 0.114, 20),
        ("acc_norm", 0.4, 0.112, 20),
    ]


def test_a_standard_error_lm_eval_could_not_compute_is_left_out() -> None:
    measurements, _ = measurement_from(_results(**{"acc_stderr,none": "N/A"}), _binding())
    assert measurements[0].standard_error is None


def test_a_bound_metric_lm_eval_did_not_report_is_an_error() -> None:
    results = _results()
    del results["results"]["arc_easy"]["acc_norm,none"]
    with pytest.raises(KeyError):
        measurement_from(results, _binding())


# ---- a definition's digest does not depend on where lm-eval is installed -----------


def test_the_definition_digest_is_independent_of_the_install_location() -> None:
    def config(root: str) -> dict:
        return {
            "task": "hellaswag",
            "process_docs": f"{root}tasks/hellaswag/utils.process_docs",
            "metric_list": [{"metric": "acc"}],
        }

    here = _portable(config("/venv-a/site-packages/lm_eval/"), "/venv-a/site-packages/lm_eval/")
    there = _portable(config("/opt/other/lm_eval/"), "/opt/other/lm_eval/")
    assert here["process_docs"] == "lm_eval/tasks/hellaswag/utils.process_docs"
    assert task_config_digest(here) == task_config_digest(there)


# ---- one release, named the same way everywhere -------------------------------------


def test_the_eval_extra_and_the_lock_pin_the_release_tasks_are_bound_under() -> None:
    """pyproject, uv.lock and LM_EVAL_VERSION agree: `uv sync` installs what the worker runs."""
    from pathlib import Path

    from packaging.requirements import Requirement

    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pytest itself requires tomli before 3.11
        import tomli as tomllib

    root = Path(__file__).resolve().parents[2]
    with (root / "pyproject.toml").open("rb") as file:
        (requirement,) = tomllib.load(file)["project"]["optional-dependencies"]["eval"]
    assert str(Requirement(requirement).specifier) == f"=={LM_EVAL_VERSION}"
    with (root / "uv.lock").open("rb") as file:
        (locked,) = [p for p in tomllib.load(file)["package"] if p["name"] == "lm-eval"]
    assert locked["version"] == LM_EVAL_VERSION
