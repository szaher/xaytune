"""What the real resolver binds, against lm-eval's own task registry. No network.

Needs lm-eval 0.4.13 (the ``eval`` extra) and is skipped without it. The Hub is
replaced by a function that names a commit, so these read lm-eval's task
definitions -- installed files -- and nothing else. Whether the dataset really
loads at that commit is the example run, not a unit test.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("lm_eval")

from xaytune.core.domain.evaluation import EvaluationSpec, EvaluatorSpec  # noqa: E402
from xaytune.evaluation import UnsupportedEvaluationError  # noqa: E402
from xaytune.evaluation.lmeval import (  # noqa: E402
    HubTaskResolver,
    LMEvalEvaluator,
    UnresolvableTaskError,
    load_task_definition,
)
from xaytune.workers.eval_lmeval import TaskChangedError, measure  # noqa: E402
from xaytune.workers.eval_lmeval_schema import (  # noqa: E402
    LM_EVAL_VERSION,
    LMEvalMeasure,
    LMEvalRealization,
    LMEvalWorkerConfig,
)

COMMIT = "5" * 40


class _Hub:
    """Names a commit for any dataset, and remembers what it was asked."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.asked: list[tuple[str, str | None]] = []

    def __call__(self, repository: str, revision: str | None) -> str:
        self.asked.append((repository, revision))
        if self.error is not None:
            raise self.error
        return COMMIT


def _spec(task: str, num_fewshot: int = 0) -> EvaluationSpec:
    config: dict[str, Any] = {
        "task": task,
        "num_fewshot": num_fewshot,
        "batch_size": 4,
        "precision": "fp32",
    }
    return EvaluationSpec(evaluator=EvaluatorSpec(name="lm-eval", config=config))


def test_a_registered_multiple_choice_task_binds_to_its_dataset_at_a_commit() -> None:
    hub = _Hub()
    binding = HubTaskResolver(hub).bind("arc_easy", 0)

    assert hub.asked == [("allenai/ai2_arc", None)], "its default branch, resolved"
    assert binding.dataset_path == "allenai/ai2_arc"
    assert binding.dataset_name == "ARC-Easy"
    assert binding.dataset_revision == COMMIT
    assert binding.output_type == "multiple_choice"
    assert binding.metrics == ("acc", "acc_norm")
    assert binding.lm_eval_version == LM_EVAL_VERSION
    assert binding.task_config_digest == load_task_definition("arc_easy").digest


def test_the_definition_digest_is_stable() -> None:
    assert load_task_definition("hellaswag").digest == load_task_definition("hellaswag").digest
    assert load_task_definition("hellaswag").digest != load_task_definition("arc_easy").digest


def test_function_references_do_not_carry_the_install_location() -> None:
    import lm_eval

    config = load_task_definition("hellaswag").config
    assert config["process_docs"] == "lm_eval/tasks/hellaswag/utils.process_docs"
    assert lm_eval.__file__.rsplit("/", 1)[0] not in repr(config)


@pytest.mark.parametrize(
    ("task", "reason"),
    [
        ("not_a_task_anywhere", "is not a task registered"),
        ("mmlu", "is a group, not a task"),
    ],
    ids=["unknown", "group"],
)
def test_what_is_not_one_registered_task_cannot_be_bound(task: str, reason: str) -> None:
    with pytest.raises(UnresolvableTaskError, match=reason):
        HubTaskResolver(_Hub()).bind(task, 0)


def test_a_task_that_runs_generated_code_is_not_even_looked_up() -> None:
    hub = _Hub()
    with pytest.raises(UnresolvableTaskError, match="unsafe_code"):
        HubTaskResolver(hub).bind("humaneval", 0)
    assert hub.asked == []


def test_a_dataset_the_hub_cannot_pin_is_refused() -> None:
    with pytest.raises(UnresolvableTaskError, match="cannot be pinned to a commit: offline"):
        HubTaskResolver(_Hub(OSError("offline"))).bind("arc_easy", 0)


@pytest.mark.parametrize(
    ("task", "reason"),
    [
        ("gsm8k", "'generate_until'"),
        ("lambada_openai", "reports ['perplexity']"),
    ],
    ids=["generation", "unsupported-metric"],
)
def test_a_task_accepted_by_name_is_refused_once_resolved(task: str, reason: str) -> None:
    """The case for the second supports(): the name alone gives no reason to refuse."""
    evaluator = LMEvalEvaluator(HubTaskResolver(_Hub()))
    declared = _spec(task)
    assert evaluator.supports(declared)

    resolved = evaluator.resolve(declared)
    support = evaluator.supports(resolved)
    assert not support
    assert reason in " | ".join(support.reasons)


def test_resolving_without_lm_eval_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real = builtins.__import__

    def without_lm_eval(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "lm_eval" or name.startswith("lm_eval."):
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_lm_eval)
    with pytest.raises(UnsupportedEvaluationError, match=r"xaytune\[eval\]"):
        LMEvalEvaluator(HubTaskResolver(_Hub())).resolve(_spec("arc_easy"))


def test_the_worker_refuses_a_definition_that_is_not_the_bound_one(tmp_path: Any) -> None:
    """Checked before any model is loaded: tmp_path holds no model at all."""
    binding = HubTaskResolver(_Hub()).bind("arc_easy", 0)
    config = LMEvalWorkerConfig(
        api_version="xaytune.lm-eval/v1alpha1",
        model_uri=str(tmp_path),
        binding=binding.model_copy(update={"task_config_digest": "sha256:" + "0" * 64}),
        measure=LMEvalMeasure(batch_size=1, precision="fp32", limit=1),
        realization=LMEvalRealization(
            evaluator_name="lm-eval", evaluator_version="0.1.0", seed=7, output_dir=str(tmp_path)
        ),
    )
    with pytest.raises(TaskChangedError, match="another benchmark"):
        measure(config)
