"""Train a candidate, run an lm-eval benchmark task on it, and decide on the accuracy.

    python examples/control_plane/06_train_and_benchmark.py \\
        --model /abs/path/to/hf-model-dir \\
        --dataset /abs/path/to/train.jsonl \\
        --task arc_easy --limit 20 \\
        --target 0.3            # the accuracy that counts as good enough

Needs the ``eval`` extra (``pip install "xaytune[eval]==1.0.0a1"``, or from a
clone ``uv sync --locked --extra eval``) and access to the Hugging Face Hub:
the task's dataset is downloaded when the evaluation runs.

The task name is **resolved at submission**: its definition in the installed
lm-eval, by digest, and its dataset on the Hub, by commit. That binding is
what the record holds and the fingerprint names, so a later run of
"``arc_easy``" against a moved dataset branch or another lm-eval release is a
different evaluation, not a silently different number under the same one.
The resolved binding is printed below.

Only log-likelihood tasks that report ``acc`` and ``acc_norm`` are accepted --
``arc_easy``, ``hellaswag``, ``piqa``, ``winogrande`` and the like. A
generation task such as ``gsm8k`` is refused at submission, with the reason,
before anything trains.

The objective maximizes ``acc`` with ``--target`` as the threshold, decided
by the built-in threshold engine as in example 05.
"""

from __future__ import annotations

import argparse
import asyncio

from sft_experiment import add_arguments, experiment, state_path

from xaytune.core.domain.evaluation import EvaluationSpec, EvaluatorSpec
from xaytune.core.domain.objective import Objective, ObjectiveMetric
from xaytune.experiment import EmbeddedControllerHost


def evaluation(args: argparse.Namespace) -> EvaluationSpec:
    """Which task, how many shots, how batched, on how many documents. Nothing defaulted."""
    config: dict[str, object] = {
        "task": args.task,
        "num_fewshot": args.num_fewshot,
        "batch_size": 4,
        "precision": "fp32",
    }
    if args.limit is not None:
        config["limit"] = args.limit
    return EvaluationSpec(evaluator=EvaluatorSpec(name="lm-eval", config=config))


async def run(args: argparse.Namespace) -> None:
    objective = Objective(
        primary=ObjectiveMetric(name="acc", direction="maximize"), target=args.target
    )
    spec = experiment(args).model_copy(
        update={"objective": objective, "evaluation": evaluation(args)}
    )
    host = EmbeddedControllerHost(state_path(args))
    try:
        handle = await host.submit(spec)
        recorded = host.repository.aggregates.load_experiment(str(handle.experiment_id))
        assert recorded.evaluation is not None
        binding = recorded.evaluation.evaluator.config["binding"]
        print(f"submitted {handle.experiment_id}; training, benchmarking, deciding")
        print(
            f"  bound {binding['task']} v{binding['task_version']} under lm-eval "
            f"{binding['lm_eval_version']}: {binding['dataset_path']} "
            f"@ {binding['dataset_revision'][:12]}, {binding['num_fewshot']}-shot"
        )
        result = await handle.wait()

        print(f"\nexperiment {result.status.value}; next stage: {result.next_stage}")
        for node in result.nodes:
            print(f"candidate {node.node_id}: {node.status.value}")
            for evaluated in node.evaluations:
                print(f"  evaluation {evaluated.evaluation_run_id}: {evaluated.status.value}")
                if evaluated.result is None:
                    continue
                for metric in evaluated.result.metrics:
                    error = (
                        "" if metric.standard_error is None else f" ± {metric.standard_error:.4f}"
                    )
                    print(
                        f"    {metric.name:<9} {metric.value:.4f}{error}"
                        f"   ({metric.sample_count} documents, seed {metric.seed})"
                    )
                for report in evaluated.result.artifacts:
                    print(f"    report: {report.uri}")
            for decision in host.repository.aggregates.decisions_for_node(str(node.node_id)):
                print(f"  decision: {decision.outcome.value} -- {decision.reason}")
    finally:
        await host.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(parser)
    parser.add_argument("--task", default="arc_easy", help="a registered lm-eval task")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="score only the first N documents")
    parser.add_argument(
        "--target", type=float, default=None, help="accuracy at or above which it succeeds"
    )
    asyncio.run(run(parser.parse_args()))
