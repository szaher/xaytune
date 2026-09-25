"""Train a candidate, evaluate the model it produced, and decide: the whole loop.

    python examples/control_plane/05_train_and_evaluate.py \\
        --model /abs/path/to/hf-model-dir \\
        --dataset /abs/path/to/train.jsonl \\
        --held-out /abs/path/to/held-out.jsonl \\
        --target 2.5            # the held-out loss that counts as good enough

Install the control plane first: ``pip install "xaytune==1.0.0a1"``, or from
a clone with ``uv sync --locked``.

After training succeeds, the host evaluates the trained model with the
built-in ``native`` evaluator: next-token loss, perplexity and token accuracy
on the held-out file, measured by a worker process through the same runtime,
operation journal and telemetry as training. The result is recorded durably,
and then decided against the objective -- minimize loss, with ``--target`` as
the threshold -- by the built-in threshold decision engine. The decision is
recorded with what it causes: the candidate ``COMPLETED`` and the experiment
``SUCCEEDED`` (naming it as its best candidate) if the loss is at or below the
target, else ``REJECTED`` and ``FAILED``.

Without ``--target`` there is nothing to decide against -- "minimize loss" is
not a statement of what is good enough -- so the candidate stays ``DECIDING``,
with the reason recorded, and ``next_stage`` says ``"decision"``.

The held-out file is pinned by the digest of its contents when the experiment
is submitted. If it changes before the evaluation runs, the evaluation fails
rather than measuring different data under the same name.
"""

from __future__ import annotations

import argparse
import asyncio

from sft_experiment import add_arguments, experiment, state_path

from xaytune.core.domain.evaluation import EvaluationSpec, EvaluatorSpec
from xaytune.core.domain.objective import Objective, ObjectiveMetric
from xaytune.evaluation.native import local_dataset
from xaytune.experiment import EmbeddedControllerHost


def evaluation(args: argparse.Namespace) -> EvaluationSpec:
    """How the trained model is measured. Every value that changes a number is declared."""
    return EvaluationSpec(
        evaluator=EvaluatorSpec(
            name="native",
            config={
                "format": "text",
                "max_seq_length": args.max_seq_length,
                "batch_size": 2,
                "metrics": ["loss", "perplexity", "token_accuracy"],
                "precision": "fp32",
            },
        ),
        # Pinned here, before anything is recorded: the evaluation names these bytes.
        dataset=local_dataset(args.held_out),
    )


async def run(args: argparse.Namespace) -> None:
    objective = Objective(
        primary=ObjectiveMetric(name="loss", direction="minimize"), target=args.target
    )
    spec = experiment(args).model_copy(
        update={"objective": objective, "evaluation": evaluation(args)}
    )
    host = EmbeddedControllerHost(state_path(args))
    try:
        handle = await host.submit(spec)
        print(f"submitted {handle.experiment_id}; training, evaluating, deciding")
        result = await handle.wait()

        print(f"\nexperiment {result.status.value}; next stage: {result.next_stage}")
        for node in result.nodes:
            print(f"candidate {node.node_id}: {node.status.value}")
            for evaluated in node.evaluations:
                print(f"  evaluation {evaluated.evaluation_run_id}: {evaluated.status.value}")
                if evaluated.result is None:
                    continue
                for metric in evaluated.result.metrics:
                    print(
                        f"    {metric.name:<15} {metric.value:.4f}"
                        f"   ({metric.sample_count} texts, seed {metric.seed})"
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
    parser.add_argument("--held-out", required=True, help="local JSONL with a 'text' field")
    parser.add_argument(
        "--target", type=float, default=None, help="held-out loss at or below which it succeeds"
    )
    asyncio.run(run(parser.parse_args()))
