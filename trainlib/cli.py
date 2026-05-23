from __future__ import annotations

import argparse
import sys

import trainlib
from trainlib.config import load_config, validate_config
from trainlib.recipes import recipe_registry


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trainlib",
        description="trainlib — An opinionated LLM training and fine-tuning library",
    )
    parser.add_argument(
        "--version", action="version", version=f"trainlib {trainlib.__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser("train", help="Run a training recipe")
    train_parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )
    train_parser.add_argument(
        "--override", action="append", default=[],
        help="Config overrides in dot notation (e.g., model.name=my-model)",
    )
    train_parser.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint"
    )
    train_parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and print config without training",
    )

    list_parser = subparsers.add_parser("list", help="List registered components")
    list_parser.add_argument(
        "type", nargs="?", default=None,
        help="Component type: recipes, formats, metrics, rewards",
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate a model")
    eval_parser.add_argument("--model", required=True, help="Model path or HF Hub name")
    eval_parser.add_argument("--benchmarks", default=None, help="Comma-separated benchmarks (e.g., mmlu,gsm8k)")
    eval_parser.add_argument("--metrics", default=None, help="Comma-separated metrics (e.g., loss,perplexity)")
    eval_parser.add_argument("--dataset", default=None, help="Path to evaluation dataset")
    eval_parser.add_argument("--num-fewshot", type=int, default=None, help="Number of few-shot examples")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "train":
        return _handle_train(args)

    if args.command == "list":
        return _handle_list(args)

    if args.command == "eval":
        return _handle_eval(args)

    return 0


def _handle_train(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config, overrides=args.override or None)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    try:
        validate_config(config)
    except Exception as e:
        print(f"Validation error: {e}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(config.model_dump_json(indent=2))
        return 0

    recipe_fn = recipe_registry.get(config.recipe)
    recipe_fn(config=config)
    return 0


def _handle_list(args: argparse.Namespace) -> int:
    from trainlib.data import format_registry
    from trainlib.eval.metrics import metric_registry
    from trainlib.recipes.align.rewards import reward_registry

    registries = {
        "recipes": recipe_registry,
        "formats": format_registry,
        "metrics": metric_registry,
        "rewards": reward_registry,
    }

    if args.type is None:
        for name, registry in registries.items():
            items = registry.list()
            print(f"{name.capitalize()}: {', '.join(items)}")
        return 0

    if args.type not in registries:
        print(f"Unknown type: '{args.type}'. Available: {', '.join(registries)}", file=sys.stderr)
        return 1

    registry = registries[args.type]
    for item in registry.list():
        print(f"  {item}")
    return 0


def _handle_eval(args: argparse.Namespace) -> int:
    if args.benchmarks:
        from trainlib.eval.benchmarks import benchmark_evaluate

        benchmarks = [b.strip() for b in args.benchmarks.split(",")]
        results = benchmark_evaluate(
            model=args.model,
            benchmarks=benchmarks,
            num_fewshot=args.num_fewshot,
        )

        for task_name, task_results in results.items():
            print(f"\n{task_name}:")
            for metric_name, value in task_results.items():
                print(f"  {metric_name}: {value}")

        return 0

    if args.dataset:
        import json
        from pathlib import Path
        from trainlib.eval import evaluate

        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"Error: Dataset not found: {args.dataset}", file=sys.stderr)
            return 1

        data = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
        metrics = [m.strip() for m in args.metrics.split(",")] if args.metrics else None

        results = evaluate(model=args.model, dataset=data, metrics=metrics)

        for metric_name, value in results.items():
            print(f"{metric_name}: {value:.4f}")

        return 0

    print("Error: Provide --benchmarks or --dataset", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
