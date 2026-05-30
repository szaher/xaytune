from __future__ import annotations

import argparse
import sys
from typing import Any

import xaytune
from xaytune.config import load_config, validate_config
from xaytune.recipes import recipe_registry


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xaytune",
        description="xaytune — An opinionated LLM training and fine-tuning library",
    )
    parser.add_argument("--version", action="version", version=f"xaytune {xaytune.__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser("train", help="Run a training recipe")
    train_parser.add_argument("--config", required=True, help="Path to YAML config file")
    train_parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config overrides in dot notation (e.g., model.name=my-model)",
    )
    train_parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    train_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print config without training",
    )

    list_parser = subparsers.add_parser("list", help="List registered components")
    list_parser.add_argument(
        "type",
        nargs="?",
        default=None,
        help="Component type: recipes, formats, metrics, rewards",
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate a model")
    eval_parser.add_argument("--model", required=True, help="Model path or HF Hub name")
    eval_parser.add_argument(
        "--benchmarks", default=None, help="Comma-separated benchmarks (e.g., mmlu,gsm8k)"
    )
    eval_parser.add_argument(
        "--metrics", default=None, help="Comma-separated metrics (e.g., loss,perplexity)"
    )
    eval_parser.add_argument("--dataset", default=None, help="Path to evaluation dataset")
    eval_parser.add_argument(
        "--num-fewshot", type=int, default=None, help="Number of few-shot examples"
    )

    export_parser = subparsers.add_parser("export", help="Export and convert models")
    export_subparsers = export_parser.add_subparsers(dest="export_command", help="Export commands")

    merge_parser = export_subparsers.add_parser("merge", help="Merge LoRA adapters into base model")
    merge_parser.add_argument("--checkpoint", required=True, help="Path to LoRA checkpoint")
    merge_parser.add_argument("--output", required=True, help="Output directory for merged model")

    gguf_parser = export_subparsers.add_parser("gguf", help="Convert model to GGUF format")
    gguf_parser.add_argument("--model", required=True, help="Path to model directory")
    gguf_parser.add_argument("--output", required=True, help="Output GGUF file path")
    gguf_parser.add_argument(
        "--quant", default="Q4_K_M", help="Quantization type (default: Q4_K_M)"
    )

    push_parser = export_subparsers.add_parser("push", help="Push model to Hugging Face Hub")
    push_parser.add_argument("--model", required=True, help="Path to model directory")
    push_parser.add_argument(
        "--repo", required=True, help="HF Hub repo (e.g., username/model-name)"
    )

    model_merge_parser = export_subparsers.add_parser(
        "model-merge", help="Merge models using weight interpolation (linear, slerp, ties, dare)"
    )
    model_merge_parser.add_argument(
        "--models", nargs="+", required=True, help="Paths to models to merge"
    )
    model_merge_parser.add_argument(
        "--method",
        required=True,
        choices=["linear", "slerp", "ties", "dare"],
        help="Merge method",
    )
    model_merge_parser.add_argument("--output", required=True, help="Output directory")
    model_merge_parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Per-model weights for linear merge",
    )
    model_merge_parser.add_argument(
        "--t", type=float, default=0.5, help="SLERP interpolation factor (default: 0.5)"
    )
    model_merge_parser.add_argument(
        "--base-model", default=None, help="Base model path (required for ties/dare)"
    )
    model_merge_parser.add_argument(
        "--density",
        type=float,
        default=0.5,
        help="Sparsification density for ties/dare (default: 0.5)",
    )
    model_merge_parser.add_argument(
        "--weight",
        type=float,
        default=1.0,
        help="Task vector scaling for ties/dare (default: 1.0)",
    )
    model_merge_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for dare (default: 42)"
    )

    compare_parser = subparsers.add_parser("compare", help="Compare two models side-by-side")
    compare_parser.add_argument("models", nargs="+", help="Model paths to compare (exactly 2)")
    compare_parser.add_argument("--benchmarks", required=True, help="Comma-separated benchmarks")
    compare_parser.add_argument(
        "--num-fewshot", type=int, default=None, help="Number of few-shot examples"
    )

    lr_find_parser = subparsers.add_parser("lr-find", help="Find optimal learning rate")
    lr_find_parser.add_argument("--config", required=True, help="Path to YAML config file")
    lr_find_parser.add_argument(
        "--start-lr", type=float, default=1e-7, help="Start LR (default: 1e-7)"
    )
    lr_find_parser.add_argument("--end-lr", type=float, default=1.0, help="End LR (default: 1.0)")
    lr_find_parser.add_argument(
        "--num-iterations", type=int, default=100, help="Number of steps (default: 100)"
    )
    lr_find_parser.add_argument(
        "--smoothing-factor", type=float, default=0.05, help="EMA smoothing (default: 0.05)"
    )
    lr_find_parser.add_argument("--output", default=None, help="Save results to JSON file")

    studio_parser = subparsers.add_parser("studio", help="Launch Training Studio web UI")
    studio_parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    studio_parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    studio_parser.add_argument("--share", action="store_true", help="Create public Gradio link")

    launch_parser = subparsers.add_parser("launch", help="Launch distributed training via torchrun")
    launch_parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=None,
        help="Number of processes per node (default: auto-detect GPUs)",
    )
    launch_parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes")
    launch_parser.add_argument("--config", required=True, help="Path to training config file")
    launch_parser.add_argument(
        "--override", action="append", default=[], help="Config overrides (key=value)"
    )

    # Data preparation subcommands
    data_parser = subparsers.add_parser("data", help="Data preparation toolkit")
    data_subparsers = data_parser.add_subparsers(dest="data_command", help="Data commands")

    dedup_parser = data_subparsers.add_parser("deduplicate", help="Remove duplicate samples")
    dedup_parser.add_argument("input", help="Path to input JSONL file")
    dedup_parser.add_argument("-o", "--output", required=True, help="Output file path")
    dedup_parser.add_argument(
        "--method",
        default="both",
        choices=["exact", "minhash", "both"],
        help="Dedup method (default: both)",
    )
    dedup_parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="MinHash threshold (default: 0.85)",
    )
    dedup_parser.add_argument(
        "--field",
        default=None,
        help="Field to compare (auto-detected if omitted)",
    )

    filter_parser = data_subparsers.add_parser("filter", help="Filter samples by quality criteria")
    filter_parser.add_argument("input", help="Path to input JSONL file")
    filter_parser.add_argument("-o", "--output", required=True, help="Output file path")
    filter_parser.add_argument(
        "--min-chars",
        type=int,
        default=None,
        help="Minimum character count",
    )
    filter_parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Maximum character count",
    )
    filter_parser.add_argument(
        "--language",
        default=None,
        help="Keep only this language (e.g., en)",
    )
    filter_parser.add_argument(
        "--drop-regex",
        default=None,
        help="Drop samples matching this regex",
    )
    filter_parser.add_argument(
        "--field",
        default=None,
        help="Field to filter on (auto-detected if omitted)",
    )

    convert_parser = data_subparsers.add_parser("convert", help="Convert between data formats")
    convert_parser.add_argument("input", help="Path to input file")
    convert_parser.add_argument("-o", "--output", required=True, help="Output file path")
    convert_parser.add_argument("--from", dest="source_format", required=True, help="Source format")
    convert_parser.add_argument("--to", dest="target_format", required=True, help="Target format")
    convert_parser.add_argument(
        "--field-map",
        default=None,
        help="Field mapping (e.g., question=instruction,answer=output)",
    )

    gen_parser = data_subparsers.add_parser("generate", help="Generate synthetic training data")
    gen_parser.add_argument(
        "--mode",
        required=True,
        choices=["augment", "distill", "evolve"],
        help="Generation mode",
    )
    gen_parser.add_argument("--seed", default=None, help="Path to seed examples JSONL")
    gen_parser.add_argument("--topic", default=None, help="Topic for distill mode")
    gen_parser.add_argument("-n", type=int, default=10, help="Number of examples to generate")
    gen_parser.add_argument("--rounds", type=int, default=1, help="Evolution rounds (evolve mode)")
    gen_parser.add_argument("--format", default="alpaca", help="Output format (default: alpaca)")
    gen_parser.add_argument("--model", required=True, help="LLM model name")
    gen_parser.add_argument("--api-base", default=None, help="API base URL")
    gen_parser.add_argument("-o", "--output", required=True, help="Output file path")

    pipeline_parser = data_subparsers.add_parser(
        "pipeline",
        help="Run a prep pipeline from YAML config",
    )
    pipeline_parser.add_argument("config", help="Path to pipeline YAML config")

    return parser


def main(argv: list[str] | None = None) -> int:
    from xaytune.plugins import discover_plugins

    discover_plugins()

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

    if args.command == "export":
        return _handle_export(args)

    if args.command == "compare":
        return _handle_compare(args)

    if args.command == "lr-find":
        return _handle_lr_find(args)

    if args.command == "studio":
        return _handle_studio(args)

    if args.command == "launch":
        return _handle_launch(args)

    if args.command == "data":
        return _handle_data(args)

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

    resume_from = None
    if args.resume:
        from xaytune.trainer.checkpointing import find_latest_checkpoint

        resume_from = find_latest_checkpoint(config.output.dir)
        if resume_from is None:
            print(
                f"Error: No checkpoint found in '{config.output.dir}'",
                file=sys.stderr,
            )
            return 1

    recipe_fn = recipe_registry.get(config.recipe)
    recipe_fn(config=config, resume_from=resume_from)
    return 0


def _handle_list(args: argparse.Namespace) -> int:
    from xaytune.data import format_registry
    from xaytune.eval.metrics import metric_registry
    from xaytune.recipes.align.rewards import reward_registry

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
        from xaytune.eval.benchmarks import benchmark_evaluate

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

        from xaytune.eval import evaluate

        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"Error: Dataset not found: {args.dataset}", file=sys.stderr)
            return 1

        data = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
        metrics = [m.strip() for m in args.metrics.split(",")] if args.metrics else None

        results = evaluate(model=args.model, dataset=data, metrics=metrics)  # type: ignore[assignment]

        for metric_name, value in results.items():
            print(f"{metric_name}: {value:.4f}")

        return 0

    print("Error: Provide --benchmarks or --dataset", file=sys.stderr)
    return 1


def _handle_export(args: argparse.Namespace) -> int:
    if args.export_command is None:
        print("Error: Specify an export command: merge, model-merge, gguf, push", file=sys.stderr)
        return 1

    if args.export_command == "merge":
        return _export_merge(args)

    if args.export_command == "gguf":
        return _export_gguf(args)

    if args.export_command == "push":
        return _export_push(args)

    if args.export_command == "model-merge":
        return _export_model_merge(args)

    return 1


def _export_merge(args: argparse.Namespace) -> int:
    from xaytune.export.merge import merge

    try:
        merge(args.checkpoint, save_to=args.output)
        print(f"Merged model saved to {args.output}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _export_gguf(args: argparse.Namespace) -> int:
    from xaytune.export.gguf import to_gguf

    try:
        to_gguf(args.model, output=args.output, quantization=args.quant)
        print(f"GGUF model saved to {args.output}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _export_push(args: argparse.Namespace) -> int:
    from xaytune.export.hub import push_to_hub

    try:
        push_to_hub(args.model, repo=args.repo)
        print(f"Model pushed to {args.repo}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _export_model_merge(args: argparse.Namespace) -> int:
    from xaytune.export.model_merge import model_merge

    try:
        result = model_merge(
            models=args.models,
            method=args.method,
            output=args.output,
            weights=args.weights,
            t=args.t,
            base_model=args.base_model,
            density=args.density,
            weight=args.weight,
            seed=args.seed,
        )
        print(result.summary())
        return 0
    except (ValueError, ImportError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _handle_launch(args: argparse.Namespace) -> int:
    import subprocess

    nproc = args.nproc_per_node
    if nproc is None:
        import torch

        if torch.cuda.is_available():
            nproc = torch.cuda.device_count()
        else:
            nproc = 1

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc-per-node={nproc}",
        f"--nnodes={args.nnodes}",
        "-m",
        "xaytune.cli",
        "train",
        "--config",
        args.config,
    ]
    for override in args.override:
        cmd.extend(["--override", override])

    try:
        result = subprocess.run(cmd)
        return result.returncode
    except FileNotFoundError:
        print(
            "Error: torchrun not found. Install PyTorch with distributed support.",
            file=sys.stderr,
        )
        return 1


def _handle_compare(args: argparse.Namespace) -> int:
    if len(args.models) != 2:
        print("Error: compare requires exactly 2 models", file=sys.stderr)
        return 1

    from xaytune.eval.benchmarks import benchmark_evaluate

    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    model_a, model_b = args.models

    results_a = benchmark_evaluate(
        model=model_a,
        benchmarks=benchmarks,
        num_fewshot=args.num_fewshot,
    )
    results_b = benchmark_evaluate(
        model=model_b,
        benchmarks=benchmarks,
        num_fewshot=args.num_fewshot,
    )

    all_tasks = sorted(set(results_a) | set(results_b))
    header = f"{'Benchmark':<20} {'Metric':<25} {model_a:<15} {model_b:<15}"
    print(header)
    print("-" * len(header))

    for task in all_tasks:
        metrics_a = results_a.get(task, {})
        metrics_b = results_b.get(task, {})
        all_metrics = sorted(set(metrics_a) | set(metrics_b))
        for metric in all_metrics:
            val_a = metrics_a.get(metric, "N/A")
            val_b = metrics_b.get(metric, "N/A")
            if isinstance(val_a, float):
                val_a = f"{val_a:.4f}"
            if isinstance(val_b, float):
                val_b = f"{val_b:.4f}"
            print(f"{task:<20} {metric:<25} {val_a:<15} {val_b:<15}")

    return 0


def _handle_studio(args: argparse.Namespace) -> int:
    try:
        from xaytune.studio.server import launch
    except ImportError:
        print(
            "Error: Install studio dependencies: pip install xaytune[studio]",
            file=sys.stderr,
        )
        return 1
    launch(host=args.host, port=args.port, share=args.share)
    return 0


def _handle_lr_find(args: argparse.Namespace) -> int:
    import json

    from xaytune.recipes.base import setup_training
    from xaytune.trainer.lr_finder import lr_find

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    components = setup_training(config)
    result = lr_find(
        components.model,
        components.train_dataloader,
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        num_iterations=args.num_iterations,
        smoothing_factor=args.smoothing_factor,
    )

    print(f"Suggested LR: {result.suggested_lr}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Results saved to {args.output}")

    return 0


def _handle_data(args: argparse.Namespace) -> int:
    if args.data_command is None:
        print(
            "Error: Specify a data command: deduplicate, filter, convert, generate, pipeline",
            file=sys.stderr,
        )
        return 1

    if args.data_command == "deduplicate":
        from xaytune.data.prep import deduplicate

        result = deduplicate(
            args.input,
            method=args.method,
            threshold=args.threshold,
            field=args.field,
        )
        result.save(args.output)
        print(result.report.summary())
        return 0

    if args.data_command == "filter":
        from xaytune.data.prep import filter_dataset

        filters = []
        if args.min_chars is not None or args.max_chars is not None:
            f: dict[str, Any] = {"type": "length"}
            if args.min_chars is not None:
                f["min_chars"] = args.min_chars
            if args.max_chars is not None:
                f["max_chars"] = args.max_chars
            filters.append(f)
        if args.language:
            filters.append({"type": "language", "keep": [args.language]})
        if args.drop_regex:
            filters.append({"type": "regex", "drop_pattern": args.drop_regex})

        if not filters:
            print(
                "Error: Specify at least one filter "
                "(--min-chars, --max-chars, --language, --drop-regex)",
                file=sys.stderr,
            )
            return 1

        result = filter_dataset(args.input, filters=filters, field=args.field)
        result.save(args.output)
        print(result.report.summary())
        return 0

    if args.data_command == "convert":
        from xaytune.data.prep import convert

        field_map = None
        if args.field_map:
            field_map = dict(pair.split("=") for pair in args.field_map.split(","))

        result = convert(
            args.input,
            output=args.output,
            source_format=args.source_format,
            target_format=args.target_format,
            field_map=field_map,
        )
        print(result.report.summary())
        return 0

    if args.data_command == "generate":
        from xaytune.data.prep import generate

        try:
            result = generate(
                mode=args.mode,
                seed=args.seed,
                topic=args.topic,
                n=args.n,
                rounds=args.rounds,
                format=args.format,
                model=args.model,
                api_base=args.api_base,
            )
            result.save(args.output)
            print(result.report.summary())
            return 0
        except (ValueError, ImportError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    if args.data_command == "pipeline":
        from xaytune.data.prep import pipeline

        try:
            result = pipeline(config=args.config)
            print(result.report.summary())
            return 0
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
