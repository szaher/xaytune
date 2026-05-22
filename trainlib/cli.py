from __future__ import annotations

import argparse
import sys

import trainlib
from trainlib.config import load_config, validate_config


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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "train":
        return _handle_train(args)

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

    print(f"Training with recipe={config.recipe}, method={config.method}")
    print(f"Model: {config.model.name}")
    print("Training loop not yet implemented — use --dry-run to validate config.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
