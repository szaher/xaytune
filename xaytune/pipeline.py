"""Multi-stage training pipeline runner.

Chains training recipes, evaluation, and export steps into a single
execution.  Each stage auto-inherits the previous stage's output as
its model input unless overridden.

Example::

    result = run_pipeline(PipelineConfig(
        name="sft-to-deploy",
        stages=[
            StageConfig(name="sft", recipe="finetune", ...),
            StageConfig(name="merge", export="merge"),
            StageConfig(name="dpo", recipe="align", method="dpo", ...),
            StageConfig(name="eval", eval=EvalStageConfig(metrics=["loss"])),
        ],
    ))
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from xaytune.pipeline_schema import (
    PipelineConfig,
    PipelineResult,
    StageConfig,
    StageResult,
)

logger = logging.getLogger(__name__)


def run_pipeline(
    config: PipelineConfig,
    *,
    resume_from: str | None = None,
    dry_run: bool = False,
) -> PipelineResult:
    """Execute a multi-stage training pipeline.

    Args:
        config: Pipeline configuration with ordered stages.
        resume_from: Skip all stages before this stage name.
        dry_run: Print the execution plan without running anything.

    Returns:
        :class:`PipelineResult` with per-stage results and metrics.
    """
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    state_path = output_root / "pipeline_state.json"

    if dry_run:
        _print_plan(config)
        return PipelineResult(name=config.name, completed=False)

    skipping = resume_from is not None
    results: dict[str, StageResult] = {}
    prev_output: str | None = None

    if resume_from and state_path.exists():
        saved = json.loads(state_path.read_text())
        for name, sr in saved.get("stages", {}).items():
            if name == resume_from:
                break
            results[name] = StageResult(**sr)
            prev_output = sr.get("output")

    total = len(config.stages)
    for i, stage in enumerate(config.stages):
        if skipping:
            if stage.name == resume_from:
                skipping = False
            else:
                logger.info(f"[{i + 1}/{total}] Skipping stage: {stage.name}")
                if stage.name in results and results[stage.name].output:
                    prev_output = results[stage.name].output
                continue

        model_path = stage.model_name or prev_output
        stage_output = str(output_root / stage.name)

        logger.info(f"[{i + 1}/{total}] Running stage: {stage.name}")
        start = time.time()

        try:
            if stage.recipe:
                result = _run_train_stage(stage, model_path, stage_output)
                prev_output = result.output
            elif stage.export:
                result = _run_export_stage(stage, model_path, stage_output)
                if result.output:
                    prev_output = result.output
            elif stage.eval:
                result = _run_eval_stage(stage, model_path)
            else:
                raise ValueError(
                    f"Stage '{stage.name}' must have one of: recipe, export, eval"
                )
        except Exception as e:
            logger.error(f"Stage '{stage.name}' failed: {e}")
            results[stage.name] = StageResult(
                type="error", status="failed", error=str(e)
            )
            _save_state(state_path, config.name, results)
            return PipelineResult(
                name=config.name, stages=results, completed=False
            )

        elapsed = time.time() - start
        logger.info(
            f"[{i + 1}/{total}] Stage '{stage.name}' completed in {elapsed:.1f}s"
        )
        results[stage.name] = result
        _save_state(state_path, config.name, results)

    logger.info(f"Pipeline '{config.name}' completed — {total} stages")
    return PipelineResult(name=config.name, stages=results, completed=True)


def _run_train_stage(
    stage: StageConfig, model_path: str | None, output_dir: str
) -> StageResult:
    from xaytune.config.schema import (
        DataConfig,
        ModelConfig,
        OutputConfig,
        TrainConfig,
    )
    from xaytune.recipes.align import align
    from xaytune.recipes.finetune import finetune
    from xaytune.recipes.pretrain import pretrain

    if model_path is None:
        raise ValueError(
            f"Stage '{stage.name}': no model specified and no previous stage output"
        )

    train_config = TrainConfig(
        recipe=stage.recipe,  # type: ignore[arg-type]
        method=stage.method or ("full" if stage.recipe != "align" else "dpo"),
        model=ModelConfig(name=model_path),
        data=stage.data or DataConfig(path="", format="text"),
        trainer=stage.trainer or TrainerConfig(),
        lora=stage.lora or LoraConfig(),
        output=OutputConfig(dir=output_dir),
        method_params=stage.method_params,
    )

    from xaytune.config.schema import LoraConfig, TrainerConfig

    recipe_fn = {"finetune": finetune, "pretrain": pretrain, "align": align}
    fn = recipe_fn.get(stage.recipe)  # type: ignore[arg-type]
    if fn is None:
        raise ValueError(f"Unknown recipe: {stage.recipe}")

    state = fn(config=train_config)
    return StageResult(
        type="train",
        output=output_dir,
        metrics={k: v for k, v in state.metrics.items()},
    )


def _run_export_stage(
    stage: StageConfig, model_path: str | None, output_dir: str
) -> StageResult:
    if model_path is None:
        raise ValueError(
            f"Stage '{stage.name}': no model to export (no previous stage output)"
        )

    action = stage.export
    if action == "merge":
        from xaytune.export import merge

        save_to = stage.save_to or output_dir
        merge(model_path, save_to=save_to)
        return StageResult(type="export", output=save_to)

    elif action == "gguf":
        from xaytune.export import to_gguf

        out_file = stage.save_to or f"{output_dir}/model.gguf"
        to_gguf(
            model_path,
            output=out_file,
            quantization=stage.quantization or "Q4_K_M",
        )
        return StageResult(type="export", output=out_file)

    elif action == "push_to_hub":
        from xaytune.export import push_to_hub

        if not stage.repo:
            raise ValueError(
                f"Stage '{stage.name}': push_to_hub requires 'repo' field"
            )
        push_to_hub(model_path, repo=stage.repo)
        return StageResult(type="export", output=stage.repo)

    else:
        raise ValueError(f"Unknown export action: {action}")


def _run_eval_stage(
    stage: StageConfig, model_path: str | None
) -> StageResult:
    if model_path is None:
        raise ValueError(
            f"Stage '{stage.name}': no model to evaluate"
        )

    eval_cfg = stage.eval
    assert eval_cfg is not None
    all_metrics: dict[str, Any] = {}

    if eval_cfg.metrics:
        from xaytune.eval import evaluate

        dataset: list[dict[str, Any]] = []
        if eval_cfg.dataset:
            import json as _json
            from pathlib import Path as _Path

            raw = _Path(eval_cfg.dataset).read_text()
            dataset = [_json.loads(line) for line in raw.strip().splitlines() if line.strip()]

        if dataset:
            results = evaluate(
                model=model_path, dataset=dataset, metrics=eval_cfg.metrics
            )
            all_metrics.update(results)

    if eval_cfg.benchmarks:
        from xaytune.eval import benchmark_evaluate

        bench_results = benchmark_evaluate(
            model=model_path,
            benchmarks=eval_cfg.benchmarks,
            num_fewshot=eval_cfg.num_fewshot,
        )
        all_metrics["benchmarks"] = bench_results

    return StageResult(type="eval", metrics=all_metrics)


def _print_plan(config: PipelineConfig) -> None:
    print(f"\nPipeline: {config.name}")
    print(f"Output:   {config.output_dir}")
    print(f"Stages:   {len(config.stages)}\n")

    prev = "(none)"
    for i, stage in enumerate(config.stages):
        model = stage.model_name or prev
        if stage.recipe:
            kind = f"train ({stage.recipe}, method={stage.method or 'default'})"
            output = f"{config.output_dir}/{stage.name}"
        elif stage.export:
            kind = f"export ({stage.export})"
            output = stage.save_to or stage.repo or f"{config.output_dir}/{stage.name}"
        elif stage.eval:
            kind = f"eval (metrics={stage.eval.metrics})"
            output = "(metrics only)"
        else:
            kind = "unknown"
            output = "?"

        print(f"  [{i + 1}] {stage.name}")
        print(f"      Type:   {kind}")
        print(f"      Model:  {model}")
        print(f"      Output: {output}")
        print()
        if stage.recipe or stage.export:
            prev = output


def _save_state(
    path: Path, name: str, results: dict[str, StageResult]
) -> None:
    data = {
        "name": name,
        "stages": {k: v.model_dump() for k, v in results.items()},
    }
    path.write_text(json.dumps(data, indent=2, default=str))


def load_pipeline_config(path: str) -> PipelineConfig:
    """Load a pipeline configuration from a YAML file."""
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)

    return PipelineConfig(**raw)
