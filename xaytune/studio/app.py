from __future__ import annotations

import time
from typing import Any

import gradio as gr
import plotly.graph_objects as go

from xaytune.config.schema import (
    DataConfig,
    EvalConfig,
    LoggingConfig,
    LoraConfig,
    ModelConfig,
    OutputConfig,
    TrainConfig,
    TrainerConfig,
)
from xaytune.studio.code_runner import CODE_TEMPLATES, run_code
from xaytune.studio.codegen import METHOD_PARAMS_SPEC, generate_code
from xaytune.studio.data_preview import preview_dataset
from xaytune.studio.dataset_browser import get_dataset_info, preview_hf_dataset, search_datasets
from xaytune.studio.examples import EXAMPLES, load_example_values
from xaytune.studio.gpu_metrics import get_gpu_metrics  # noqa: F401
from xaytune.studio.hub_browser import search_models
from xaytune.studio.jobs import JobManager, JobStatus

RECIPE_METHODS: dict[str, list[str]] = {
    "finetune": ["full", "lora", "qlora"],
    "pretrain": ["full"],
    "align": ["dpo", "grpo", "ppo", "orpo", "simpo"],
}


STATUS_COLORS: dict[str, str] = {
    "pending": "#6b7280",
    "running": "#2563eb",
    "completed": "#16a34a",
    "failed": "#dc2626",
    "cancelled": "#d97706",
}


def validate_form(
    recipe: str,
    method: str,
    model_name: str,
    data_path: str,
    quantization: str | None,
    learning_rate: float,
    batch_size: int,
    warmup_steps: int,
    warmup_ratio: float,
    eval_split: float,
) -> list[str]:
    errors: list[str] = []

    if not model_name or not model_name.strip():
        errors.append("**Model Name** is required")
    if not data_path or not data_path.strip():
        errors.append("**Data Path** is required")

    valid_methods = RECIPE_METHODS.get(recipe, [])
    if method not in valid_methods:
        errors.append(
            f"**{recipe}** recipe requires method: {', '.join(valid_methods)} (got '{method}')"
        )

    if method == "qlora" and quantization != "4bit":
        errors.append("**QLoRA** requires quantization set to **4bit**")

    if learning_rate <= 0:
        errors.append("**Learning Rate** must be greater than 0")

    if batch_size < 1:
        errors.append("**Batch Size** must be at least 1")

    if warmup_steps > 0 and warmup_ratio > 0:
        errors.append("**Warmup Steps** and **Warmup Ratio** are mutually exclusive — set one to 0")

    if not (0.0 <= eval_split <= 1.0):
        errors.append("**Eval Split** must be between 0.0 and 1.0")

    return errors


def build_config(
    recipe: str,
    method: str,
    model_name: str,
    data_path: str,
    data_format: str,
    quantization: str | None = None,
    dtype: str = "auto",
    trust_remote_code: bool = False,
    max_seq_length: int = 2048,
    packing: bool = True,
    source: str = "local",
    streaming: bool = False,
    eval_split: float = 0.0,
    eval_path: str | None = None,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    batch_size: int = 4,
    gradient_accumulation: int = 1,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    max_steps: int = -1,
    warmup_steps: int = 0,
    warmup_ratio: float = 0.0,
    scheduler: str = "cosine",
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    mixed_precision: str = "bf16",
    seed: int = 42,
    checkpoint_every_n_steps: int = 500,
    save_last: bool = True,
    activation_checkpointing: bool = False,
    async_checkpoint: bool = False,
    eval_every_n_steps: int = 500,
    early_stopping_patience: int = 0,
    early_stopping_metric: str = "eval_loss",
    early_stopping_min_delta: float = 0.0,
    log_every_n_steps: int = 10,
    output_dir: str = "output",
    merge_on_complete: bool = False,
    method_params: dict[str, Any] | None = None,
) -> TrainConfig:
    model = ModelConfig(
        name=model_name,
        quantization=quantization,  # type: ignore[arg-type]
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    data = DataConfig(
        path=data_path,
        format=data_format,
        source=source,  # type: ignore[arg-type]
        max_seq_length=max_seq_length,
        packing=packing,
        streaming=streaming,
        eval_split=eval_split,
        eval_path=eval_path or None,
    )

    lora = LoraConfig()
    if method in ("lora", "qlora"):
        lora = LoraConfig(rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)

    trainer = TrainerConfig(
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        scheduler=scheduler,  # type: ignore[arg-type]
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        mixed_precision=mixed_precision,  # type: ignore[arg-type]
        seed=seed,
        checkpoint_every_n_steps=checkpoint_every_n_steps,
        save_last=save_last,
        activation_checkpointing=activation_checkpointing,
        async_checkpoint=async_checkpoint,
    )

    eval_cfg = EvalConfig(
        every_n_steps=eval_every_n_steps,
        early_stopping_patience=early_stopping_patience,
        early_stopping_metric=early_stopping_metric,
        early_stopping_min_delta=early_stopping_min_delta,
    )

    logging_cfg = LoggingConfig(log_every_n_steps=log_every_n_steps)
    output = OutputConfig(dir=output_dir, merge_on_complete=merge_on_complete)

    return TrainConfig(
        recipe=recipe,
        method=method,
        model=model,
        data=data,
        lora=lora,
        trainer=trainer,
        eval=eval_cfg,
        logging=logging_cfg,
        output=output,
        method_params=method_params or {},
    )


def _format_time(ts: float | None) -> str:
    if ts is None:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _status_badge(status: str) -> str:
    color = STATUS_COLORS.get(status, "#6b7280")
    return (
        f'<span style="background:{color}; color:white; padding:2px 10px; '
        f'border-radius:12px; font-size:0.85em; font-weight:600;">'
        f"{status.upper()}</span>"
    )


def create_app(
    job_manager: JobManager | None = None,
    theme: gr.Theme | None = None,
) -> gr.Blocks:
    mgr = job_manager or JobManager()

    with gr.Blocks(title="xaytune Studio", theme=theme) as app:
        gr.Markdown("# xaytune Studio\nConfigure, launch, and monitor LLM training jobs.")

        # ── Train Tab ──────────────────────────────────────────────
        with gr.Tab("Train"):
            mode = gr.Radio(
                choices=["Simple", "Advanced"],
                value="Simple",
                label="Mode",
                info="Simple: essential options only. Advanced: full control.",
            )

            if EXAMPLES:
                with gr.Accordion("Examples", open=False):
                    gr.Markdown(
                        "Load a pre-configured example to populate the form. "
                        "You can modify values before launching."
                    )
                    with gr.Row():
                        example_dropdown = gr.Dropdown(
                            choices=list(EXAMPLES.keys()),
                            label="Example Config",
                            info="Select an example configuration to load.",
                        )
                        load_example_btn = gr.Button("Load Example", size="sm", variant="secondary")

            with gr.Accordion("Recipe & Model", open=True):
                with gr.Row():
                    recipe = gr.Dropdown(
                        choices=["finetune", "pretrain", "align"],
                        value="finetune",
                        label="Recipe",
                        info="finetune: adapt model. pretrain: from scratch. align: RLHF.",
                    )
                    method = gr.Dropdown(
                        choices=["full", "lora", "qlora"],
                        value="full",
                        label="Method",
                        info="full: all weights. lora/qlora: parameter-efficient.",
                    )
                with gr.Row():
                    model_name = gr.Textbox(
                        label="Model Name",
                        placeholder="meta-llama/Llama-3-8B",
                        info="HuggingFace model ID or local path to model directory.",
                    )
                    hf_token = gr.Textbox(
                        label="HF Token",
                        placeholder="hf_...",
                        type="password",
                        info="Token for gated models. Leave blank if logged in.",
                    )
                with gr.Row():
                    quantization = gr.Dropdown(
                        choices=["None", "4bit", "8bit"],
                        value="None",
                        label="Quantization",
                        info="4bit required for QLoRA. 8bit saves memory. None for full precision.",
                    )
                    dtype = gr.Dropdown(
                        choices=["auto", "float16", "bfloat16", "float32"],
                        value="auto",
                        label="Dtype",
                        info="Model loading precision. 'auto' uses the model's native dtype.",
                    )
                    trust_remote_code = gr.Checkbox(
                        value=False,
                        label="Trust Remote Code",
                        info="Allow custom code from model repo. Required by some models.",
                    )

                with gr.Accordion("Model Search (HuggingFace Hub)", open=False):
                    model_search_query = gr.Textbox(
                        label="Search",
                        placeholder="llama, mistral, phi...",
                        info="Search HuggingFace Hub for text-generation models.",
                    )
                    model_search_btn = gr.Button("Search Models", size="sm")
                    model_search_results = gr.Dataframe(
                        headers=["Model ID", "Downloads", "Likes"],
                        label="Results",
                        interactive=False,
                    )

                    def _search_models(query: str):
                        if not query or not query.strip():
                            return []
                        results = search_models(query.strip())
                        return [[r["model_id"], r["downloads"], r["likes"]] for r in results]

                    model_search_btn.click(
                        fn=_search_models,
                        inputs=[model_search_query],
                        outputs=[model_search_results],
                    )

                    def _select_model(evt: gr.SelectData):
                        return evt.value

                    model_search_results.select(
                        fn=_select_model,
                        outputs=[model_name],
                    )

                def _on_recipe_change(r: str):
                    methods = RECIPE_METHODS.get(r, ["full"])
                    return gr.update(choices=methods, value=methods[0])

                recipe.change(
                    fn=_on_recipe_change,
                    inputs=[recipe],
                    outputs=[method],
                )

                def _on_method_change(m: str):
                    quant_update = {}
                    if m == "qlora":
                        quant_update = {"value": "4bit"}
                    return gr.update(visible=m in ("lora", "qlora")), gr.update(**quant_update)  # type: ignore[arg-type]

            with gr.Accordion("Data", open=True):
                with gr.Row():
                    data_path = gr.Textbox(
                        label="Data Path",
                        placeholder="data/train.jsonl",
                        info="Path to training data file or HuggingFace dataset name.",
                    )
                    data_format = gr.Dropdown(
                        choices=["alpaca", "sharegpt", "chat", "text", "preference"],
                        value="alpaca",
                        label="Data Format",
                        info="alpaca: instruction/output. sharegpt: chat. text: raw text. preference: RLHF.",
                    )
                    source = gr.Dropdown(
                        choices=["local", "huggingface"],
                        value="local",
                        label="Source",
                        info="Where to load data from.",
                    )
                with gr.Row():
                    max_seq_length = gr.Number(
                        value=2048,
                        label="Max Sequence Length",
                        precision=0,
                        minimum=64,
                        maximum=131072,
                        step=64,
                        info="Maximum token length per sample. Longer sequences use more memory.",
                    )
                    eval_split = gr.Number(
                        value=0.0,
                        label="Eval Split",
                        precision=2,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        info="Fraction of data for evaluation (0.0 = no eval split, 0.1 = 10%).",
                    )
                    eval_path = gr.Textbox(
                        label="Eval Data Path",
                        placeholder="(optional) data/eval.jsonl",
                        info="Separate evaluation dataset. Overrides eval_split if set.",
                    )
                with gr.Row():
                    packing = gr.Checkbox(
                        value=True,
                        label="Sequence Packing",
                        info="Pack multiple short samples into one sequence for efficiency.",
                    )
                    streaming = gr.Checkbox(
                        value=False,
                        label="Streaming",
                        info="Stream data instead of loading all into memory.",
                    )

            with gr.Accordion("Data Preview", open=False):
                preview_btn = gr.Button("Preview Data", size="sm")
                preview_table = gr.Dataframe(
                    label="Sample Data",
                    interactive=False,
                )

                def _preview_data(path: str, fmt: str, src: str):
                    if not path or not path.strip():
                        return []
                    samples = preview_dataset(path.strip(), format=fmt, source=src, num_samples=5)
                    if not samples:
                        return []
                    headers = list(samples[0].keys())
                    rows = []
                    for s in samples:
                        row = []
                        for h in headers:
                            val = s.get(h, "")
                            text = str(val)
                            if len(text) > 200:
                                text = text[:200] + "..."
                            row.append(text)
                        rows.append(row)
                    return rows

                preview_btn.click(
                    fn=_preview_data,
                    inputs=[data_path, data_format, source],
                    outputs=[preview_table],
                )

            with gr.Accordion(
                "LoRA", open=True, visible=False, elem_id="lora_accordion"
            ) as lora_accordion:
                with gr.Row():
                    lora_rank = gr.Number(
                        value=16,
                        label="Rank",
                        precision=0,
                        minimum=1,
                        maximum=256,
                        step=1,
                        info="LoRA rank. Higher = more capacity, slower. 8-64 typical.",
                    )
                    lora_alpha = gr.Number(
                        value=32,
                        label="Alpha",
                        precision=0,
                        minimum=1,
                        maximum=512,
                        step=1,
                        info="LoRA scaling factor. Common rule: alpha = 2 * rank.",
                    )
                    lora_dropout = gr.Number(
                        value=0.05,
                        label="Dropout",
                        precision=2,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        info="Dropout probability for LoRA layers. 0.0-0.1 typical.",
                    )

            method.change(
                fn=_on_method_change,
                inputs=[method],
                outputs=[lora_accordion, quantization],
            )

            with gr.Accordion(
                "Method Parameters",
                open=True,
                visible=False,
            ) as method_params_accordion:
                mp_info = gr.Markdown("Select an alignment method to configure parameters.")
                mp_beta = gr.Number(
                    value=0.1,
                    label="beta",
                    visible=False,
                    info="Preference strength temperature.",
                )
                mp_kl_coeff = gr.Number(
                    value=0.04,
                    label="kl_coeff",
                    visible=False,
                    info="KL divergence penalty weight.",
                )
                mp_clip_eps = gr.Number(
                    value=0.2,
                    label="clip_eps",
                    visible=False,
                    info="Policy clipping range.",
                )
                mp_lambda_weight = gr.Number(
                    value=1.0,
                    label="lambda_weight",
                    visible=False,
                    info="Odds-ratio loss weight.",
                )
                mp_gamma = gr.Number(
                    value=0.5,
                    label="gamma",
                    visible=False,
                    info="Sigmoid offset.",
                )

            _mp_fields = {
                "beta": mp_beta,
                "kl_coeff": mp_kl_coeff,
                "clip_eps": mp_clip_eps,
                "lambda_weight": mp_lambda_weight,
                "gamma": mp_gamma,
            }

            def _on_method_for_params(m: str):
                spec = METHOD_PARAMS_SPEC.get(m, [])
                show_accordion = len(spec) > 0
                active_names = {p["name"] for p in spec}
                updates: list = [gr.update(visible=show_accordion)]
                if spec:
                    info_text = f"Parameters for **{m}**:"
                    updates.append(gr.update(value=info_text, visible=True))
                else:
                    updates.append(gr.update(visible=False))
                for name in ("beta", "kl_coeff", "clip_eps", "lambda_weight", "gamma"):
                    if name in active_names:
                        default = next(p["default"] for p in spec if p["name"] == name)
                        updates.append(gr.update(visible=True, value=default))
                    else:
                        updates.append(gr.update(visible=False))
                return updates

            method.change(
                fn=_on_method_for_params,
                inputs=[method],
                outputs=[
                    method_params_accordion,
                    mp_info,
                    mp_beta,
                    mp_kl_coeff,
                    mp_clip_eps,
                    mp_lambda_weight,
                    mp_gamma,
                ],
            )

            with gr.Column(visible=False) as advanced_column:
                with gr.Accordion("Training", open=False):
                    with gr.Row():
                        batch_size = gr.Number(
                            value=4,
                            label="Batch Size",
                            precision=0,
                            minimum=1,
                            step=1,
                            info="Samples per device per step. Reduce if out of memory.",
                        )
                        grad_accum = gr.Number(
                            value=1,
                            label="Gradient Accumulation",
                            precision=0,
                            minimum=1,
                            step=1,
                            info="Accumulate gradients over N steps before updating.",
                        )
                        learning_rate = gr.Number(
                            value=2e-4,
                            label="Learning Rate",
                            minimum=1e-7,
                            maximum=1.0,
                            info="Typical: 1e-5 (pretrain), 2e-4 (finetune), 5e-5 (align).",
                        )
                    with gr.Row():
                        num_epochs = gr.Number(
                            value=3,
                            label="Num Epochs",
                            precision=0,
                            minimum=1,
                            step=1,
                            info="Number of full passes through the dataset.",
                        )
                        max_steps = gr.Number(
                            value=-1,
                            label="Max Steps",
                            precision=0,
                            minimum=-1,
                            step=1,
                            info="Stop after N steps regardless of epochs. -1 = no limit.",
                        )
                        seed = gr.Number(
                            value=42,
                            label="Seed",
                            precision=0,
                            minimum=0,
                            step=1,
                            info="Random seed for reproducibility.",
                        )
                    with gr.Row():
                        mixed_precision = gr.Dropdown(
                            choices=["fp16", "bf16", "fp32"],
                            value="bf16",
                            label="Mixed Precision",
                            info="bf16: modern GPUs. fp16: wider compat. fp32: full.",
                        )
                        scheduler = gr.Dropdown(
                            choices=["cosine", "linear", "constant", "constant_with_warmup"],
                            value="cosine",
                            label="LR Scheduler",
                            info="How the learning rate decays. cosine is most common.",
                        )
                    with gr.Row():
                        warmup_steps = gr.Number(
                            value=0,
                            label="Warmup Steps",
                            precision=0,
                            minimum=0,
                            step=1,
                            info="Ramp LR from 0 over N steps. Exclusive with warmup ratio.",
                        )
                        warmup_ratio = gr.Number(
                            value=0.0,
                            label="Warmup Ratio",
                            precision=2,
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            info="Warmup as fraction of total steps.",
                        )
                        weight_decay = gr.Number(
                            value=0.01,
                            label="Weight Decay",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.001,
                            info="L2 regularization strength. 0.01-0.1 typical.",
                        )
                        max_grad_norm = gr.Number(
                            value=1.0,
                            label="Max Grad Norm",
                            minimum=0.0,
                            step=0.1,
                            info="Gradient clipping threshold. 1.0 is standard.",
                        )

                with gr.Accordion("Evaluation", open=False):
                    with gr.Row():
                        eval_every = gr.Number(
                            value=500,
                            label="Eval Every N Steps",
                            precision=0,
                            minimum=1,
                            step=10,
                            info="Run evaluation every N training steps.",
                        )
                        es_patience = gr.Number(
                            value=0,
                            label="Early Stopping Patience",
                            precision=0,
                            minimum=0,
                            step=1,
                            info="Stop after N evals without improvement. 0 = disabled.",
                        )
                    with gr.Row():
                        es_metric = gr.Textbox(
                            value="eval_loss",
                            label="Early Stopping Metric",
                            info="Metric to monitor for early stopping.",
                        )
                        es_min_delta = gr.Number(
                            value=0.0,
                            label="Min Delta",
                            minimum=0.0,
                            step=0.001,
                            info="Minimum change to count as improvement.",
                        )

                with gr.Accordion("Logging & Output", open=False):
                    with gr.Row():
                        log_every = gr.Number(
                            value=10,
                            label="Log Every N Steps",
                            precision=0,
                            minimum=1,
                            step=1,
                            info="How often to log metrics to console/backends.",
                        )
                        output_dir = gr.Textbox(
                            value="output",
                            label="Output Directory",
                            info="Where to save checkpoints and final model.",
                        )
                        merge_on_complete = gr.Checkbox(
                            value=False,
                            label="Merge on Complete",
                            info="Merge LoRA weights into base after training finishes.",
                        )

                with gr.Accordion("Advanced", open=False):
                    with gr.Row():
                        checkpoint_every = gr.Number(
                            value=500,
                            label="Checkpoint Every N Steps",
                            precision=0,
                            minimum=1,
                            step=50,
                            info="Save a checkpoint every N training steps.",
                        )
                        save_last = gr.Checkbox(
                            value=True,
                            label="Save Last Checkpoint",
                            info="Always save a checkpoint at the end of training.",
                        )
                    with gr.Row():
                        activation_ckpt = gr.Checkbox(
                            value=False,
                            label="Activation Checkpointing",
                            info="Recompute activations during backward to save memory.",
                        )
                        async_ckpt = gr.Checkbox(
                            value=False,
                            label="Async Checkpoint",
                            info="Save checkpoints in background to avoid blocking.",
                        )

            mode.change(
                fn=lambda m: gr.update(visible=m == "Advanced"),
                inputs=[mode],
                outputs=[advanced_column],
            )

            if EXAMPLES:
                _example_outputs = [
                    recipe,
                    method,
                    model_name,
                    data_path,
                    data_format,
                    quantization,
                    dtype,
                    trust_remote_code,
                    source,
                    max_seq_length,
                    packing,
                    streaming,
                    eval_split,
                    eval_path,
                    lora_rank,
                    lora_alpha,
                    lora_dropout,
                    batch_size,
                    grad_accum,
                    learning_rate,
                    num_epochs,
                    max_steps,
                    seed,
                    mixed_precision,
                    scheduler,
                    warmup_steps,
                    warmup_ratio,
                    weight_decay,
                    max_grad_norm,
                    eval_every,
                    es_patience,
                    es_metric,
                    es_min_delta,
                    log_every,
                    output_dir,
                    merge_on_complete,
                    checkpoint_every,
                    save_last,
                    activation_ckpt,
                    async_ckpt,
                    mp_beta,
                    mp_kl_coeff,
                    mp_clip_eps,
                    mp_lambda_weight,
                    mp_gamma,
                ]

                def _load_example(name: str | None):
                    if not name:
                        return [gr.update()] * len(_example_outputs)
                    vals = load_example_values(name)
                    if not vals:
                        return [gr.update()] * len(_example_outputs)
                    return [
                        gr.update(value=vals.get("recipe", "finetune")),
                        gr.update(value=vals.get("method", "full")),
                        gr.update(value=vals.get("model_name", "")),
                        gr.update(value=vals.get("data_path", "")),
                        gr.update(value=vals.get("data_format", "alpaca")),
                        gr.update(value=vals.get("quantization", "None")),
                        gr.update(value=vals.get("dtype", "auto")),
                        gr.update(value=vals.get("trust_remote_code", False)),
                        gr.update(value=vals.get("source", "local")),
                        gr.update(value=vals.get("max_seq_length", 2048)),
                        gr.update(value=vals.get("packing", True)),
                        gr.update(value=vals.get("streaming", False)),
                        gr.update(value=vals.get("eval_split", 0.0)),
                        gr.update(value=vals.get("eval_path", "")),
                        gr.update(value=vals.get("lora_rank", 16)),
                        gr.update(value=vals.get("lora_alpha", 32)),
                        gr.update(value=vals.get("lora_dropout", 0.05)),
                        gr.update(value=vals.get("batch_size", 4)),
                        gr.update(value=vals.get("gradient_accumulation", 1)),
                        gr.update(value=vals.get("learning_rate", 2e-4)),
                        gr.update(value=vals.get("num_epochs", 3)),
                        gr.update(value=vals.get("max_steps", -1)),
                        gr.update(value=vals.get("seed", 42)),
                        gr.update(value=vals.get("mixed_precision", "bf16")),
                        gr.update(value=vals.get("scheduler", "cosine")),
                        gr.update(value=vals.get("warmup_steps", 0)),
                        gr.update(value=vals.get("warmup_ratio", 0.0)),
                        gr.update(value=vals.get("weight_decay", 0.01)),
                        gr.update(value=vals.get("max_grad_norm", 1.0)),
                        gr.update(value=vals.get("eval_every_n_steps", 500)),
                        gr.update(value=vals.get("early_stopping_patience", 0)),
                        gr.update(value=vals.get("early_stopping_metric", "eval_loss")),
                        gr.update(value=vals.get("early_stopping_min_delta", 0.0)),
                        gr.update(value=vals.get("log_every_n_steps", 10)),
                        gr.update(value=vals.get("output_dir", "output")),
                        gr.update(value=vals.get("merge_on_complete", False)),
                        gr.update(value=vals.get("checkpoint_every_n_steps", 500)),
                        gr.update(value=vals.get("save_last", True)),
                        gr.update(value=vals.get("activation_checkpointing", False)),
                        gr.update(value=vals.get("async_checkpoint", False)),
                        gr.update(value=vals.get("beta", 0.1)),
                        gr.update(value=vals.get("kl_coeff", 0.04)),
                        gr.update(value=vals.get("clip_eps", 0.2)),
                        gr.update(value=vals.get("lambda_weight", 1.0)),
                        gr.update(value=vals.get("gamma", 0.5)),
                    ]

                load_example_btn.click(
                    fn=_load_example,
                    inputs=[example_dropdown],
                    outputs=_example_outputs,
                )

            validation_output = gr.Markdown("")
            with gr.Row():
                submit_btn = gr.Button("Start Training", variant="primary", size="lg")
                gen_code_btn = gr.Button("Generate Code", size="lg")
            train_status = gr.Markdown("")
            generated_code = gr.Code(
                label="Generated Python",
                language="python",
                interactive=False,
                lines=10,
            )

            all_inputs = [
                recipe,
                method,
                model_name,
                data_path,
                data_format,
                quantization,
                dtype,
                trust_remote_code,
                source,
                max_seq_length,
                packing,
                streaming,
                eval_split,
                eval_path,
                lora_rank,
                lora_alpha,
                lora_dropout,
                batch_size,
                grad_accum,
                learning_rate,
                num_epochs,
                max_steps,
                seed,
                mixed_precision,
                scheduler,
                warmup_steps,
                warmup_ratio,
                weight_decay,
                max_grad_norm,
                eval_every,
                es_patience,
                es_metric,
                es_min_delta,
                log_every,
                output_dir,
                merge_on_complete,
                checkpoint_every,
                save_last,
                activation_ckpt,
                async_ckpt,
                mp_beta,
                mp_kl_coeff,
                mp_clip_eps,
                mp_lambda_weight,
                mp_gamma,
                hf_token,
            ]

            def _submit(
                recipe_v,
                method_v,
                model_v,
                data_v,
                format_v,
                quant_v,
                dtype_v,
                trust_v,
                source_v,
                seq_len_v,
                packing_v,
                streaming_v,
                esplit_v,
                epath_v,
                lr_rank_v,
                lr_alpha_v,
                lr_drop_v,
                bs_v,
                ga_v,
                lr_v,
                epochs_v,
                steps_v,
                seed_v,
                mp_v,
                sched_v,
                warmup_v,
                wratio_v,
                wd_v,
                gn_v,
                eval_v,
                esp_v,
                esm_v,
                esd_v,
                log_v,
                out_v,
                merge_v,
                ckpt_v,
                slast_v,
                actckpt_v,
                asyncckpt_v,
                beta_v,
                kl_coeff_v,
                clip_eps_v,
                lambda_w_v,
                gamma_v,
                token_v,
            ):
                if token_v and token_v.strip():
                    try:
                        from huggingface_hub import login

                        login(token=token_v.strip(), add_to_git_credential=False)
                    except Exception as e:
                        return (
                            f'<div style="color:#dc2626; padding:8px; '
                            f"border:1px solid #dc2626; border-radius:8px; "
                            f'background:#fef2f2;">\n\n'
                            f"**HF Login error:** {e}\n\n</div>"
                        ), ""

                quant_val = None if quant_v == "None" else quant_v
                errors = validate_form(
                    recipe=recipe_v,
                    method=method_v,
                    model_name=model_v,
                    data_path=data_v,
                    quantization=quant_val,
                    learning_rate=float(lr_v),
                    batch_size=int(bs_v),
                    warmup_steps=int(warmup_v),
                    warmup_ratio=float(wratio_v),
                    eval_split=float(esplit_v),
                )
                if errors:
                    error_md = "\n".join(f"- {e}" for e in errors)
                    return (
                        f'<div style="color:#dc2626; padding:8px; '
                        f"border:1px solid #dc2626; border-radius:8px; "
                        f'background:#fef2f2;">\n\n'
                        f"**Validation Errors:**\n\n{error_md}\n\n</div>"
                    ), ""

                mparams: dict[str, Any] = {}
                spec = METHOD_PARAMS_SPEC.get(method_v, [])
                param_values = {
                    "beta": beta_v,
                    "kl_coeff": kl_coeff_v,
                    "clip_eps": clip_eps_v,
                    "lambda_weight": lambda_w_v,
                    "gamma": gamma_v,
                }
                for p in spec:
                    mparams[p["name"]] = float(param_values[p["name"]])

                try:
                    config = build_config(
                        recipe=recipe_v,
                        method=method_v,
                        model_name=model_v.strip(),
                        data_path=data_v.strip(),
                        data_format=format_v,
                        quantization=quant_val,
                        dtype=dtype_v,
                        trust_remote_code=trust_v,
                        source=source_v,
                        max_seq_length=int(seq_len_v),
                        packing=packing_v,
                        streaming=streaming_v,
                        eval_split=float(esplit_v),
                        eval_path=epath_v or None,
                        lora_rank=int(lr_rank_v),
                        lora_alpha=int(lr_alpha_v),
                        lora_dropout=float(lr_drop_v),
                        batch_size=int(bs_v),
                        gradient_accumulation=int(ga_v),
                        learning_rate=float(lr_v),
                        num_epochs=int(epochs_v),
                        max_steps=int(steps_v),
                        warmup_steps=int(warmup_v),
                        warmup_ratio=float(wratio_v),
                        scheduler=sched_v,
                        weight_decay=float(wd_v),
                        max_grad_norm=float(gn_v),
                        mixed_precision=mp_v,
                        seed=int(seed_v),
                        checkpoint_every_n_steps=int(ckpt_v),
                        save_last=slast_v,
                        activation_checkpointing=actckpt_v,
                        async_checkpoint=asyncckpt_v,
                        eval_every_n_steps=int(eval_v),
                        early_stopping_patience=int(esp_v),
                        early_stopping_metric=esm_v,
                        early_stopping_min_delta=float(esd_v),
                        log_every_n_steps=int(log_v),
                        output_dir=out_v,
                        merge_on_complete=merge_v,
                        method_params=mparams,
                    )
                except Exception as e:
                    return (
                        f'<div style="color:#dc2626; padding:8px; '
                        f"border:1px solid #dc2626; border-radius:8px; "
                        f'background:#fef2f2;">\n\n'
                        f"**Config error:** {e}\n\n</div>"
                    ), ""

                job_id = mgr.submit(config)
                return "", (
                    f'<div style="color:#16a34a; padding:8px; '
                    f"border:1px solid #16a34a; border-radius:8px; "
                    f'background:#f0fdf4;">\n\n'
                    f"**Job submitted:** `{job_id}`\n\n</div>"
                )

            submit_btn.click(
                fn=_submit,
                inputs=all_inputs,
                outputs=[validation_output, train_status],
            )

            def _gen_code(
                recipe_v,
                method_v,
                model_v,
                data_v,
                format_v,
                quant_v,
                dtype_v,
                trust_v,
                source_v,
                seq_len_v,
                packing_v,
                streaming_v,
                esplit_v,
                epath_v,
                lr_rank_v,
                lr_alpha_v,
                lr_drop_v,
                bs_v,
                ga_v,
                lr_v,
                epochs_v,
                steps_v,
                seed_v,
                mp_v,
                sched_v,
                warmup_v,
                wratio_v,
                wd_v,
                gn_v,
                eval_v,
                esp_v,
                esm_v,
                esd_v,
                log_v,
                out_v,
                merge_v,
                ckpt_v,
                slast_v,
                actckpt_v,
                asyncckpt_v,
                beta_v,
                kl_coeff_v,
                clip_eps_v,
                lambda_w_v,
                gamma_v,
                _token_v,
            ):
                try:
                    quant_val = None if quant_v == "None" else quant_v
                    code = generate_code(
                        recipe=recipe_v,
                        method=method_v,
                        model_name=model_v.strip() if model_v else "",
                        data_path=data_v.strip() if data_v else "",
                        data_format=format_v,
                        quantization=quant_val,
                        dtype=dtype_v,
                        trust_remote_code=trust_v,
                        max_seq_length=int(float(seq_len_v)),
                        packing=packing_v,
                        streaming=streaming_v,
                        eval_split=float(esplit_v),
                        eval_path=epath_v or "",
                        lora_rank=int(float(lr_rank_v)),
                        lora_alpha=int(float(lr_alpha_v)),
                        lora_dropout=float(lr_drop_v),
                        batch_size=int(float(bs_v)),
                        gradient_accumulation=int(float(ga_v)),
                        learning_rate=float(lr_v),
                        num_epochs=int(float(epochs_v)),
                        max_steps=int(float(steps_v)),
                        seed=int(float(seed_v)),
                        mixed_precision=mp_v,
                        scheduler=sched_v,
                        warmup_steps=int(float(warmup_v)),
                        warmup_ratio=float(wratio_v),
                        weight_decay=float(wd_v),
                        max_grad_norm=float(gn_v),
                        eval_every_n_steps=int(float(eval_v)),
                        early_stopping_patience=int(float(esp_v)),
                        early_stopping_metric=esm_v,
                        early_stopping_min_delta=float(esd_v),
                        log_every_n_steps=int(float(log_v)),
                        output_dir=out_v,
                        merge_on_complete=merge_v,
                        checkpoint_every_n_steps=int(float(ckpt_v)),
                        save_last=slast_v,
                        activation_checkpointing=actckpt_v,
                        async_checkpoint=asyncckpt_v,
                        beta=float(beta_v),
                        kl_coeff=float(kl_coeff_v),
                        clip_eps=float(clip_eps_v),
                        lambda_weight=float(lambda_w_v),
                        gamma=float(gamma_v),
                    )
                    return code
                except Exception as exc:
                    return f"# Error generating code: {exc}"

            gen_code_btn.click(
                fn=_gen_code,
                inputs=all_inputs,
                outputs=[generated_code],
            )

        # ── Monitor Tab ────────────────────────────────────────────
        with gr.Tab("Monitor"):
            with gr.Row():
                job_dropdown = gr.Dropdown(
                    choices=[],
                    label="Job ID",
                    allow_custom_value=True,
                    info="Select a running or completed job to monitor.",
                )
                refresh_jobs_btn = gr.Button("Refresh Jobs", size="sm")
                cancel_btn = gr.Button("Cancel Job", variant="stop", size="sm", visible=False)

            def _refresh_job_list():
                jobs = mgr.list_jobs()
                ids = [j.job_id[:8] + "..." for j in jobs]
                full_ids = [j.job_id for j in jobs]
                choices = list(zip(ids, full_ids)) if ids else []
                return gr.update(choices=choices, value=full_ids[0] if full_ids else None)

            refresh_jobs_btn.click(fn=_refresh_job_list, outputs=[job_dropdown])

            def _cancel_job(job_id: str | None):
                if not job_id:
                    return "No job selected."
                try:
                    mgr.cancel(job_id)
                    return f"Cancelled job `{job_id[:8]}...`"
                except KeyError:
                    return f"Unknown job: {job_id}"

            cancel_btn.click(
                fn=_cancel_job,
                inputs=[job_dropdown],
                outputs=[gr.Markdown()],
            )

            monitor_status = gr.Markdown("Select a job to monitor.")
            with gr.Row():
                with gr.Column(scale=1):
                    loss_plot = gr.Plot(label="Loss & Learning Rate")
                with gr.Column(scale=1):
                    gpu_plot = gr.Plot(label="GPU Memory & Utilization")
            with gr.Row():
                with gr.Column(scale=1):
                    throughput_display = gr.Markdown("")
                with gr.Column(scale=1):
                    metrics_display = gr.Markdown("")
            log_output = gr.Code(label="Training Logs", language=None, lines=12)

            timer = gr.Timer(value=2, active=False)

            timer.tick(
                fn=lambda job_id: _poll(mgr, job_id),
                inputs=[job_dropdown],
                outputs=[
                    monitor_status,
                    loss_plot,
                    gpu_plot,
                    throughput_display,
                    metrics_display,
                    cancel_btn,
                    log_output,
                ],
            )

            job_dropdown.change(
                fn=lambda _: True,
                inputs=[job_dropdown],
                outputs=[timer],
            )

        # ── History Tab ────────────────────────────────────────────
        with gr.Tab("History"):
            refresh_history_btn = gr.Button("Refresh", size="sm")
            history_table = gr.Dataframe(
                headers=[
                    "Job ID",
                    "Recipe",
                    "Status",
                    "Created",
                    "Completed",
                    "Final Loss",
                    "Tags",
                ],
                label="Training Jobs",
            )

            with gr.Row():
                tag_job_dropdown = gr.Dropdown(
                    choices=[],
                    label="Job",
                    allow_custom_value=True,
                )
                tag_input = gr.Textbox(
                    label="Tag",
                    placeholder="experiment-v1",
                )
                add_tag_btn = gr.Button("Add Tag", size="sm")
                remove_tag_btn = gr.Button("Remove Tag", size="sm")

            def _refresh_history():
                jobs = mgr.list_jobs()
                ids = [j.job_id[:8] + "..." for j in jobs]
                full_ids = [j.job_id for j in jobs]
                choices = list(zip(ids, full_ids)) if ids else []
                rows = []
                for j in jobs:
                    loss = "-"
                    if j.state and "metrics" in j.state:
                        loss_val = j.state["metrics"].get("loss")
                        if isinstance(loss_val, (int, float)):
                            loss = f"{loss_val:.4f}"
                    rows.append(
                        [
                            j.job_id[:8],
                            j.recipe,
                            j.status.value,
                            _format_time(j.created_at),
                            _format_time(j.completed_at),
                            loss,
                            ", ".join(j.tags),
                        ]
                    )
                return rows, gr.update(choices=choices)

            refresh_history_btn.click(
                fn=_refresh_history,
                outputs=[history_table, tag_job_dropdown],
            )

            def _add_tag(job_id: str | None, tag: str):
                if job_id and tag and tag.strip():
                    try:
                        mgr.add_tag(job_id, tag.strip())
                    except KeyError:
                        pass

            def _remove_tag(job_id: str | None, tag: str):
                if job_id and tag and tag.strip():
                    try:
                        mgr.remove_tag(job_id, tag.strip())
                    except KeyError:
                        pass

            add_tag_btn.click(
                fn=_add_tag,
                inputs=[tag_job_dropdown, tag_input],
            )
            remove_tag_btn.click(
                fn=_remove_tag,
                inputs=[tag_job_dropdown, tag_input],
            )

        # ── Compare Tab ───────────────────────────────────────────
        with gr.Tab("Compare"):
            compare_refresh_btn = gr.Button("Refresh Jobs", size="sm")
            compare_jobs = gr.Dropdown(
                choices=[],
                label="Select Jobs to Compare",
                multiselect=True,
                allow_custom_value=True,
                info="Select two or more jobs to compare their training curves.",
            )

            def _refresh_compare_list():
                jobs = mgr.list_jobs()
                ids = [j.job_id[:8] + "..." for j in jobs]
                full_ids = [j.job_id for j in jobs]
                choices = list(zip(ids, full_ids)) if ids else []
                return gr.update(choices=choices)

            compare_refresh_btn.click(
                fn=_refresh_compare_list,
                outputs=[compare_jobs],
            )

            compare_btn = gr.Button("Compare", variant="primary")
            compare_plot = gr.Plot(label="Loss Comparison")
            compare_table = gr.Dataframe(
                headers=["Job ID", "Recipe", "Method", "Steps", "Final Loss", "Tags"],
                label="Metrics Comparison",
                interactive=False,
            )

            def _compare(job_ids: list[str] | None):
                if not job_ids or len(job_ids) < 2:
                    return _empty_plot(), []

                colors = [
                    "#4f46e5",
                    "#dc2626",
                    "#16a34a",
                    "#d97706",
                    "#7c3aed",
                    "#0891b2",
                    "#be185d",
                    "#65a30d",
                ]

                fig = go.Figure()
                table_rows = []
                for i, jid in enumerate(job_ids):
                    try:
                        job = mgr.get_status(jid)
                    except KeyError:
                        continue

                    hist = job.metrics_history
                    if hist:
                        steps = [h["step"] for h in hist]
                        losses = [h.get("loss", 0) for h in hist]
                        color = colors[i % len(colors)]
                        fig.add_trace(
                            go.Scatter(
                                x=steps,
                                y=losses,
                                mode="lines+markers",
                                name=jid[:8],
                                line={"color": color, "width": 2},
                                marker={"size": 3},
                            )
                        )

                    final_loss = "-"
                    if job.state and "metrics" in job.state:
                        loss_val = job.state["metrics"].get("loss")
                        if isinstance(loss_val, (int, float)):
                            final_loss = f"{loss_val:.4f}"

                    total_steps = 0
                    if job.state and "global_step" in job.state:
                        total_steps = job.state["global_step"]

                    table_rows.append(
                        [
                            jid[:8],
                            job.recipe,
                            job.state.get("method", "-") if job.state else "-",
                            total_steps,
                            final_loss,
                            ", ".join(job.tags),
                        ]
                    )

                fig.update_layout(
                    title="Training Loss Comparison",
                    xaxis_title="Step",
                    yaxis_title="Loss",
                    template="plotly_white",
                    margin={"l": 40, "r": 20, "t": 40, "b": 40},
                )

                return fig, table_rows

            compare_btn.click(
                fn=_compare,
                inputs=[compare_jobs],
                outputs=[compare_plot, compare_table],
            )

        # ── Datasets Tab ──────────────────────────────────────────
        with gr.Tab("Datasets"):
            gr.Markdown("Search and preview HuggingFace datasets.")
            with gr.Row():
                ds_search_query = gr.Textbox(
                    label="Search",
                    placeholder="alpaca, code, math, chat...",
                    info="Search HuggingFace Hub for datasets.",
                )
                ds_search_btn = gr.Button("Search Datasets", size="sm")
            ds_results = gr.Dataframe(
                headers=["Dataset ID", "Downloads", "Likes", "Tags"],
                label="Results",
                interactive=False,
            )

            def _search_datasets(query: str):
                if not query or not query.strip():
                    return []
                results = search_datasets(query.strip())
                return [[r["dataset_id"], r["downloads"], r["likes"], r["tags"]] for r in results]

            ds_search_btn.click(
                fn=_search_datasets,
                inputs=[ds_search_query],
                outputs=[ds_results],
            )

            with gr.Accordion("Preview", open=False):
                with gr.Row():
                    ds_preview_id = gr.Textbox(
                        label="Dataset ID",
                        placeholder="tatsu-lab/alpaca",
                        info="Enter a HuggingFace dataset ID to preview.",
                    )
                    ds_split = gr.Dropdown(
                        choices=["train", "test", "validation"],
                        value="train",
                        label="Split",
                    )
                    ds_preview_btn = gr.Button("Preview", size="sm")

                ds_preview_table = gr.Dataframe(
                    label="Samples",
                    interactive=False,
                )
                ds_info_md = gr.Markdown("")

                def _preview_hf(dataset_id: str, split: str):
                    if not dataset_id or not dataset_id.strip():
                        return [], ""
                    samples = preview_hf_dataset(dataset_id.strip(), split=split, num_samples=5)
                    if not samples:
                        return [], "No samples found or dataset could not be loaded."
                    headers = list(samples[0].keys())
                    rows = []
                    for s in samples:
                        row = []
                        for h in headers:
                            val = s.get(h, "")
                            text = str(val)
                            if len(text) > 200:
                                text = text[:200] + "..."
                            row.append(text)
                        rows.append(row)

                    info = get_dataset_info(dataset_id.strip())
                    info_text = ""
                    if info:
                        info_text = f"**{info.get('id', '')}**"
                        desc = info.get("description", "")
                        if desc:
                            info_text += f"\n\n{desc}"
                        downloads = info.get("downloads", 0)
                        info_text += f"\n\nDownloads: **{downloads:,}**"
                        tags = info.get("tags", [])
                        if tags:
                            info_text += f" | Tags: {', '.join(tags[:10])}"

                    return rows, info_text

                ds_preview_btn.click(
                    fn=_preview_hf,
                    inputs=[ds_preview_id, ds_split],
                    outputs=[ds_preview_table, ds_info_md],
                )

                def _select_dataset(evt: gr.SelectData):
                    return evt.value

                ds_results.select(
                    fn=_select_dataset,
                    outputs=[ds_preview_id],
                )

        # ── Code Tab ─────────────────────────────────────────────
        with gr.Tab("Code"):
            gr.Markdown(
                "Write and run Python code using the xaytune API. "
                "The `xaytune` module is pre-imported."
            )
            with gr.Row():
                code_template = gr.Dropdown(
                    choices=list(CODE_TEMPLATES.keys()),
                    value="Custom",
                    label="Template",
                    info="Load a starter template.",
                )
                code_run_btn = gr.Button("Run", variant="primary", size="sm")
                code_clear_btn = gr.Button("Clear Output", size="sm")

            code_editor = gr.Code(
                value=CODE_TEMPLATES["Custom"],
                language="python",
                interactive=True,
                lines=20,
                label="Python Editor",
            )
            code_output = gr.Code(
                label="Output",
                language=None,
                interactive=False,
                lines=12,
            )
            code_status = gr.Markdown("")

            def _on_template_change(name: str):
                return gr.update(value=CODE_TEMPLATES.get(name, ""))

            code_template.change(
                fn=_on_template_change,
                inputs=[code_template],
                outputs=[code_editor],
            )

            def _run_code(code: str):
                if not code or not code.strip():
                    return "", "No code to run."
                result = run_code(code)
                output_parts = []
                if result.stdout:
                    output_parts.append(result.stdout)
                if result.stderr:
                    output_parts.append(f"[stderr]\n{result.stderr}")
                if result.error:
                    output_parts.append(f"[error]\n{result.error}")
                output_text = "\n".join(output_parts) if output_parts else "(no output)"
                if result.error:
                    status = f'<span style="color:#dc2626;">Failed</span> in {result.duration:.1f}s'
                else:
                    status = (
                        f'<span style="color:#16a34a;">Completed</span> in {result.duration:.1f}s'
                    )
                return output_text, status

            code_run_btn.click(
                fn=_run_code,
                inputs=[code_editor],
                outputs=[code_output, code_status],
            )

            code_clear_btn.click(
                fn=lambda: ("", ""),
                outputs=[code_output, code_status],
            )

    return app  # type: ignore[no-any-return]


_PollResult = tuple[str, go.Figure, go.Figure, str, str, Any, str]


def _poll(
    mgr: JobManager,
    job_id: str | None,
) -> tuple:
    import gradio as gr

    empty_fig = _empty_plot("Loss")
    empty = ("Select a job to monitor.", empty_fig, empty_fig, "", "", gr.update(visible=False), "")
    if not job_id:
        return empty

    try:
        job = mgr.get_status(job_id)
    except KeyError:
        return (f"Unknown job: {job_id}", empty_fig, empty_fig, "", "", gr.update(visible=False), "")

    is_running = job.status == JobStatus.RUNNING
    cancel_update = gr.update(visible=is_running)
    history = job.metrics_history

    # --- Status + Progress ---
    badge = _status_badge(job.status.value)
    status_md = f"### {badge}"

    if job.error:
        import html as _html

        escaped = _html.escape(job.error)
        status_md += (
            '\n\n<details open><summary style="color:#dc2626; font-weight:600;">'
            "Error Details</summary>"
            '<pre style="background:#fef2f2; color:#991b1b; padding:12px; '
            "border-radius:8px; border:1px solid #fecaca; overflow-x:auto; "
            f'font-size:0.85em; white-space:pre-wrap;">{escaped}</pre></details>'
        )

    if job.started_at:
        elapsed = time.time() - job.started_at
        mins, secs = divmod(int(elapsed), 60)
        status_md += f"\n\nElapsed: **{mins}m {secs}s**"

    if job.state:
        step = job.state.get("global_step", 0)
        max_steps = job.state.get("max_steps", -1)
        num_epochs = job.state.get("num_epochs", 0)
        epoch = job.state.get("epoch", 0)

        if max_steps > 0 and step > 0:
            pct = min(100, int(step / max_steps * 100))
            eta_str = ""
            if history and len(history) >= 2:
                avg_dt = (history[-1]["timestamp"] - history[0]["timestamp"]) / len(history)
                remaining = max_steps - step
                eta_s = int(remaining * avg_dt)
                eta_m, eta_sec = divmod(eta_s, 60)
                eta_str = f" | ETA: {eta_m}m {eta_sec}s"
            status_md += (
                '\n\n<div style="background:#e5e7eb; border-radius:8px; '
                'height:20px; margin:8px 0;">'
                f'<div style="background:#4f46e5; border-radius:8px; '
                f"height:20px; width:{pct}%; min-width:2px; "
                f'transition:width 0.3s;"></div></div>'
                f"**Step {step} / {max_steps}** ({pct}%){eta_str}"
            )
        elif num_epochs > 0:
            pct = min(100, int((epoch + 1) / num_epochs * 100))
            status_md += f"\n\n**Epoch {epoch + 1} / {num_epochs}** ({pct}%)"

    # --- Loss & LR Plot ---
    loss_fig = _make_loss_plot(history)

    # --- GPU Plot ---
    gpu_fig = _make_gpu_plot(history)

    # --- Throughput ---
    throughput_md = _make_throughput_md(history)

    # --- Metrics ---
    metrics_md = ""
    if job.state and "global_step" in job.state:
        step = job.state["global_step"]
        metrics = job.state.get("metrics", {})
        parts = [f"**Step:** {step}"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"**{k}:** {v:.4f}")
            else:
                parts.append(f"**{k}:** {v}")
        metrics_md = " | ".join(parts)

    # --- Logs ---
    log_lines = job.log_buffer.get_all()
    if log_lines:
        log_text = "\n".join(log_lines[-200:])
    else:
        logs_from_disk = mgr.get_logs(job_id)
        lines = logs_from_disk.splitlines()[-200:] if logs_from_disk else []
        log_text = "\n".join(lines)

    return status_md, loss_fig, gpu_fig, throughput_md, metrics_md, cancel_update, log_text


def _make_loss_plot(history: list[dict[str, Any]]) -> go.Figure:
    if not history:
        return _empty_plot("Loss & Learning Rate")

    steps = [h["step"] for h in history]
    losses = [h.get("loss") for h in history]
    eval_losses = [h.get("eval_loss") for h in history]
    lrs = [h.get("learning_rate") for h in history]

    fig = go.Figure()

    if any(v is not None for v in losses):
        fig.add_trace(go.Scatter(
            x=steps, y=losses, mode="lines",
            name="Loss", line={"color": "#4f46e5", "width": 2}, yaxis="y",
        ))

    if any(v is not None for v in eval_losses):
        eval_steps = [s for s, v in zip(steps, eval_losses) if v is not None]
        eval_vals = [v for v in eval_losses if v is not None]
        fig.add_trace(go.Scatter(
            x=eval_steps, y=eval_vals, mode="markers",
            name="Eval Loss", marker={"color": "#dc2626", "size": 8, "symbol": "diamond"},
            yaxis="y",
        ))

    if any(v is not None for v in lrs):
        fig.add_trace(go.Scatter(
            x=steps, y=lrs, mode="lines",
            name="Learning Rate", line={"color": "#059669", "width": 1, "dash": "dash"},
            yaxis="y2",
        ))

    fig.update_layout(
        title="Loss & Learning Rate",
        xaxis_title="Step",
        yaxis={"title": "Loss", "side": "left"},
        yaxis2={"title": "Learning Rate", "side": "right", "overlaying": "y"},
        template="plotly_white",
        margin={"l": 40, "r": 60, "t": 40, "b": 40},
        legend={"x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,0.8)"},
    )
    return fig


def _make_gpu_plot(history: list[dict[str, Any]]) -> go.Figure:
    if not history:
        return _empty_plot("GPU")

    steps = [h["step"] for h in history]
    mem = [h.get("gpu_memory_mb") for h in history]
    peak = [h.get("gpu_memory_peak_mb") for h in history]
    util = [h.get("gpu_utilization") for h in history]

    has_mem = any(v is not None for v in mem)
    has_util = any(v is not None for v in util)

    if not has_mem and not has_util:
        fig = _empty_plot("GPU")
        fig.add_annotation(text="No GPU metrics available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font={"size": 14, "color": "#9ca3af"})
        return fig

    fig = go.Figure()

    if has_mem:
        fig.add_trace(go.Scatter(
            x=steps, y=mem, mode="lines", fill="tozeroy",
            name="GPU Memory (MB)", line={"color": "#4f46e5", "width": 1},
            fillcolor="rgba(79, 70, 229, 0.15)", yaxis="y",
        ))
        if any(v is not None for v in peak):
            fig.add_trace(go.Scatter(
                x=steps, y=peak, mode="lines",
                name="Peak (MB)", line={"color": "#dc2626", "width": 1, "dash": "dot"},
                yaxis="y",
            ))

    if has_util:
        fig.add_trace(go.Scatter(
            x=steps, y=util, mode="lines",
            name="Utilization (%)", line={"color": "#059669", "width": 2},
            yaxis="y2" if has_mem else "y",
        ))

    layout_kwargs: dict[str, Any] = {
        "title": "GPU Memory & Utilization",
        "xaxis_title": "Step",
        "template": "plotly_white",
        "margin": {"l": 40, "r": 60, "t": 40, "b": 40},
        "legend": {"x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,0.8)"},
    }
    if has_mem:
        layout_kwargs["yaxis"] = {"title": "Memory (MB)", "side": "left"}
    if has_util and has_mem:
        layout_kwargs["yaxis2"] = {
            "title": "Utilization (%)", "side": "right",
            "overlaying": "y", "range": [0, 100],
        }
    elif has_util:
        layout_kwargs["yaxis"] = {"title": "Utilization (%)", "range": [0, 100]}

    fig.update_layout(**layout_kwargs)
    return fig


def _make_throughput_md(history: list[dict[str, Any]]) -> str:
    if not history:
        return ""

    parts = ["### Throughput"]
    last = history[-1]

    sps = last.get("samples_per_sec")
    if sps is not None:
        parts.append(f"**Samples/sec:** {sps:.1f}")

    steps_sec = last.get("steps_per_sec")
    if steps_sec is not None:
        parts.append(f"**Steps/sec:** {steps_sec:.2f}")

    if len(history) >= 2:
        total_time = history[-1]["timestamp"] - history[0]["timestamp"]
        total_steps = len(history)
        if total_time > 0:
            avg_sps = total_steps / total_time
            parts.append(f"**Avg steps/sec:** {avg_sps:.2f}")

            step = last.get("step", 0)
            max_steps_h = [h for h in history if h.get("step", 0) > 0]
            if max_steps_h:
                avg_step_time = total_time / total_steps
                parts.append(f"**Avg step time:** {avg_step_time:.2f}s")

    gpu_mem = last.get("gpu_memory_mb")
    if gpu_mem is not None:
        parts.append(f"**GPU Memory:** {gpu_mem:.0f} MB")
        peak = last.get("gpu_memory_peak_mb")
        if peak is not None:
            parts.append(f"**Peak:** {peak:.0f} MB")

    gpu_util = last.get("gpu_utilization")
    if gpu_util is not None:
        parts.append(f"**GPU Utilization:** {gpu_util:.0f}%")

    return "\n\n".join(parts)


def _empty_plot(title: str = "Training Loss") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title="Step",
        yaxis_title="",
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
    )
    return fig
