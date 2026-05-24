from __future__ import annotations

import time
from typing import Any

import gradio as gr
import plotly.graph_objects as go

from trainlib.config.schema import (
    DataConfig,
    EvalConfig,
    LoggingConfig,
    LoraConfig,
    ModelConfig,
    OutputConfig,
    TrainConfig,
    TrainerConfig,
)
from trainlib.studio.jobs import JobManager

RECIPE_METHODS: dict[str, list[str]] = {
    "finetune": ["full", "lora", "qlora"],
    "pretrain": ["full"],
    "align": ["dpo", "grpo", "ppo", "orpo", "simpo"],
}

METHOD_PARAMS_SPEC: dict[str, list[dict[str, Any]]] = {
    "dpo": [{"name": "beta", "default": 0.1, "info": "Preference strength temperature."}],
    "grpo": [{"name": "kl_coeff", "default": 0.04, "info": "KL divergence penalty weight."}],
    "ppo": [{"name": "clip_eps", "default": 0.2, "info": "Policy clipping range."}],
    "orpo": [{"name": "lambda_weight", "default": 1.0, "info": "Odds-ratio loss weight."}],
    "simpo": [
        {"name": "beta", "default": 2.0, "info": "Preference scaling."},
        {"name": "gamma", "default": 0.5, "info": "Sigmoid offset."},
    ],
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
            f"**{recipe}** recipe requires method: {', '.join(valid_methods)} "
            f"(got '{method}')"
        )

    if method == "qlora" and quantization != "4bit":
        errors.append("**QLoRA** requires quantization set to **4bit**")

    if learning_rate <= 0:
        errors.append("**Learning Rate** must be greater than 0")

    if batch_size < 1:
        errors.append("**Batch Size** must be at least 1")

    if warmup_steps > 0 and warmup_ratio > 0:
        errors.append(
            "**Warmup Steps** and **Warmup Ratio** are mutually exclusive — set one to 0"
        )

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


def create_app(job_manager: JobManager | None = None) -> gr.Blocks:
    mgr = job_manager or JobManager()

    with gr.Blocks(title="trainlib Studio") as app:
        gr.Markdown(
            "# trainlib Studio\n"
            "Configure, launch, and monitor LLM training jobs."
        )

        # ── Train Tab ──────────────────────────────────────────────
        with gr.Tab("Train"):
            mode = gr.Radio(
                choices=["Simple", "Advanced"],
                value="Simple",
                label="Mode",
                info="Simple: essential options only. Advanced: full control.",
            )

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
                        choices=["alpaca", "sharegpt", "completion", "pretrain"],
                        value="alpaca",
                        label="Data Format",
                        info="alpaca: instruction/output. sharegpt: chat. completion: raw text.",
                    )
                    source = gr.Dropdown(
                        choices=["local", "huggingface"],
                        value="local",
                        label="Source",
                        info="Where to load data from.",
                    )
                with gr.Row():
                    max_seq_length = gr.Number(
                        value=2048, label="Max Sequence Length", precision=0,
                        minimum=64, maximum=131072, step=64,
                        info="Maximum token length per sample. Longer sequences use more memory.",
                    )
                    eval_split = gr.Number(
                        value=0.0, label="Eval Split", precision=2,
                        minimum=0.0, maximum=1.0, step=0.05,
                        info="Fraction of data for evaluation (0.0 = no eval split, 0.1 = 10%).",
                    )
                    eval_path = gr.Textbox(
                        label="Eval Data Path",
                        placeholder="(optional) data/eval.jsonl",
                        info="Separate evaluation dataset. Overrides eval_split if set.",
                    )
                with gr.Row():
                    packing = gr.Checkbox(
                        value=True, label="Sequence Packing",
                        info="Pack multiple short samples into one sequence for efficiency.",
                    )
                    streaming = gr.Checkbox(
                        value=False, label="Streaming",
                        info="Stream data instead of loading all into memory.",
                    )

            with gr.Accordion(
                "LoRA", open=True, visible=False, elem_id="lora_accordion"
            ) as lora_accordion:
                with gr.Row():
                    lora_rank = gr.Number(
                        value=16, label="Rank", precision=0,
                        minimum=1, maximum=256, step=1,
                        info="LoRA rank. Higher = more capacity, slower. 8-64 typical.",
                    )
                    lora_alpha = gr.Number(
                        value=32, label="Alpha", precision=0,
                        minimum=1, maximum=512, step=1,
                        info="LoRA scaling factor. Common rule: alpha = 2 * rank.",
                    )
                    lora_dropout = gr.Number(
                        value=0.05, label="Dropout", precision=2,
                        minimum=0.0, maximum=1.0, step=0.01,
                        info="Dropout probability for LoRA layers. 0.0-0.1 typical.",
                    )

            method.change(
                fn=_on_method_change,
                inputs=[method],
                outputs=[lora_accordion, quantization],
            )

            with gr.Accordion(
                "Method Parameters", open=True, visible=False,
            ) as method_params_accordion:
                mp_info = gr.Markdown("Select an alignment method to configure parameters.")
                mp_beta = gr.Number(
                    value=0.1, label="beta", visible=False,
                    info="Preference strength temperature.",
                )
                mp_kl_coeff = gr.Number(
                    value=0.04, label="kl_coeff", visible=False,
                    info="KL divergence penalty weight.",
                )
                mp_clip_eps = gr.Number(
                    value=0.2, label="clip_eps", visible=False,
                    info="Policy clipping range.",
                )
                mp_lambda_weight = gr.Number(
                    value=1.0, label="lambda_weight", visible=False,
                    info="Odds-ratio loss weight.",
                )
                mp_gamma = gr.Number(
                    value=0.5, label="gamma", visible=False,
                    info="Sigmoid offset.",
                )

            _mp_fields = {
                "beta": mp_beta, "kl_coeff": mp_kl_coeff,
                "clip_eps": mp_clip_eps, "lambda_weight": mp_lambda_weight,
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
                    method_params_accordion, mp_info,
                    mp_beta, mp_kl_coeff, mp_clip_eps, mp_lambda_weight, mp_gamma,
                ],
            )

            with gr.Column(visible=False) as advanced_column:
                with gr.Accordion("Training", open=False):
                    with gr.Row():
                        batch_size = gr.Number(
                            value=4, label="Batch Size", precision=0,
                            minimum=1, step=1,
                            info="Samples per device per step. Reduce if out of memory.",
                        )
                        grad_accum = gr.Number(
                            value=1, label="Gradient Accumulation", precision=0,
                            minimum=1, step=1,
                            info="Accumulate gradients over N steps before updating.",
                        )
                        learning_rate = gr.Number(
                            value=2e-4, label="Learning Rate",
                            minimum=1e-7, maximum=1.0,
                            info="Typical: 1e-5 (pretrain), 2e-4 (finetune), 5e-5 (align).",
                        )
                    with gr.Row():
                        num_epochs = gr.Number(
                            value=3, label="Num Epochs", precision=0,
                            minimum=1, step=1,
                            info="Number of full passes through the dataset.",
                        )
                        max_steps = gr.Number(
                            value=-1, label="Max Steps", precision=0,
                            minimum=-1, step=1,
                            info="Stop after N steps regardless of epochs. -1 = no limit.",
                        )
                        seed = gr.Number(
                            value=42, label="Seed", precision=0,
                            minimum=0, step=1,
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
                            value=0, label="Warmup Steps", precision=0,
                            minimum=0, step=1,
                            info="Ramp LR from 0 over N steps. Exclusive with warmup ratio.",
                        )
                        warmup_ratio = gr.Number(
                            value=0.0, label="Warmup Ratio", precision=2,
                            minimum=0.0, maximum=1.0, step=0.01,
                            info="Warmup as fraction of total steps.",
                        )
                        weight_decay = gr.Number(
                            value=0.01, label="Weight Decay",
                            minimum=0.0, maximum=1.0, step=0.001,
                            info="L2 regularization strength. 0.01-0.1 typical.",
                        )
                        max_grad_norm = gr.Number(
                            value=1.0, label="Max Grad Norm",
                            minimum=0.0, step=0.1,
                            info="Gradient clipping threshold. 1.0 is standard.",
                        )

                with gr.Accordion("Evaluation", open=False):
                    with gr.Row():
                        eval_every = gr.Number(
                            value=500, label="Eval Every N Steps", precision=0,
                            minimum=1, step=10,
                            info="Run evaluation every N training steps.",
                        )
                        es_patience = gr.Number(
                            value=0, label="Early Stopping Patience", precision=0,
                            minimum=0, step=1,
                            info="Stop after N evals without improvement. 0 = disabled.",
                        )
                    with gr.Row():
                        es_metric = gr.Textbox(
                            value="eval_loss", label="Early Stopping Metric",
                            info="Metric to monitor for early stopping.",
                        )
                        es_min_delta = gr.Number(
                            value=0.0, label="Min Delta",
                            minimum=0.0, step=0.001,
                            info="Minimum change to count as improvement.",
                        )

                with gr.Accordion("Logging & Output", open=False):
                    with gr.Row():
                        log_every = gr.Number(
                            value=10, label="Log Every N Steps", precision=0,
                            minimum=1, step=1,
                            info="How often to log metrics to console/backends.",
                        )
                        output_dir = gr.Textbox(
                            value="output", label="Output Directory",
                            info="Where to save checkpoints and final model.",
                        )
                        merge_on_complete = gr.Checkbox(
                            value=False, label="Merge on Complete",
                            info="Merge LoRA weights into base after training finishes.",
                        )

                with gr.Accordion("Advanced", open=False):
                    with gr.Row():
                        checkpoint_every = gr.Number(
                            value=500, label="Checkpoint Every N Steps", precision=0,
                            minimum=1, step=50,
                            info="Save a checkpoint every N training steps.",
                        )
                        save_last = gr.Checkbox(
                            value=True, label="Save Last Checkpoint",
                            info="Always save a checkpoint at the end of training.",
                        )
                    with gr.Row():
                        activation_ckpt = gr.Checkbox(
                            value=False, label="Activation Checkpointing",
                            info="Recompute activations during backward to save memory.",
                        )
                        async_ckpt = gr.Checkbox(
                            value=False, label="Async Checkpoint",
                            info="Save checkpoints in background to avoid blocking.",
                        )

            mode.change(
                fn=lambda m: gr.update(visible=m == "Advanced"),
                inputs=[mode],
                outputs=[advanced_column],
            )

            validation_output = gr.Markdown("")
            submit_btn = gr.Button("Start Training", variant="primary", size="lg")
            train_status = gr.Markdown("")

            all_inputs = [
                recipe, method, model_name, data_path, data_format,
                quantization, dtype, trust_remote_code,
                source, max_seq_length, packing, streaming, eval_split, eval_path,
                lora_rank, lora_alpha, lora_dropout,
                batch_size, grad_accum, learning_rate,
                num_epochs, max_steps, seed,
                mixed_precision, scheduler,
                warmup_steps, warmup_ratio, weight_decay, max_grad_norm,
                eval_every, es_patience, es_metric, es_min_delta,
                log_every, output_dir, merge_on_complete,
                checkpoint_every, save_last, activation_ckpt, async_ckpt,
                mp_beta, mp_kl_coeff, mp_clip_eps, mp_lambda_weight, mp_gamma,
            ]

            def _submit(
                recipe_v, method_v, model_v, data_v, format_v,
                quant_v, dtype_v, trust_v,
                source_v, seq_len_v, packing_v, streaming_v, esplit_v, epath_v,
                lr_rank_v, lr_alpha_v, lr_drop_v,
                bs_v, ga_v, lr_v,
                epochs_v, steps_v, seed_v,
                mp_v, sched_v,
                warmup_v, wratio_v, wd_v, gn_v,
                eval_v, esp_v, esm_v, esd_v,
                log_v, out_v, merge_v,
                ckpt_v, slast_v, actckpt_v, asyncckpt_v,
                beta_v, kl_coeff_v, clip_eps_v, lambda_w_v, gamma_v,
            ):
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
                        f'border:1px solid #dc2626; border-radius:8px; '
                        f'background:#fef2f2;">\n\n'
                        f"**Validation Errors:**\n\n{error_md}\n\n</div>"
                    ), ""

                mparams: dict[str, Any] = {}
                spec = METHOD_PARAMS_SPEC.get(method_v, [])
                param_values = {
                    "beta": beta_v, "kl_coeff": kl_coeff_v,
                    "clip_eps": clip_eps_v, "lambda_weight": lambda_w_v,
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
                        f'border:1px solid #dc2626; border-radius:8px; '
                        f'background:#fef2f2;">\n\n'
                        f"**Config error:** {e}\n\n</div>"
                    ), ""

                job_id = mgr.submit(config)
                return "", (
                    f'<div style="color:#16a34a; padding:8px; '
                    f'border:1px solid #16a34a; border-radius:8px; '
                    f'background:#f0fdf4;">\n\n'
                    f"**Job submitted:** `{job_id}`\n\n</div>"
                )

            submit_btn.click(
                fn=_submit,
                inputs=all_inputs,
                outputs=[validation_output, train_status],
            )

        # ── Monitor Tab ────────────────────────────────────────────
        with gr.Tab("Monitor"):
            with gr.Row():
                job_dropdown = gr.Dropdown(
                    choices=[], label="Job ID", allow_custom_value=True,
                    info="Select a running or completed job to monitor.",
                )
                refresh_jobs_btn = gr.Button("Refresh Jobs", size="sm")

            def _refresh_job_list():
                jobs = mgr.list_jobs()
                ids = [j.job_id[:8] + "..." for j in jobs]
                full_ids = [j.job_id for j in jobs]
                choices = list(zip(ids, full_ids)) if ids else []
                return gr.update(choices=choices, value=full_ids[0] if full_ids else None)

            refresh_jobs_btn.click(fn=_refresh_job_list, outputs=[job_dropdown])

            monitor_status = gr.Markdown("Select a job to monitor.")
            loss_plot = gr.Plot(label="Training Loss")
            metrics_display = gr.Markdown("")
            history_state = gr.State([])

            timer = gr.Timer(value=2, active=False)

            timer.tick(
                fn=lambda job_id, history: _poll(mgr, job_id, history),
                inputs=[job_dropdown, history_state],
                outputs=[monitor_status, loss_plot, metrics_display, history_state],
            )

            job_dropdown.change(
                fn=lambda _: ([], True),
                inputs=[job_dropdown],
                outputs=[history_state, timer],
            )

        # ── History Tab ────────────────────────────────────────────
        with gr.Tab("History"):
            refresh_history_btn = gr.Button("Refresh", size="sm")
            history_table = gr.Dataframe(
                headers=["Job ID", "Recipe", "Status", "Created", "Completed", "Final Loss"],
                label="Training Jobs",
            )

            def _refresh_history():
                jobs = mgr.list_jobs()
                rows = []
                for j in jobs:
                    loss = "-"
                    if j.state and "metrics" in j.state:
                        loss_val = j.state["metrics"].get("loss")
                        if isinstance(loss_val, (int, float)):
                            loss = f"{loss_val:.4f}"
                    rows.append([
                        j.job_id[:8],
                        j.recipe,
                        j.status.value,
                        _format_time(j.created_at),
                        _format_time(j.completed_at),
                        loss,
                    ])
                return rows

            refresh_history_btn.click(fn=_refresh_history, outputs=[history_table])

    return app  # type: ignore[no-any-return]


def _poll(
    mgr: JobManager,
    job_id: str | None,
    history: list[dict[str, Any]],
) -> tuple[str, go.Figure, str, list[dict[str, Any]]]:
    if not job_id:
        return "Select a job to monitor.", _empty_plot(), "", history

    try:
        job = mgr.get_status(job_id)
    except KeyError:
        return f"Unknown job: {job_id}", _empty_plot(), "", history

    badge = _status_badge(job.status.value)
    status_md = f"### {badge}"

    if job.error:
        status_md += (
            f'\n\n<div style="color:#dc2626; padding:6px; '
            f'border-left:3px solid #dc2626; background:#fef2f2;">'
            f"{job.error}</div>"
        )

    if job.started_at and job.status.value == "running":
        elapsed = time.time() - job.started_at
        mins, secs = divmod(int(elapsed), 60)
        status_md += f"\n\nElapsed: **{mins}m {secs}s**"

    if job.state and "global_step" in job.state:
        step = job.state["global_step"]
        metrics = job.state.get("metrics", {})

        seen_steps = {h["step"] for h in history}
        if step not in seen_steps and "loss" in metrics:
            history = [*history, {"step": step, "loss": metrics["loss"]}]

        metrics_lines = [f"**Step:** {step}"]
        for k, v in metrics.items():
            if isinstance(v, float):
                metrics_lines.append(f"**{k}:** {v:.4f}")
            else:
                metrics_lines.append(f"**{k}:** {v}")
        metrics_md = " | ".join(metrics_lines)
    else:
        metrics_md = ""

    if history:
        steps = [h["step"] for h in history]
        losses = [h["loss"] for h in history]
        fig = go.Figure(go.Scatter(
            x=steps, y=losses, mode="lines+markers",
            line={"color": "#4f46e5", "width": 2},
            marker={"size": 4},
        ))
        fig.update_layout(
            title="Training Loss",
            xaxis_title="Step",
            yaxis_title="Loss",
            template="plotly_white",
            margin={"l": 40, "r": 20, "t": 40, "b": 40},
        )
    else:
        fig = _empty_plot()

    return status_md, fig, metrics_md, history


def _empty_plot() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title="Training Loss",
        xaxis_title="Step",
        yaxis_title="Loss",
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
    )
    return fig
