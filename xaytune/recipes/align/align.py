from __future__ import annotations

import copy
from typing import Any

from xaytune.config.schema import (
    DataConfig,
    GenerationConfig,
    ModelConfig,
    OnlineRLConfig,
    TrainConfig,
    TrainerConfig,
)
from xaytune.recipes import base as _base
from xaytune.recipes.align.loss_dispatch import (
    _RL_METHODS,
    create_alignment_loss_fn,
    is_alignment_method,
)
from xaytune.trainer.callbacks import TrainState


def align(
    *,
    config: TrainConfig | None = None,
    model: Any | None = None,
    tokenizer: Any | None = None,
    dataset: str | None = None,
    method: str = "dpo",
    format: str = "preference",
    num_epochs: int = 1,
    learning_rate: float = 5e-6,
    batch_size: int = 4,
    resume_from: str | None = None,
    **kwargs: Any,
) -> TrainState:
    """Align a language model using preference-based or RL methods.

    Supports DPO, SimPO, ORPO, PPO, GRPO, and REINFORCE.  A frozen
    reference model is created automatically for methods that need one.
    Method-specific hyperparameters (``beta``, ``kl_coeff``, etc.) are
    extracted from ``**kwargs`` and forwarded to the loss function.

    Args:
        config: Complete training configuration. When provided, all other
            arguments except ``resume_from`` are ignored.
        model: HuggingFace model name or local path.
        dataset: Path to a preference JSONL file (each line:
            ``{"prompt": "...", "chosen": "...", "rejected": "..."}``).
        method: Alignment method — ``"dpo"``, ``"simpo"``, ``"orpo"``,
            ``"ppo"``, ``"grpo"``, or ``"reinforce"``.
        format: Data format — ``"preference"`` for paired data.
        num_epochs: Number of training epochs.
        learning_rate: Peak learning rate.
        batch_size: Per-device batch size.
        resume_from: Path to a checkpoint directory to resume from.
        **kwargs: Method hyperparameters (``beta``, ``kl_coeff``,
            ``lambda_weight``, ``gamma``, ``clip_eps``) and any extra
            ``TrainerConfig`` fields.

    Returns:
        Final training state with loss, global step count, and other metrics.

    Raises:
        ValueError: If neither ``config`` nor both ``model`` and ``dataset``
            are provided.

    Example::

        state = xaytune.align(
            model="meta-llama/Llama-3-8B",
            dataset="data/prefs.jsonl",
            method="dpo",
            beta=0.1,
            max_steps=200,
        )
    """
    injected_model = None
    if config is None:
        if dataset is None:
            raise ValueError("Either 'config' or both 'model' and 'dataset' are required.")

        model_name = model if isinstance(model, str) else "custom"
        if not isinstance(model, str) and model is not None:
            injected_model = model
        elif model is None:
            raise ValueError("Either 'config' or both 'model' and 'dataset' are required.")

        trainer_fields = {}
        method_params = {}
        trainer_param_names = {f for f in TrainerConfig.model_fields}
        method_param_names = {
            "beta",
            "kl_coeff",
            "lambda_weight",
            "gamma",
            "clip_eps",
        }
        online_rl_param_names = {
            "reward_name",
            "reward_kwargs",
            "group_size",
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "do_sample",
        }
        online_rl_params: dict[str, Any] = {}
        for k in list(kwargs.keys()):
            if k in trainer_param_names:
                trainer_fields[k] = kwargs.pop(k)
            elif k in method_param_names:
                method_params[k] = kwargs.pop(k)
            elif k in online_rl_param_names:
                online_rl_params[k] = kwargs.pop(k)

        online_rl_config = OnlineRLConfig()
        if online_rl_params:
            gen_fields = {}
            rl_fields: dict[str, Any] = {"enabled": True}
            gen_param_names = {
                "max_new_tokens",
                "temperature",
                "top_p",
                "top_k",
                "do_sample",
                "group_size",
            }
            for k, v in online_rl_params.items():
                if k in gen_param_names:
                    gen_fields[k] = v
                else:
                    rl_fields[k] = v
            if gen_fields:
                rl_fields["generation"] = GenerationConfig(**gen_fields)
            online_rl_config = OnlineRLConfig(**rl_fields)

        config = TrainConfig(
            recipe="align",
            method=method,
            model=ModelConfig(name=model_name),
            data=DataConfig(path=dataset, format=format),
            trainer=TrainerConfig(
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                **trainer_fields,
            ),
            method_params=method_params,
            online_rl=online_rl_config,
        )

    components = _base.setup_training(
        config,
        resume_from=resume_from,
        model=injected_model,
        tokenizer=tokenizer,
    )

    loss_fn: Any = None
    if is_alignment_method(config.method):
        ref_model = copy.deepcopy(components.model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

        if config.online_rl.enabled and config.method in _RL_METHODS:
            from xaytune.recipes.align.online_step import OnlineRLStep

            loss_fn = OnlineRLStep(
                ref_model=ref_model,
                tokenizer=components.tokenizer,
                method=config.method,
                generation_config=config.online_rl.generation,
                reward_name=config.online_rl.reward_name,
                reward_kwargs=config.online_rl.reward_kwargs,
                kl_coeff=config.method_params.get("kl_coeff", 0.04),
                clip_eps=config.method_params.get("clip_eps", 0.2),
            )
        else:
            loss_fn = create_alignment_loss_fn(
                method=config.method,
                ref_model=ref_model,
                **config.method_params,
            )

    state = components.trainer.train(
        model=components.model,
        train_dataloader=components.train_dataloader,
        loss_fn=loss_fn,
        resume_state=components.resume_state,
        resume_checkpoint_dir=resume_from,
    )

    return state
