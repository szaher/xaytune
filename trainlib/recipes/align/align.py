from __future__ import annotations

import copy
from typing import Any

from trainlib.config.schema import (
    DataConfig,
    ModelConfig,
    TrainConfig,
    TrainerConfig,
)
from trainlib.recipes import base as _base
from trainlib.recipes.align.loss_dispatch import (
    create_alignment_loss_fn,
    is_alignment_method,
)
from trainlib.trainer.callbacks import TrainState


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

        state = trainlib.align(
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
            raise ValueError(
                "Either 'config' or both 'model' and 'dataset' are required."
            )

        model_name = model if isinstance(model, str) else "custom"
        if not isinstance(model, str) and model is not None:
            injected_model = model
        elif model is None:
            raise ValueError(
                "Either 'config' or both 'model' and 'dataset' are required."
            )

        trainer_fields = {}
        method_params = {}
        trainer_param_names = {f for f in TrainerConfig.model_fields}
        method_param_names = {
            "beta", "kl_coeff", "lambda_weight", "gamma", "clip_eps",
        }
        for k in list(kwargs.keys()):
            if k in trainer_param_names:
                trainer_fields[k] = kwargs.pop(k)
            elif k in method_param_names:
                method_params[k] = kwargs.pop(k)

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
        )

    components = _base.setup_training(
        config, resume_from=resume_from,
        model=injected_model, tokenizer=tokenizer,
    )

    loss_fn = None
    if is_alignment_method(config.method):
        ref_model = copy.deepcopy(components.model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

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
