from __future__ import annotations

from typing import Any

import torch

from trainlib.config.schema import TrainerConfig
from trainlib.trainer.callbacks import CallbackManager, TrainState


class Trainer:
    def __init__(
        self,
        config: TrainerConfig,
        callback_manager: CallbackManager | None = None,
    ) -> None:
        self.config = config
        self.callback_manager = callback_manager or CallbackManager()

    def compute_total_steps(
        self,
        dataset_size: int,
        batch_size: int,
    ) -> int:
        steps_per_epoch = dataset_size // batch_size
        if self.config.gradient_accumulation > 1:
            steps_per_epoch = steps_per_epoch // self.config.gradient_accumulation
        total = steps_per_epoch * self.config.num_epochs
        if self.config.max_steps > 0:
            total = min(total, self.config.max_steps)
        return total

    def train(
        self,
        *,
        model: Any,
        train_dataloader: Any,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
    ) -> TrainState:
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        state = TrainState(
            num_epochs=self.config.num_epochs,
            max_steps=self.config.max_steps,
        )

        self.callback_manager.fire("train_start", state)

        for epoch in range(self.config.num_epochs):
            state.epoch = epoch
            self.callback_manager.fire("epoch_start", state)

            for step, batch in enumerate(train_dataloader):
                state.step = step
                self.callback_manager.fire("step_start", state)

                loss = self._training_step(model, batch, optimizer, state)
                state.metrics["loss"] = loss

                self.callback_manager.fire("step_end", state)
                state.global_step += 1

                if self.config.max_steps > 0 and state.global_step >= self.config.max_steps:
                    state.stop_training()

                if state.should_stop:
                    break

            self.callback_manager.fire("epoch_end", state)
            if state.should_stop:
                break

        self.callback_manager.fire("train_end", state)
        return state

    def _training_step(
        self,
        model: Any,
        batch: dict[str, Any],
        optimizer: Any,
        state: TrainState,
    ) -> float:
        outputs = model(**batch) if isinstance(batch, dict) else model(batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs

        if self.config.gradient_accumulation > 1:
            loss = loss / self.config.gradient_accumulation

        loss.backward()

        if (state.step + 1) % self.config.gradient_accumulation == 0 or state.step == 0:
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.max_grad_norm
                )
            optimizer.step()
            optimizer.zero_grad()

        return loss.item() if hasattr(loss, "item") else float(loss)
