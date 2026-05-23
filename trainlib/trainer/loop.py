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
        resume_state: TrainState | None = None,
        resume_checkpoint_dir: str | None = None,
    ) -> TrainState:
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        self._optimizer = optimizer

        # Determine device type for autocast
        self._device_type = "cpu"
        try:
            first_param = next(iter(model.parameters()))
            if first_param.is_cuda:
                self._device_type = "cuda"
        except (StopIteration, TypeError):
            pass

        # Set up mixed precision
        self._amp_dtype: torch.dtype | None = None
        self._scaler: torch.amp.GradScaler | None = None
        if self.config.mixed_precision == "fp16":
            self._amp_dtype = torch.float16
            if self._device_type == "cuda":
                self._scaler = torch.amp.GradScaler()
        elif self.config.mixed_precision == "bf16":
            self._amp_dtype = torch.bfloat16
        # fp32 → no autocast, no scaler

        # Resume optimizer/scaler state from checkpoint
        if resume_checkpoint_dir:
            from pathlib import Path

            opt_path = Path(resume_checkpoint_dir) / "optimizer.pt"
            if opt_path.exists():
                optimizer.load_state_dict(torch.load(opt_path, weights_only=True))
            scaler_path = Path(resume_checkpoint_dir) / "scaler.pt"
            if self._scaler is not None and scaler_path.exists():
                self._scaler.load_state_dict(torch.load(scaler_path, weights_only=True))

        if resume_state is not None:
            state = TrainState(
                step=resume_state.step,
                epoch=resume_state.epoch,
                global_step=resume_state.global_step,
                num_epochs=self.config.num_epochs,
                max_steps=self.config.max_steps,
                metrics=dict(resume_state.metrics),
            )
        else:
            state = TrainState(
                num_epochs=self.config.num_epochs,
                max_steps=self.config.max_steps,
            )

        # Determine the step to resume from in the first epoch
        resumed_step = resume_state.step if resume_state is not None else -1

        self.callback_manager.fire("train_start", state)

        for epoch in range(state.epoch, self.config.num_epochs):
            state.epoch = epoch
            self.callback_manager.fire("epoch_start", state)

            for step, batch in enumerate(train_dataloader):
                # Skip already-completed steps in the resumed epoch
                if epoch == (resume_state.epoch if resume_state else -1) and step <= resumed_step:
                    continue

                state.step = step
                self.callback_manager.fire("step_start", state)

                loss = self._training_step(model, batch, optimizer, state)
                state.metrics["loss"] = loss
                state.global_step += 1

                self.callback_manager.fire("step_end", state)

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
        # Forward pass with optional autocast
        if self._amp_dtype is not None:
            with torch.amp.autocast(self._device_type, dtype=self._amp_dtype):
                outputs = model(**batch) if isinstance(batch, dict) else model(batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs
        else:
            outputs = model(**batch) if isinstance(batch, dict) else model(batch)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs

        if self.config.gradient_accumulation > 1:
            loss = loss / self.config.gradient_accumulation

        # Backward with optional scaler
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()

        if (state.step + 1) % self.config.gradient_accumulation == 0 or state.step == 0:
            if self._scaler is not None:
                if self.config.max_grad_norm > 0:
                    self._scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                optimizer.step()
            optimizer.zero_grad()

        return loss.item() if hasattr(loss, "item") else float(loss)
