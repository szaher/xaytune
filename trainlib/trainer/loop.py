from __future__ import annotations

from typing import Any

import torch

from trainlib.config.schema import TrainerConfig
from trainlib.trainer.callbacks import CallbackManager, TrainState
from trainlib.trainer.device import (
    detect_device_type_from_model,
    supports_amp,
    supports_grad_scaler,
)
from trainlib.trainer.scheduler import create_scheduler, resolve_warmup_steps


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
        loss_fn: Any | None = None,
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
        self._loss_fn = loss_fn

        # Determine device type for autocast
        self._device_type = detect_device_type_from_model(model)

        # Set up mixed precision
        self._amp_dtype: torch.dtype | None = None
        self._scaler: torch.amp.GradScaler | None = None
        if self.config.mixed_precision == "fp16":
            self._amp_dtype = torch.float16
        elif self.config.mixed_precision == "bf16":
            self._amp_dtype = torch.bfloat16

        if self._amp_dtype is not None and not supports_amp(self._device_type):
            self._amp_dtype = None
        if supports_grad_scaler(self._device_type, self._amp_dtype):
            self._scaler = torch.amp.GradScaler()

        # Create learning rate scheduler
        if scheduler is None:
            num_batches = len(train_dataloader)
            steps_per_epoch = num_batches
            if self.config.gradient_accumulation > 1:
                steps_per_epoch = num_batches // self.config.gradient_accumulation
            total_steps = steps_per_epoch * self.config.num_epochs
            if self.config.max_steps > 0:
                total_steps = min(total_steps, self.config.max_steps)
            warmup = resolve_warmup_steps(
                self.config.warmup_steps,
                self.config.warmup_ratio,
                total_steps,
            )
            scheduler = create_scheduler(
                optimizer, self.config.scheduler, total_steps, warmup
            )
        self._scheduler = scheduler

        # Resume optimizer/scaler/scheduler state from checkpoint
        if resume_checkpoint_dir:
            from pathlib import Path

            opt_path = Path(resume_checkpoint_dir) / "optimizer.pt"
            if opt_path.exists():
                optimizer.load_state_dict(torch.load(opt_path, weights_only=True))
            scaler_path = Path(resume_checkpoint_dir) / "scaler.pt"
            if self._scaler is not None and scaler_path.exists():
                self._scaler.load_state_dict(torch.load(scaler_path, weights_only=True))
            scheduler_path = Path(resume_checkpoint_dir) / "scheduler.pt"
            if self._scheduler is not None and scheduler_path.exists():
                self._scheduler.load_state_dict(torch.load(scheduler_path, weights_only=True))

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

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        device = torch.device(self._device_type)
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _training_step(
        self,
        model: Any,
        batch: dict[str, Any],
        optimizer: Any,
        state: TrainState,
    ) -> float:
        if isinstance(batch, dict):
            batch = self._move_batch_to_device(batch)

        # Skip forward pass for preference batches — alignment loss_fn does its own
        skip_forward = (
            self._loss_fn is not None
            and isinstance(batch, dict)
            and "chosen_input_ids" in batch
        )

        # Forward pass with optional autocast
        if self._amp_dtype is not None:
            with torch.amp.autocast(self._device_type, dtype=self._amp_dtype):
                if skip_forward:
                    loss = self._loss_fn(model, batch, None)
                elif isinstance(batch, dict):
                    outputs = model(**batch)
                    if self._loss_fn is not None:
                        loss = self._loss_fn(model, batch, outputs)
                    else:
                        loss = outputs.loss if hasattr(outputs, "loss") else outputs
                else:
                    outputs = model(batch)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs
        else:
            if skip_forward:
                loss = self._loss_fn(model, batch, None)
            elif isinstance(batch, dict):
                outputs = model(**batch)
                if self._loss_fn is not None:
                    loss = self._loss_fn(model, batch, outputs)
                else:
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs
            else:
                outputs = model(batch)
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
            if self._scheduler is not None:
                self._scheduler.step()
                last_lr = self._scheduler.get_last_lr()
                if last_lr:
                    state.metrics["learning_rate"] = last_lr[0]

        return loss.item() if hasattr(loss, "item") else float(loss)
