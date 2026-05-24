from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import torch

from xaytune.config.schema import DataConfig, ModelConfig, OutputConfig, TrainConfig, TrainerConfig
from xaytune.models.loader import ModelResult
from xaytune.recipes.base import TrainingComponents, setup_training
from xaytune.trainer.callbacks import TrainState
from xaytune.trainer.distributed import DistributedContext


class TestTrainingComponents:
    def test_is_namedtuple(self):
        tc = TrainingComponents(
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_dataloader=MagicMock(),
            eval_dataloader=None,
            trainer=MagicMock(),
        )
        assert tc.model is not None
        assert tc.tokenizer is not None
        assert tc.trainer is not None
        assert tc.eval_dataloader is None

    def test_is_namedtuple_with_distributed_ctx(self):
        tc = TrainingComponents(
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_dataloader=MagicMock(),
            eval_dataloader=None,
            trainer=MagicMock(),
            distributed_ctx=DistributedContext(),
        )
        assert tc.distributed_ctx is not None
        assert not tc.distributed_ctx.is_distributed

    def test_distributed_ctx_defaults_to_none(self):
        tc = TrainingComponents(
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_dataloader=MagicMock(),
            eval_dataloader=None,
            trainer=MagicMock(),
        )
        assert tc.distributed_ctx is None

    def test_resume_state_defaults_to_none(self):
        tc = TrainingComponents(
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_dataloader=MagicMock(),
            eval_dataloader=None,
            trainer=MagicMock(),
        )
        assert tc.resume_state is None

    def test_fields(self):
        fields = TrainingComponents._fields
        assert "model" in fields
        assert "tokenizer" in fields
        assert "train_dataloader" in fields
        assert "eval_dataloader" in fields
        assert "trainer" in fields
        assert "distributed_ctx" in fields
        assert "resume_state" in fields


@patch("xaytune.recipes.base.validate_dataset_sample")
class TestSetupTraining:
    def _make_config(self, method="full", **trainer_kwargs):
        return TrainConfig(
            recipe="finetune",
            method=method,
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="data.jsonl", format="alpaca"),
            trainer=TrainerConfig(**trainer_kwargs),
        )

    @patch("xaytune.recipes.base.load_model")
    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.DataLoader")
    def test_full_finetune_setup(self, mock_dl_cls, mock_load_ds, mock_load_model, mock_validate):
        mock_model_result = MagicMock()
        mock_model_result.model = MagicMock()
        mock_model_result.tokenizer = MagicMock()
        mock_load_model.return_value = mock_model_result
        mock_load_ds.return_value = [{"input_ids": [1, 2, 3]}]
        mock_dl_cls.return_value = MagicMock()

        config = self._make_config(method="full")
        components = setup_training(config)

        mock_load_model.assert_called_once_with(
            "test-model",
            quantization=None,
            dtype="auto",
            trust_remote_code=False,
        )
        assert components.model is mock_model_result.model
        assert components.tokenizer is mock_model_result.tokenizer
        assert components.trainer is not None

    @patch("xaytune.recipes.base.apply_lora")
    @patch("xaytune.recipes.base.load_model")
    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.DataLoader")
    def test_lora_setup_applies_peft(
        self, mock_dl_cls, mock_load_ds, mock_load_model, mock_apply_lora, mock_validate
    ):
        mock_model_result = MagicMock()
        mock_model_result.model = MagicMock()
        mock_model_result.tokenizer = MagicMock()
        mock_load_model.return_value = mock_model_result

        lora_result = MagicMock()
        lora_result.model = MagicMock()
        lora_result.tokenizer = mock_model_result.tokenizer
        mock_apply_lora.return_value = lora_result

        mock_load_ds.return_value = [{"input_ids": [1]}]
        mock_dl_cls.return_value = MagicMock()

        config = self._make_config(method="lora")
        components = setup_training(config)

        mock_apply_lora.assert_called_once()
        assert components.model is lora_result.model

    @patch("xaytune.recipes.base.load_model")
    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.DataLoader")
    def test_qlora_uses_4bit_quantization(
        self, mock_dl_cls, mock_load_ds, mock_load_model, mock_validate
    ):
        mock_model_result = MagicMock()
        mock_model_result.model = MagicMock()
        mock_model_result.tokenizer = MagicMock()
        mock_load_model.return_value = mock_model_result
        mock_load_ds.return_value = [{"input_ids": [1]}]
        mock_dl_cls.return_value = MagicMock()

        config = self._make_config(method="qlora")
        with patch("xaytune.recipes.base.apply_lora") as mock_apply_lora:
            mock_apply_lora.return_value = mock_model_result
            setup_training(config)

        mock_load_model.assert_called_once_with(
            "test-model",
            quantization="4bit",
            dtype="auto",
            trust_remote_code=False,
        )

    @patch("xaytune.recipes.base.load_model")
    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.DataLoader")
    def test_eval_split_creates_eval_dataloader(
        self, mock_dl_cls, mock_load_ds, mock_load_model, mock_validate
    ):
        mock_model_result = MagicMock()
        mock_model_result.model = MagicMock()
        mock_model_result.tokenizer = MagicMock()
        mock_load_model.return_value = mock_model_result
        mock_load_ds.return_value = ([{"input_ids": [1]}], [{"input_ids": [2]}])
        mock_dl_cls.return_value = MagicMock()

        config = TrainConfig(
            recipe="finetune",
            method="full",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="data.jsonl", format="alpaca", eval_split=0.1),
            trainer=TrainerConfig(),
        )
        components = setup_training(config)

        assert mock_dl_cls.call_count == 2
        assert components.eval_dataloader is not None

    @patch("xaytune.recipes.base.load_model")
    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.DataLoader")
    def test_no_eval_split_no_eval_dataloader(
        self, mock_dl_cls, mock_load_ds, mock_load_model, mock_validate
    ):
        mock_model_result = MagicMock()
        mock_model_result.model = MagicMock()
        mock_model_result.tokenizer = MagicMock()
        mock_load_model.return_value = mock_model_result
        mock_load_ds.return_value = [{"input_ids": [1]}]
        mock_dl_cls.return_value = MagicMock()

        config = self._make_config(method="full")
        components = setup_training(config)

        assert mock_dl_cls.call_count == 1
        assert components.eval_dataloader is None


# -- Helpers for distributed tests --


def _mock_model_result() -> ModelResult:
    model = MagicMock()
    model.parameters.return_value = [torch.randn(10, requires_grad=True)]
    model.to.return_value = model
    tokenizer = MagicMock()
    return ModelResult(model=model, tokenizer=tokenizer, name="test")


def _mock_dataset() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "input_ids": torch.tensor([1, 2]),
            "labels": torch.tensor([1, 2]),
            "attention_mask": torch.tensor([1, 1]),
        },
        {
            "input_ids": torch.tensor([3, 4]),
            "labels": torch.tensor([3, 4]),
            "attention_mask": torch.tensor([1, 1]),
        },
    ]


def _make_config(**trainer_kwargs: Any) -> TrainConfig:
    trainer_kwargs.setdefault("save_last", False)
    return TrainConfig(
        recipe="finetune",
        model=ModelConfig(name="test-model"),
        data=DataConfig(path="fake.jsonl", format="alpaca"),
        trainer=TrainerConfig(batch_size=2, num_epochs=1, max_steps=1, **trainer_kwargs),
    )


class TestSetupTrainingDistributed:
    @patch("xaytune.recipes.base.wrap_model_distributed")
    @patch("xaytune.recipes.base.init_distributed")
    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.load_model")
    def test_single_gpu_no_wrapping(self, mock_load_model, mock_load_dataset, mock_init, mock_wrap):
        """Single GPU: no model wrapping, no DistributedSampler."""
        mock_init.return_value = DistributedContext()  # default: not distributed
        mock_load_model.return_value = _mock_model_result()
        mock_load_dataset.return_value = _mock_dataset()

        config = _make_config()
        components = setup_training(config)

        mock_wrap.assert_not_called()
        assert components.distributed_ctx is not None
        assert not components.distributed_ctx.is_distributed

    @patch("xaytune.recipes.base.wrap_model_distributed")
    @patch("xaytune.recipes.base.init_distributed")
    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.load_model")
    def test_distributed_wraps_model(
        self, mock_load_model, mock_load_dataset, mock_init, mock_wrap
    ):
        """Distributed: model should be wrapped."""
        ctx = DistributedContext(rank=0, world_size=2, local_rank=0)
        mock_init.return_value = ctx
        mock_load_model.return_value = _mock_model_result()
        mock_load_dataset.return_value = _mock_dataset()
        mock_wrap.return_value = MagicMock()

        config = _make_config()
        components = setup_training(config)

        mock_wrap.assert_called_once()
        assert components.distributed_ctx.is_distributed

    @patch("xaytune.recipes.base.wrap_model_distributed", return_value=MagicMock())
    @patch("xaytune.recipes.base.init_distributed")
    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.load_model")
    def test_distributed_uses_distributed_sampler(
        self, mock_load_model, mock_load_dataset, mock_init, mock_wrap
    ):
        """Distributed: DataLoader should use DistributedSampler."""
        ctx = DistributedContext(rank=0, world_size=2, local_rank=0)
        mock_init.return_value = ctx
        mock_load_model.return_value = _mock_model_result()
        mock_load_dataset.return_value = _mock_dataset()

        config = _make_config()
        components = setup_training(config)

        from torch.utils.data.distributed import DistributedSampler

        assert isinstance(components.train_dataloader.sampler, DistributedSampler)

    @patch("xaytune.recipes.base.wrap_model_distributed", return_value=MagicMock())
    @patch("xaytune.recipes.base.init_distributed")
    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.load_model")
    def test_distributed_eval_dataloader_uses_distributed_sampler(
        self, mock_load_model, mock_load_dataset, mock_init, mock_wrap
    ):
        """Distributed with eval split: eval DataLoader should use DistributedSampler."""
        ctx = DistributedContext(rank=0, world_size=2, local_rank=0)
        mock_init.return_value = ctx
        mock_load_model.return_value = _mock_model_result()
        mock_load_dataset.return_value = (_mock_dataset(), _mock_dataset())

        config = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="fake.jsonl", format="alpaca", eval_split=0.1),
            trainer=TrainerConfig(batch_size=2, num_epochs=1, max_steps=1),
        )
        components = setup_training(config)

        from torch.utils.data.distributed import DistributedSampler

        assert components.eval_dataloader is not None
        assert isinstance(components.eval_dataloader.sampler, DistributedSampler)

    @patch("xaytune.recipes.base.init_distributed")
    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.load_model")
    def test_single_gpu_no_distributed_sampler(self, mock_load_model, mock_load_dataset, mock_init):
        """Single GPU: DataLoader should NOT use DistributedSampler."""
        mock_init.return_value = DistributedContext()
        mock_load_model.return_value = _mock_model_result()
        mock_load_dataset.return_value = _mock_dataset()

        config = _make_config()
        components = setup_training(config)

        from torch.utils.data.distributed import DistributedSampler

        assert not isinstance(components.train_dataloader.sampler, DistributedSampler)

    @patch("xaytune.recipes.base.cleanup_distributed")
    @patch("xaytune.recipes.base.wrap_model_distributed", return_value=MagicMock())
    @patch("xaytune.recipes.base.init_distributed")
    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.load_model")
    def test_cleanup_registered_on_train_end(
        self, mock_load_model, mock_load_dataset, mock_init, mock_wrap, mock_cleanup
    ):
        """Distributed: cleanup_distributed should fire on train_end."""
        ctx = DistributedContext(rank=0, world_size=2, local_rank=0)
        mock_init.return_value = ctx
        mock_load_model.return_value = _mock_model_result()
        mock_load_dataset.return_value = _mock_dataset()

        config = _make_config()
        components = setup_training(config)

        # Fire train_end to trigger cleanup
        from xaytune.trainer.callbacks import TrainState

        state = TrainState()
        components.trainer.callback_manager.fire("train_end", state)

        mock_cleanup.assert_called_once_with(ctx)

    @patch("xaytune.recipes.base.cleanup_distributed")
    @patch("xaytune.recipes.base.init_distributed")
    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.load_model")
    def test_no_cleanup_registered_for_single_gpu(
        self, mock_load_model, mock_load_dataset, mock_init, mock_cleanup
    ):
        """Single GPU: cleanup_distributed should NOT be registered."""
        mock_init.return_value = DistributedContext()
        mock_load_model.return_value = _mock_model_result()
        mock_load_dataset.return_value = _mock_dataset()

        config = _make_config()
        components = setup_training(config)

        from xaytune.trainer.callbacks import TrainState

        state = TrainState()
        components.trainer.callback_manager.fire("train_end", state)

        mock_cleanup.assert_not_called()

    @patch("xaytune.recipes.base.wrap_model_distributed")
    @patch("xaytune.recipes.base.init_distributed")
    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.load_model")
    def test_model_moved_to_device(self, mock_load_model, mock_load_dataset, mock_init, mock_wrap):
        """Model should be moved to the distributed context device."""
        ctx = DistributedContext()
        mock_init.return_value = ctx
        mr = _mock_model_result()
        mock_load_model.return_value = mr
        mock_load_dataset.return_value = _mock_dataset()

        config = _make_config()
        setup_training(config)

        mr.model.to.assert_called_once_with(ctx.device)

    @patch("xaytune.recipes.base.wrap_model_distributed")
    @patch("xaytune.recipes.base.init_distributed")
    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.load_model")
    def test_wrap_called_with_correct_args(
        self, mock_load_model, mock_load_dataset, mock_init, mock_wrap
    ):
        """wrap_model_distributed should receive fsdp_config and deepspeed_config."""
        ctx = DistributedContext(rank=0, world_size=2, local_rank=0)
        mock_init.return_value = ctx
        mr = _mock_model_result()
        mock_load_model.return_value = mr
        mock_load_dataset.return_value = _mock_dataset()
        mock_wrap.return_value = MagicMock()

        config = _make_config()
        setup_training(config)

        call_kwargs = mock_wrap.call_args.kwargs
        assert call_kwargs["strategy"] == "fsdp"  # auto resolves to fsdp for world_size=2
        assert call_kwargs["ctx"] is ctx
        assert call_kwargs["fsdp_config"] is config.fsdp
        assert call_kwargs["deepspeed_config"] is config.deepspeed_config
        assert call_kwargs["mixed_precision"] == config.trainer.mixed_precision


class TestSetupTrainingCheckpointCallbacks:
    @patch("xaytune.recipes.base.register_checkpoint_callbacks")
    @patch("xaytune.recipes.base.load_model")
    @patch("xaytune.recipes.base.load_dataset")
    def test_checkpoint_callbacks_registered(self, mock_load_ds, mock_load_model, mock_register):
        mock_load_model.return_value = _mock_model_result()
        mock_load_ds.return_value = _mock_dataset()

        config = _make_config(checkpoint_every_n_steps=100, save_last=True)
        setup_training(config)

        mock_register.assert_called_once()
        call_kwargs = mock_register.call_args.kwargs
        assert call_kwargs["checkpoint_every_n_steps"] == 100
        assert call_kwargs["save_last"] is True
        assert call_kwargs["output_dir"] == config.output.dir
        assert call_kwargs["is_main_process"] is True

    @patch("xaytune.recipes.base.register_checkpoint_callbacks")
    @patch("xaytune.recipes.base.load_model")
    @patch("xaytune.recipes.base.load_dataset")
    def test_checkpoint_callbacks_use_default_config(
        self, mock_load_ds, mock_load_model, mock_register
    ):
        mock_load_model.return_value = _mock_model_result()
        mock_load_ds.return_value = _mock_dataset()

        config = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="fake.jsonl", format="alpaca"),
        )
        setup_training(config)

        call_kwargs = mock_register.call_args.kwargs
        assert call_kwargs["checkpoint_every_n_steps"] == 500  # default
        assert call_kwargs["save_last"] is True  # default


class TestSetupTrainingResume:
    @patch("xaytune.recipes.base.load_checkpoint")
    @patch("xaytune.recipes.base.load_model")
    @patch("xaytune.recipes.base.load_dataset")
    def test_resume_loads_model_weights(self, mock_load_ds, mock_load_model, mock_load_ckpt):
        mock_load_model.return_value = _mock_model_result()
        mock_load_ds.return_value = _mock_dataset()
        mock_load_ckpt.return_value = TrainState(global_step=50, epoch=1, step=10)

        config = _make_config()
        components = setup_training(config, resume_from="/ckpt/checkpoint-50")

        mock_load_ckpt.assert_called_once()
        assert components.resume_state is not None
        assert components.resume_state.global_step == 50
        assert components.resume_state.epoch == 1

    @patch("xaytune.recipes.base.load_model")
    @patch("xaytune.recipes.base.load_dataset")
    def test_no_resume_state_when_not_requested(self, mock_load_ds, mock_load_model):
        mock_load_model.return_value = _mock_model_result()
        mock_load_ds.return_value = _mock_dataset()

        config = _make_config()
        components = setup_training(config)

        assert components.resume_state is None


class TestSeedSetting:
    @patch("xaytune.recipes.base.seed_all")
    @patch("xaytune.recipes.base.load_model")
    @patch("xaytune.recipes.base.load_dataset")
    def test_seed_applied(self, mock_load_ds, mock_load_model, mock_seed_all):
        mock_load_model.return_value = _mock_model_result()
        mock_load_ds.return_value = _mock_dataset()

        config = _make_config()
        config.trainer.seed = 123
        setup_training(config)

        mock_seed_all.assert_called_once_with(123)


class TestActivationCheckpointing:
    @patch("xaytune.recipes.base.load_model")
    @patch("xaytune.recipes.base.load_dataset")
    def test_enabled_calls_gradient_checkpointing(self, mock_load_ds, mock_load_model):
        mock_load_model.return_value = _mock_model_result()
        mock_load_ds.return_value = _mock_dataset()

        config = _make_config(activation_checkpointing=True)
        components = setup_training(config)

        components.model.gradient_checkpointing_enable.assert_called_once_with(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    @patch("xaytune.recipes.base.load_model")
    @patch("xaytune.recipes.base.load_dataset")
    def test_disabled_skips_gradient_checkpointing(self, mock_load_ds, mock_load_model):
        mock_load_model.return_value = _mock_model_result()
        mock_load_ds.return_value = _mock_dataset()

        config = _make_config(activation_checkpointing=False)
        components = setup_training(config)

        components.model.gradient_checkpointing_enable.assert_not_called()


class TestMergeOnComplete:
    @patch("xaytune.recipes.base.load_model")
    @patch("xaytune.recipes.base.load_dataset")
    def test_merge_called_for_lora(self, mock_load_ds, mock_load_model):
        mr = _mock_model_result()
        merged_model = MagicMock()
        mr.model.merge_and_unload.return_value = merged_model
        mock_load_model.return_value = mr

        with patch("xaytune.recipes.base.apply_lora") as mock_apply_lora:
            lora_result = MagicMock()
            lora_result.model = mr.model
            lora_result.tokenizer = mr.tokenizer
            mock_apply_lora.return_value = lora_result
            mock_load_ds.return_value = _mock_dataset()

            config = TrainConfig(
                recipe="finetune",
                method="lora",
                model=ModelConfig(name="test-model"),
                data=DataConfig(path="fake.jsonl", format="alpaca"),
                trainer=TrainerConfig(
                    batch_size=2,
                    num_epochs=1,
                    max_steps=1,
                    save_last=False,
                ),
                output=OutputConfig(dir="/tmp/test-merge", merge_on_complete=True),
            )
            components = setup_training(config)

        state = TrainState()
        components.trainer.callback_manager.fire("train_end", state)

        mr.model.merge_and_unload.assert_called_once()
        merged_model.save_pretrained.assert_called_once_with("/tmp/test-merge/merged")
        mr.tokenizer.save_pretrained.assert_called_once_with("/tmp/test-merge/merged")

    @patch("xaytune.recipes.base.load_model")
    @patch("xaytune.recipes.base.load_dataset")
    def test_merge_not_called_for_full_finetune(self, mock_load_ds, mock_load_model):
        mr = _mock_model_result()
        mock_load_model.return_value = mr
        mock_load_ds.return_value = _mock_dataset()

        config = TrainConfig(
            recipe="finetune",
            method="full",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="fake.jsonl", format="alpaca"),
            trainer=TrainerConfig(
                batch_size=2,
                num_epochs=1,
                max_steps=1,
                save_last=False,
            ),
            output=OutputConfig(dir="/tmp/test-merge", merge_on_complete=True),
        )
        components = setup_training(config)

        state = TrainState()
        components.trainer.callback_manager.fire("train_end", state)

        mr.model.merge_and_unload.assert_not_called()

    @patch("xaytune.recipes.base.load_model")
    @patch("xaytune.recipes.base.load_dataset")
    def test_merge_not_called_when_disabled(self, mock_load_ds, mock_load_model):
        mr = _mock_model_result()
        mock_load_model.return_value = mr
        mock_load_ds.return_value = _mock_dataset()

        with patch("xaytune.recipes.base.apply_lora") as mock_apply_lora:
            lora_result = MagicMock()
            lora_result.model = mr.model
            lora_result.tokenizer = mr.tokenizer
            mock_apply_lora.return_value = lora_result

            config = TrainConfig(
                recipe="finetune",
                method="lora",
                model=ModelConfig(name="test-model"),
                data=DataConfig(path="fake.jsonl", format="alpaca"),
                trainer=TrainerConfig(
                    batch_size=2,
                    num_epochs=1,
                    max_steps=1,
                    save_last=False,
                ),
                output=OutputConfig(dir="/tmp/test-merge", merge_on_complete=False),
            )
            components = setup_training(config)

        state = TrainState()
        components.trainer.callback_manager.fire("train_end", state)

        mr.model.merge_and_unload.assert_not_called()


@patch("xaytune.recipes.base.validate_dataset_sample")
class TestDataPacking:
    @patch("xaytune.recipes.base.pack_sequences")
    @patch("xaytune.recipes.base.load_model")
    @patch("xaytune.recipes.base.load_dataset")
    def test_packing_called_when_enabled(
        self, mock_load_ds, mock_load_model, mock_pack, mock_validate
    ):
        mock_load_model.return_value = _mock_model_result()
        mock_load_ds.return_value = [
            {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
            {"input_ids": [4, 5], "labels": [4, 5]},
        ]
        mock_pack.return_value = [
            {"input_ids": [1, 2, 3, 4, 5, 0], "labels": [1, 2, 3, 4, 5, -100]},
        ]

        config = _make_config()
        config.data.packing = True
        config.data.max_seq_length = 2048
        setup_training(config)

        mock_pack.assert_called_once()
        call_kwargs = mock_pack.call_args.kwargs
        assert call_kwargs["max_seq_length"] == 2048

    @patch("xaytune.recipes.base.pack_sequences")
    @patch("xaytune.recipes.base.load_model")
    @patch("xaytune.recipes.base.load_dataset")
    def test_packing_skipped_when_disabled(
        self, mock_load_ds, mock_load_model, mock_pack, mock_validate
    ):
        mock_load_model.return_value = _mock_model_result()
        mock_load_ds.return_value = [
            {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
        ]

        config = _make_config()
        config.data.packing = False
        setup_training(config)

        mock_pack.assert_not_called()

    @patch("xaytune.recipes.base.pack_sequences")
    @patch("xaytune.recipes.base.load_model")
    @patch("xaytune.recipes.base.load_dataset")
    def test_packing_skipped_for_tensor_data(
        self, mock_load_ds, mock_load_model, mock_pack, mock_validate
    ):
        mock_load_model.return_value = _mock_model_result()
        mock_load_ds.return_value = _mock_dataset()

        config = _make_config()
        config.data.packing = True
        setup_training(config)

        mock_pack.assert_not_called()


class TestAsyncCheckpointWiring:
    @patch("xaytune.recipes.base.register_checkpoint_callbacks")
    @patch("xaytune.recipes.base.load_model")
    @patch("xaytune.recipes.base.load_dataset")
    def test_async_saver_created_when_enabled(self, mock_load_ds, mock_load_model, mock_register):
        mock_load_model.return_value = _mock_model_result()
        mock_load_ds.return_value = _mock_dataset()

        config = _make_config(async_checkpoint=True)
        setup_training(config)

        call_kwargs = mock_register.call_args.kwargs
        assert call_kwargs["async_saver"] is not None

    @patch("xaytune.recipes.base.register_checkpoint_callbacks")
    @patch("xaytune.recipes.base.load_model")
    @patch("xaytune.recipes.base.load_dataset")
    def test_no_async_saver_when_disabled(self, mock_load_ds, mock_load_model, mock_register):
        mock_load_model.return_value = _mock_model_result()
        mock_load_ds.return_value = _mock_dataset()

        config = _make_config(async_checkpoint=False)
        setup_training(config)

        call_kwargs = mock_register.call_args.kwargs
        assert call_kwargs["async_saver"] is None
