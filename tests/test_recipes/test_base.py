from unittest.mock import MagicMock, patch

from trainlib.config.schema import DataConfig, ModelConfig, TrainConfig, TrainerConfig
from trainlib.recipes.base import TrainingComponents, setup_training


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

    def test_fields(self):
        fields = TrainingComponents._fields
        assert "model" in fields
        assert "tokenizer" in fields
        assert "train_dataloader" in fields
        assert "eval_dataloader" in fields
        assert "trainer" in fields


class TestSetupTraining:
    def _make_config(self, method="full", **trainer_kwargs):
        return TrainConfig(
            recipe="finetune",
            method=method,
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="data.jsonl", format="alpaca"),
            trainer=TrainerConfig(**trainer_kwargs),
        )

    @patch("trainlib.recipes.base.load_model")
    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.DataLoader")
    def test_full_finetune_setup(self, mock_dl_cls, mock_load_ds, mock_load_model):
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

    @patch("trainlib.recipes.base.apply_lora")
    @patch("trainlib.recipes.base.load_model")
    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.DataLoader")
    def test_lora_setup_applies_peft(
        self, mock_dl_cls, mock_load_ds, mock_load_model, mock_apply_lora
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

    @patch("trainlib.recipes.base.load_model")
    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.DataLoader")
    def test_qlora_uses_4bit_quantization(self, mock_dl_cls, mock_load_ds, mock_load_model):
        mock_model_result = MagicMock()
        mock_model_result.model = MagicMock()
        mock_model_result.tokenizer = MagicMock()
        mock_load_model.return_value = mock_model_result
        mock_load_ds.return_value = [{"input_ids": [1]}]
        mock_dl_cls.return_value = MagicMock()

        config = self._make_config(method="qlora")
        with patch("trainlib.recipes.base.apply_lora") as mock_apply_lora:
            mock_apply_lora.return_value = mock_model_result
            setup_training(config)

        mock_load_model.assert_called_once_with(
            "test-model",
            quantization="4bit",
            dtype="auto",
            trust_remote_code=False,
        )

    @patch("trainlib.recipes.base.load_model")
    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.DataLoader")
    def test_eval_split_creates_eval_dataloader(self, mock_dl_cls, mock_load_ds, mock_load_model):
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

    @patch("trainlib.recipes.base.load_model")
    @patch("trainlib.recipes.base.load_dataset")
    @patch("trainlib.recipes.base.DataLoader")
    def test_no_eval_split_no_eval_dataloader(self, mock_dl_cls, mock_load_ds, mock_load_model):
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
