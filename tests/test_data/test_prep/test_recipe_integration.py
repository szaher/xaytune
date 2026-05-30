from xaytune.config.schema import DataPrepStepConfig, TrainConfig


class TestDataPrepConfig:
    def test_train_config_accepts_data_prep(self):
        config = TrainConfig(
            recipe="finetune",
            model={"name": "test-model"},
            data={"path": "data.jsonl", "format": "alpaca"},
            data_prep=[
                {"filter": {"min_chars": 50}},
                {"deduplicate": {"method": "exact"}},
            ],
        )
        assert len(config.data_prep) == 2

    def test_train_config_default_no_data_prep(self):
        config = TrainConfig(
            recipe="finetune",
            model={"name": "test-model"},
            data={"path": "data.jsonl", "format": "alpaca"},
        )
        assert config.data_prep == []

    def test_data_prep_step_validation(self):
        step = DataPrepStepConfig(filter={"min_chars": 50})
        assert step.filter == {"min_chars": 50}
        assert step.deduplicate is None
        assert step.convert is None
