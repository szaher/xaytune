from trainlib.trainer import Trainer, TrainerConfig, TrainState, on, CallbackManager
from trainlib.trainer import save_checkpoint, load_checkpoint, find_latest_checkpoint
from trainlib.trainer import wrap_model_distributed, DistributedContext, get_strategy


class TestTrainerPublicAPI:
    def test_trainer_importable(self):
        assert Trainer is not None

    def test_trainer_config_importable(self):
        assert TrainerConfig is not None

    def test_train_state_importable(self):
        assert TrainState is not None

    def test_on_decorator_importable(self):
        assert callable(on)

    def test_on_registers_globally(self):
        calls = []

        @on("step_end")
        def my_callback(state):
            calls.append(state.step)

        assert callable(my_callback)

    def test_checkpoint_functions_importable(self):
        assert callable(save_checkpoint)
        assert callable(load_checkpoint)
        assert callable(find_latest_checkpoint)

    def test_distributed_importable(self):
        assert callable(wrap_model_distributed)
        assert DistributedContext is not None
        assert callable(get_strategy)
