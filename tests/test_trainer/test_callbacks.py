import pytest
from trainlib.trainer.callbacks import CallbackManager, TrainState


class TestTrainState:
    def test_initial_state(self):
        state = TrainState()
        assert state.step == 0
        assert state.epoch == 0
        assert state.global_step == 0
        assert state.metrics == {}
        assert state.should_stop is False

    def test_stop_training(self):
        state = TrainState()
        state.stop_training()
        assert state.should_stop is True

    def test_update_metrics(self):
        state = TrainState()
        state.metrics["loss"] = 0.5
        assert state.metrics["loss"] == 0.5


class TestCallbackManager:
    def test_register_and_fire(self):
        manager = CallbackManager()
        calls = []

        @manager.on("step_end")
        def my_callback(state):
            calls.append(state.step)

        state = TrainState(step=5)
        manager.fire("step_end", state)
        assert calls == [5]

    def test_multiple_callbacks_same_event(self):
        manager = CallbackManager()
        calls = []

        @manager.on("step_end")
        def cb1(state):
            calls.append("cb1")

        @manager.on("step_end")
        def cb2(state):
            calls.append("cb2")

        manager.fire("step_end", TrainState())
        assert calls == ["cb1", "cb2"]

    def test_different_events(self):
        manager = CallbackManager()
        calls = []

        @manager.on("train_start")
        def on_start(state):
            calls.append("start")

        @manager.on("train_end")
        def on_end(state):
            calls.append("end")

        manager.fire("train_start", TrainState())
        manager.fire("train_end", TrainState())
        assert calls == ["start", "end"]

    def test_fire_unknown_event_is_noop(self):
        manager = CallbackManager()
        manager.fire("nonexistent", TrainState())  # should not raise

    def test_all_event_types(self):
        manager = CallbackManager()
        events = [
            "train_start", "train_end",
            "epoch_start", "epoch_end",
            "step_start", "step_end",
            "eval_start", "eval_end",
            "checkpoint_saved", "error",
        ]
        fired = []
        for event in events:
            @manager.on(event)
            def cb(state, e=event):
                fired.append(e)

        for event in events:
            manager.fire(event, TrainState())
        assert fired == events

    def test_on_returns_original_function(self):
        manager = CallbackManager()

        @manager.on("step_end")
        def my_func(state):
            return 42

        assert my_func(TrainState()) == 42

    def test_callback_can_stop_training(self):
        manager = CallbackManager()

        @manager.on("step_end")
        def early_stop(state):
            if state.step >= 10:
                state.stop_training()

        state = TrainState(step=10)
        manager.fire("step_end", state)
        assert state.should_stop is True
