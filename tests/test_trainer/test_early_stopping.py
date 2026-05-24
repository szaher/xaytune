from __future__ import annotations

from xaytune.trainer.callbacks import CallbackManager, TrainState
from xaytune.trainer.early_stopping import register_early_stopping_callbacks


class TestEarlyStopping:
    def _setup(self, patience=3, metric="eval_loss", min_delta=0.0):
        cb = CallbackManager()
        register_early_stopping_callbacks(
            callback_manager=cb,
            patience=patience,
            metric=metric,
            min_delta=min_delta,
        )
        return cb

    def test_stops_after_patience_exhausted(self):
        cb = self._setup(patience=2)
        state = TrainState()

        state.metrics["eval_loss"] = 1.0
        cb.fire("eval_end", state)
        assert not state.should_stop

        state.metrics["eval_loss"] = 1.0
        cb.fire("eval_end", state)
        assert not state.should_stop

        state.metrics["eval_loss"] = 1.0
        cb.fire("eval_end", state)
        assert state.should_stop

    def test_resets_patience_on_improvement(self):
        cb = self._setup(patience=2)
        state = TrainState()

        state.metrics["eval_loss"] = 1.0
        cb.fire("eval_end", state)

        state.metrics["eval_loss"] = 1.1
        cb.fire("eval_end", state)
        assert not state.should_stop

        state.metrics["eval_loss"] = 0.5
        cb.fire("eval_end", state)
        assert not state.should_stop

        state.metrics["eval_loss"] = 0.6
        cb.fire("eval_end", state)
        assert not state.should_stop

        state.metrics["eval_loss"] = 0.6
        cb.fire("eval_end", state)
        assert state.should_stop

    def test_respects_min_delta(self):
        cb = self._setup(patience=2, min_delta=0.1)
        state = TrainState()

        state.metrics["eval_loss"] = 1.0
        cb.fire("eval_end", state)

        state.metrics["eval_loss"] = 0.95
        cb.fire("eval_end", state)
        assert not state.should_stop

        state.metrics["eval_loss"] = 0.92
        cb.fire("eval_end", state)
        assert state.should_stop

    def test_min_delta_improvement_resets(self):
        cb = self._setup(patience=2, min_delta=0.1)
        state = TrainState()

        state.metrics["eval_loss"] = 1.0
        cb.fire("eval_end", state)

        state.metrics["eval_loss"] = 0.95
        cb.fire("eval_end", state)

        state.metrics["eval_loss"] = 0.8
        cb.fire("eval_end", state)
        assert not state.should_stop

    def test_max_mode_for_accuracy(self):
        cb = self._setup(patience=2, metric="eval_accuracy")
        state = TrainState()

        state.metrics["eval_accuracy"] = 0.8
        cb.fire("eval_end", state)

        state.metrics["eval_accuracy"] = 0.75
        cb.fire("eval_end", state)

        state.metrics["eval_accuracy"] = 0.78
        cb.fire("eval_end", state)
        assert state.should_stop

    def test_max_mode_improvement_resets(self):
        cb = self._setup(patience=2, metric="eval_accuracy")
        state = TrainState()

        state.metrics["eval_accuracy"] = 0.8
        cb.fire("eval_end", state)

        state.metrics["eval_accuracy"] = 0.75
        cb.fire("eval_end", state)

        state.metrics["eval_accuracy"] = 0.9
        cb.fire("eval_end", state)
        assert not state.should_stop

    def test_missing_metric_no_crash(self):
        cb = self._setup(patience=1)
        state = TrainState()

        cb.fire("eval_end", state)
        cb.fire("eval_end", state)
        cb.fire("eval_end", state)
        assert not state.should_stop

    def test_perplexity_uses_min_mode(self):
        cb = self._setup(patience=1, metric="eval_perplexity")
        state = TrainState()

        state.metrics["eval_perplexity"] = 10.0
        cb.fire("eval_end", state)

        state.metrics["eval_perplexity"] = 10.5
        cb.fire("eval_end", state)
        assert state.should_stop
