from __future__ import annotations

from unittest.mock import MagicMock

from xaytune.trainer.callbacks import CallbackManager, TrainState
from xaytune.trainer.eval_callback import register_eval_callbacks


def _make_eval_dataloader(n=3):
    mock_output = MagicMock()
    mock_output.loss = MagicMock()
    mock_output.loss.item.return_value = 0.5

    model = MagicMock()
    model.training = True
    model.return_value = mock_output

    batches = [{"input_ids": MagicMock(), "labels": MagicMock()} for _ in range(n)]
    return model, batches


class TestPeriodicEval:
    def test_eval_fires_at_interval(self):
        cb = CallbackManager()
        model, dl = _make_eval_dataloader()

        register_eval_callbacks(
            callback_manager=cb,
            model=model,
            eval_dataloader=dl,
            every_n_steps=3,
            metrics=["loss"],
        )

        eval_count = []

        @cb.on("eval_start")
        def _on_start(state):
            eval_count.append(state.global_step)

        for step in range(1, 7):
            state = TrainState(global_step=step)
            cb.fire("step_end", state)

        assert eval_count == [3, 6]

    def test_no_eval_at_step_zero(self):
        cb = CallbackManager()
        model, dl = _make_eval_dataloader()

        register_eval_callbacks(
            callback_manager=cb,
            model=model,
            eval_dataloader=dl,
            every_n_steps=1,
            metrics=["loss"],
        )

        eval_count = []

        @cb.on("eval_start")
        def _on_start(state):
            eval_count.append(state.global_step)

        cb.fire("step_end", TrainState(global_step=0))
        assert eval_count == []

    def test_disabled_when_interval_zero(self):
        cb = CallbackManager()
        model, dl = _make_eval_dataloader()

        register_eval_callbacks(
            callback_manager=cb,
            model=model,
            eval_dataloader=dl,
            every_n_steps=0,
            metrics=["loss"],
        )

        eval_count = []

        @cb.on("eval_start")
        def _on_start(state):
            eval_count.append(1)

        cb.fire("step_end", TrainState(global_step=5))
        assert eval_count == []

    def test_skips_when_not_main_process(self):
        cb = CallbackManager()
        model, dl = _make_eval_dataloader()

        register_eval_callbacks(
            callback_manager=cb,
            model=model,
            eval_dataloader=dl,
            every_n_steps=1,
            metrics=["loss"],
            is_main_process=False,
        )

        eval_count = []

        @cb.on("eval_start")
        def _on_start(state):
            eval_count.append(1)

        cb.fire("step_end", TrainState(global_step=1))
        assert eval_count == []

    def test_metrics_stored_with_eval_prefix(self):
        cb = CallbackManager()
        model, dl = _make_eval_dataloader()

        register_eval_callbacks(
            callback_manager=cb,
            model=model,
            eval_dataloader=dl,
            every_n_steps=1,
            metrics=["loss", "perplexity"],
        )

        state = TrainState(global_step=1)
        cb.fire("step_end", state)

        assert "eval_loss" in state.metrics
        assert "eval_perplexity" in state.metrics
        assert state.metrics["eval_loss"] == 0.5
        assert state.metrics["eval_perplexity"] > 0

    def test_eval_start_and_end_events_fire(self):
        cb = CallbackManager()
        model, dl = _make_eval_dataloader()

        register_eval_callbacks(
            callback_manager=cb,
            model=model,
            eval_dataloader=dl,
            every_n_steps=1,
            metrics=["loss"],
        )

        events = []

        @cb.on("eval_start")
        def _on_start(state):
            events.append("eval_start")

        @cb.on("eval_end")
        def _on_end(state):
            events.append("eval_end")

        cb.fire("step_end", TrainState(global_step=1))
        assert events == ["eval_start", "eval_end"]

    def test_model_switched_to_eval_and_back(self):
        cb = CallbackManager()
        model, dl = _make_eval_dataloader()

        register_eval_callbacks(
            callback_manager=cb,
            model=model,
            eval_dataloader=dl,
            every_n_steps=1,
            metrics=["loss"],
        )

        cb.fire("step_end", TrainState(global_step=1))

        model.eval.assert_called_once()
        model.train.assert_called_once()

    def test_model_not_restored_if_not_training(self):
        cb = CallbackManager()
        model, dl = _make_eval_dataloader()
        model.training = False

        register_eval_callbacks(
            callback_manager=cb,
            model=model,
            eval_dataloader=dl,
            every_n_steps=1,
            metrics=["loss"],
        )

        cb.fire("step_end", TrainState(global_step=1))

        model.eval.assert_called_once()
        model.train.assert_not_called()

    def test_only_requested_metrics_stored(self):
        cb = CallbackManager()
        model, dl = _make_eval_dataloader()

        register_eval_callbacks(
            callback_manager=cb,
            model=model,
            eval_dataloader=dl,
            every_n_steps=1,
            metrics=["loss"],
        )

        state = TrainState(global_step=1)
        cb.fire("step_end", state)

        assert "eval_loss" in state.metrics
        assert "eval_perplexity" not in state.metrics

    def test_eval_metrics_from_multiple_batches(self):
        cb = CallbackManager()

        losses = [0.2, 0.4, 0.6]
        call_idx = {"i": 0}

        mock_output = MagicMock()

        def _side_effect(**kwargs):
            out = MagicMock()
            loss_mock = MagicMock()
            loss_mock.item.return_value = losses[call_idx["i"]]
            out.loss = loss_mock
            call_idx["i"] += 1
            return out

        model = MagicMock()
        model.training = True
        model.side_effect = _side_effect
        model.return_value = mock_output

        # Override __call__ to use side_effect
        model.__call__ = _side_effect

        dl = [
            {"input_ids": MagicMock(), "labels": MagicMock()},
            {"input_ids": MagicMock(), "labels": MagicMock()},
            {"input_ids": MagicMock(), "labels": MagicMock()},
        ]

        register_eval_callbacks(
            callback_manager=cb,
            model=model,
            eval_dataloader=dl,
            every_n_steps=1,
            metrics=["loss"],
        )

        state = TrainState(global_step=1)
        cb.fire("step_end", state)

        expected_loss = sum(losses) / len(losses)
        assert abs(state.metrics["eval_loss"] - expected_loss) < 1e-6
