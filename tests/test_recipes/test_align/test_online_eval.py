from unittest.mock import MagicMock, patch

import torch

from xaytune.config.schema import GenerationConfig
from xaytune.recipes.align.online_eval import OnlineEvalCallback
from xaytune.trainer.callbacks import CallbackManager, TrainState


def _make_callback(every_n_steps=2):
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    encoded = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
    tokenizer.return_value = encoded

    return OnlineEvalCallback(
        eval_prompts=["What is 2+2?", "Explain gravity."],
        tokenizer=tokenizer,
        generation_config=GenerationConfig(max_new_tokens=16, group_size=1),
        reward_name="default",
        every_n_steps=every_n_steps,
    )


class TestOnlineEvalCallback:
    def test_registers_on_step_end(self):
        cb = CallbackManager()
        callback = _make_callback()
        callback.register(cb)
        assert len(cb._callbacks["step_end"]) > 0

    def test_fires_at_interval(self):
        cb = CallbackManager()
        callback = _make_callback(every_n_steps=2)
        callback.register(cb)

        eval_results = {"online_eval/mean_reward": 0.65, "online_eval/std_reward": 0.1}

        with patch.object(callback, "evaluate", return_value=eval_results) as mock_eval:
            state1 = TrainState(global_step=1, metrics={})
            cb.fire("step_end", state1)
            assert "online_eval/mean_reward" not in state1.metrics
            mock_eval.assert_not_called()

            state2 = TrainState(global_step=2, metrics={})
            cb.fire("step_end", state2)
            assert state2.metrics["online_eval/mean_reward"] == 0.65
            mock_eval.assert_called_once()

    def test_produces_metrics(self):
        cb = CallbackManager()
        callback = _make_callback(every_n_steps=1)
        callback.register(cb)

        eval_results = {"online_eval/mean_reward": 0.7, "online_eval/std_reward": 0.15}

        with patch.object(callback, "evaluate", return_value=eval_results):
            state = TrainState(global_step=1, metrics={})
            cb.fire("step_end", state)

            assert "online_eval/mean_reward" in state.metrics
            assert "online_eval/std_reward" in state.metrics
            assert abs(state.metrics["online_eval/mean_reward"] - 0.7) < 0.01

    def test_skips_step_zero(self):
        cb = CallbackManager()
        callback = _make_callback(every_n_steps=1)
        callback.register(cb)

        with patch.object(callback, "evaluate") as mock_eval:
            state = TrainState(global_step=0, metrics={})
            cb.fire("step_end", state)
            assert "online_eval/mean_reward" not in state.metrics
            mock_eval.assert_not_called()

    def test_evaluate_returns_empty_for_no_prompts(self):
        tokenizer = MagicMock()
        callback = OnlineEvalCallback(
            eval_prompts=[],
            tokenizer=tokenizer,
            generation_config=GenerationConfig(),
        )
        state = TrainState(global_step=1, metrics={})
        result = callback.evaluate(state)
        assert result == {}
