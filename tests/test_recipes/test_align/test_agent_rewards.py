import json

from xaytune.recipes.align.agent_rewards import (
    ParsedToolCall,
    agent_composite_reward,
    efficiency_reward,
    parse_tool_calls,
    task_completion_reward,
    tool_use_quality_reward,
)


def _make_tool_call(name: str, args: dict) -> str:
    obj = {"name": name, "arguments": args}
    return f"<tool_call>\n{json.dumps(obj)}\n</tool_call>"


def _make_tool_result(content: str) -> str:
    return f"<tool_result>\n{content}\n</tool_result>"


class TestParseToolCalls:
    def test_single_call(self):
        response = _make_tool_call("search", {"q": "cats"})
        calls = parse_tool_calls(response)
        assert len(calls) == 1
        assert calls[0].name == "search"
        assert calls[0].arguments == {"q": "cats"}

    def test_multiple_calls(self):
        response = (
            _make_tool_call("search", {"q": "cats"})
            + "\n"
            + _make_tool_call("fetch", {"url": "http://example.com"})
        )
        calls = parse_tool_calls(response)
        assert len(calls) == 2
        assert calls[0].name == "search"
        assert calls[1].name == "fetch"

    def test_no_calls(self):
        calls = parse_tool_calls("Just a plain response with no tools.")
        assert calls == []

    def test_malformed_json(self):
        response = "<tool_call>\nnot valid json\n</tool_call>"
        calls = parse_tool_calls(response)
        assert calls == []

    def test_custom_parser(self):
        def my_parser(text):
            return [ParsedToolCall(name="custom", arguments={"from": "parser"})]

        calls = parse_tool_calls("anything", parser=my_parser)
        assert len(calls) == 1
        assert calls[0].name == "custom"

    def test_mixed_valid_and_invalid(self):
        response = _make_tool_call("good", {"a": 1}) + "\n<tool_call>\nbad json\n</tool_call>"
        calls = parse_tool_calls(response)
        assert len(calls) == 1
        assert calls[0].name == "good"


class TestToolUseQualityReward:
    def test_correct_tools(self):
        response = _make_tool_call("search", {"q": "cats"})
        score = tool_use_quality_reward("Find cats", response, expected_tools=["search"])
        assert score == 1.0

    def test_wrong_tools(self):
        response = _make_tool_call("calculator", {"expr": "2+2"})
        score = tool_use_quality_reward("Find cats", response, expected_tools=["search"])
        assert score == 0.0

    def test_partial_tools(self):
        response = _make_tool_call("search", {"q": "cats"})
        score = tool_use_quality_reward(
            "Search and calculate",
            response,
            expected_tools=["search", "calculator"],
        )
        assert score == 0.5

    def test_no_calls_returns_zero(self):
        score = tool_use_quality_reward("Hi", "Hello!", expected_tools=["search"])
        assert score == 0.0

    def test_no_expectations_any_call_is_good(self):
        response = _make_tool_call("anything", {})
        score = tool_use_quality_reward("Hi", response)
        assert score == 1.0

    def test_required_args(self):
        response = _make_tool_call("search", {"q": "cats"})
        score = tool_use_quality_reward(
            "Search",
            response,
            expected_tools=["search"],
            required_args={"search": ["q"]},
        )
        assert score == 1.0

    def test_missing_required_args(self):
        response = _make_tool_call("search", {})
        score = tool_use_quality_reward(
            "Search",
            response,
            expected_tools=["search"],
            required_args={"search": ["q", "limit"]},
        )
        assert score < 1.0

    def test_returns_bounded(self):
        for response in ["", "hello", _make_tool_call("x", {})]:
            score = tool_use_quality_reward("p", response)
            assert 0.0 <= score <= 1.0


class TestTaskCompletionReward:
    def test_success_markers(self):
        score = task_completion_reward(
            "Do something",
            "I have completed the task. Done.",
            success_markers=["Done", "completed"],
        )
        assert score == 1.0

    def test_partial_success(self):
        score = task_completion_reward(
            "Do something",
            "Task is Done.",
            success_markers=["Done", "verified", "tested"],
        )
        assert abs(score - 1.0 / 3.0) < 0.01

    def test_failure_markers(self):
        score = task_completion_reward(
            "Do something",
            "I cannot do this. Error occurred.",
            failure_markers=["Error", "cannot"],
        )
        assert score == 0.0

    def test_failure_overrides_success(self):
        score = task_completion_reward(
            "Do it",
            "Done but Error occurred",
            success_markers=["Done"],
            failure_markers=["Error"],
        )
        assert score == 0.0

    def test_final_answer_after_tools(self):
        response = (
            _make_tool_call("search", {"q": "test"})
            + "\n"
            + _make_tool_result("results here")
            + "\nHere are the results I found."
        )
        score = task_completion_reward("Search", response)
        assert score == 1.0

    def test_no_final_answer(self):
        response = (
            _make_tool_call("search", {"q": "test"}) + "\n" + _make_tool_result("results here")
        )
        score = task_completion_reward("Search", response)
        assert score == 0.0

    def test_plain_response(self):
        score = task_completion_reward("Hi", "Hello there!")
        assert score == 1.0

    def test_empty_response(self):
        score = task_completion_reward("Hi", "")
        assert score == 0.0

    def test_returns_bounded(self):
        for response in ["", "hello", "Error", "Done"]:
            score = task_completion_reward("p", response)
            assert 0.0 <= score <= 1.0


class TestEfficiencyReward:
    def test_no_calls_with_response(self):
        score = efficiency_reward("Hi", "Hello!")
        assert score == 1.0

    def test_one_call(self):
        response = _make_tool_call("search", {"q": "test"})
        score = efficiency_reward("Search", response, max_steps=10)
        assert score == 0.9

    def test_max_calls(self):
        response = "\n".join(_make_tool_call(f"tool_{i}", {}) for i in range(10))
        score = efficiency_reward("Do it", response, max_steps=10)
        assert score == 0.0

    def test_over_max(self):
        response = "\n".join(_make_tool_call(f"tool_{i}", {}) for i in range(15))
        score = efficiency_reward("Do it", response, max_steps=10)
        assert score == 0.0

    def test_optimal_steps_exact(self):
        response = "\n".join(_make_tool_call(f"tool_{i}", {}) for i in range(3))
        score = efficiency_reward("Do it", response, max_steps=10, optimal_steps=3)
        assert score == 1.0

    def test_optimal_steps_off(self):
        response = "\n".join(_make_tool_call(f"tool_{i}", {}) for i in range(5))
        score = efficiency_reward("Do it", response, max_steps=10, optimal_steps=3)
        assert score == 0.8

    def test_empty_response(self):
        score = efficiency_reward("Hi", "")
        assert score == 0.0

    def test_returns_bounded(self):
        for n in range(12):
            response = (
                "\n".join(_make_tool_call(f"t{i}", {}) for i in range(n)) if n > 0 else "hello"
            )
            score = efficiency_reward("p", response, max_steps=10)
            assert 0.0 <= score <= 1.0


class TestAgentCompositeReward:
    def test_weighted_sum(self):
        response = (
            _make_tool_call("search", {"q": "test"})
            + "\n"
            + _make_tool_result("found it")
            + "\nHere is the answer. Done."
        )
        score = agent_composite_reward(
            "Search for test",
            response,
            quality_weight=1.0,
            completion_weight=0.0,
            efficiency_weight=0.0,
            expected_tools=["search"],
        )
        quality_only = tool_use_quality_reward(
            "Search for test", response, expected_tools=["search"]
        )
        assert abs(score - quality_only) < 1e-6

    def test_all_weights_equal(self):
        response = (
            _make_tool_call("search", {"q": "test"})
            + "\n"
            + _make_tool_result("result")
            + "\nDone."
        )
        score = agent_composite_reward(
            "Search",
            response,
            quality_weight=1.0,
            completion_weight=1.0,
            efficiency_weight=1.0,
            expected_tools=["search"],
            success_markers=["Done"],
            max_steps=10,
        )
        q = tool_use_quality_reward("Search", response, expected_tools=["search"])
        c = task_completion_reward("Search", response, success_markers=["Done"])
        e = efficiency_reward("Search", response, max_steps=10)
        expected = q + c + e
        assert abs(score - expected) < 1e-6

    def test_returns_bounded(self):
        score = agent_composite_reward("p", "hello")
        assert 0.0 <= score <= 1.0


class TestRewardRegistration:
    def test_all_registered(self):
        from xaytune.recipes.align.rewards import reward_registry

        for name in [
            "tool_use_quality",
            "task_completion",
            "efficiency",
            "agent_composite",
        ]:
            assert reward_registry.has(name), f"{name} not registered"


class TestIntegration:
    def test_score_completions_with_agent_reward(self):
        from xaytune.recipes.align.reward_scoring import score_completions

        response = (
            _make_tool_call("search", {"q": "cats"})
            + "\n"
            + _make_tool_result("found cats")
            + "\nHere are the cats."
        )
        scores = score_completions(
            prompts=["Find cats"],
            responses=[response],
            reward_name="agent_composite",
            reward_kwargs={
                "expected_tools": ["search"],
                "success_markers": ["cats"],
                "max_steps": 5,
            },
        )
        assert scores.shape == (1,)
        assert 0.0 <= scores[0].item() <= 3.0  # weights sum > 1 possible
