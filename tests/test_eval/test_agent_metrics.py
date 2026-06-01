import json

from xaytune.eval.agent_metrics import (
    compute_error_recovery_rate,
    compute_step_efficiency,
    compute_task_success_rate,
    compute_tool_use_accuracy,
    evaluate_agent,
)


def _tc(name, args):
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': args})}\n</tool_call>"


def _tr(content):
    return f"<tool_result>\n{content}\n</tool_result>"


class TestToolUseAccuracy:
    def test_all_correct(self):
        responses = [
            {"prompt": "Search", "response": _tc("search", {"q": "cats"})},
            {"prompt": "Search", "response": _tc("search", {"q": "dogs"})},
        ]
        score = compute_tool_use_accuracy(responses, expected_tools=["search"])
        assert score == 1.0

    def test_all_wrong(self):
        responses = [
            {"prompt": "Search", "response": _tc("calculator", {"expr": "2+2"})},
        ]
        score = compute_tool_use_accuracy(responses, expected_tools=["search"])
        assert score == 0.0

    def test_mixed(self):
        responses = [
            {"prompt": "Search", "response": _tc("search", {"q": "cats"})},
            {"prompt": "Search", "response": _tc("calculator", {"expr": "2+2"})},
        ]
        score = compute_tool_use_accuracy(responses, expected_tools=["search"])
        assert score == 0.5

    def test_empty(self):
        assert compute_tool_use_accuracy([]) == 0.0

    def test_no_expectations(self):
        responses = [
            {"prompt": "Hi", "response": _tc("anything", {})},
        ]
        score = compute_tool_use_accuracy(responses)
        assert score == 1.0


class TestTaskSuccessRate:
    def test_all_success(self):
        responses = [
            {"prompt": "Do it", "response": _tc("x", {}) + _tr("ok") + "\nDone."},
            {"prompt": "Do it", "response": _tc("y", {}) + _tr("ok") + "\nFinished."},
        ]
        score = compute_task_success_rate(responses)
        assert score == 1.0

    def test_all_failure(self):
        responses = [
            {"prompt": "Do it", "response": _tc("x", {}) + _tr("ok")},
            {"prompt": "Do it", "response": _tc("y", {}) + _tr("ok")},
        ]
        score = compute_task_success_rate(responses)
        assert score == 0.0

    def test_with_markers(self):
        responses = [
            {"prompt": "Do it", "response": "Done successfully."},
            {"prompt": "Do it", "response": "Error occurred."},
        ]
        score = compute_task_success_rate(
            responses,
            success_markers=["Done"],
            failure_markers=["Error"],
        )
        assert score == 0.5

    def test_empty(self):
        assert compute_task_success_rate([]) == 0.0


class TestStepEfficiency:
    def test_no_calls(self):
        responses = [{"prompt": "Hi", "response": "Hello!"}]
        score = compute_step_efficiency(responses, max_steps=5)
        assert score == 1.0

    def test_one_call(self):
        responses = [{"prompt": "Search", "response": _tc("search", {"q": "x"})}]
        score = compute_step_efficiency(responses, max_steps=10)
        assert score == 0.9

    def test_many_calls(self):
        calls = "\n".join(_tc(f"t{i}", {}) for i in range(10))
        responses = [{"prompt": "Do it", "response": calls}]
        score = compute_step_efficiency(responses, max_steps=10)
        assert score == 0.0

    def test_empty(self):
        assert compute_step_efficiency([]) == 0.0


class TestErrorRecoveryRate:
    def test_no_errors(self):
        responses = [{"prompt": "Hi", "response": "Hello!"}]
        score = compute_error_recovery_rate(responses)
        assert score == 1.0

    def test_error_with_recovery(self):
        response = (
            _tc("search", {"q": "test"})
            + "\n"
            + _tr("Error: connection failed")
            + "\n"
            + _tc("search", {"q": "test"})
            + "\n"
            + _tr("Found results")
            + "\nHere are the results."
        )
        responses = [{"prompt": "Search", "response": response}]
        score = compute_error_recovery_rate(responses)
        assert score == 1.0

    def test_error_no_recovery(self):
        response = _tc("search", {"q": "test"}) + "\n" + _tr("Error: failed")
        responses = [{"prompt": "Search", "response": response}]
        score = compute_error_recovery_rate(responses)
        assert score == 0.0

    def test_empty(self):
        assert compute_error_recovery_rate([]) == 0.0


class TestEvaluateAgent:
    def test_returns_all_metrics(self):
        responses = [
            {
                "prompt": "Search for cats",
                "response": (
                    _tc("search", {"q": "cats"})
                    + "\n"
                    + _tr("Found cats")
                    + "\nHere are the cats. Done."
                ),
            },
        ]
        results = evaluate_agent(
            responses,
            expected_tools=["search"],
            success_markers=["Done"],
            max_steps=5,
        )
        assert "tool_use_accuracy" in results
        assert "task_success_rate" in results
        assert "step_efficiency" in results
        assert "error_recovery_rate" in results
        for v in results.values():
            assert 0.0 <= v <= 1.0

    def test_good_vs_bad(self):
        good = [
            {
                "prompt": "Search",
                "response": _tc("search", {"q": "x"}) + _tr("ok") + "\nDone.",
            }
        ]
        bad = [
            {
                "prompt": "Search",
                "response": _tc("wrong", {}) + _tr("Error: failed"),
            }
        ]
        good_results = evaluate_agent(good, expected_tools=["search"], success_markers=["Done"])
        bad_results = evaluate_agent(bad, expected_tools=["search"], success_markers=["Done"])
        assert good_results["tool_use_accuracy"] > bad_results["tool_use_accuracy"]
        assert good_results["task_success_rate"] > bad_results["task_success_rate"]


class TestRegistration:
    def test_metrics_registered(self):
        from xaytune.eval.metrics import metric_registry

        metric_names = [
            "tool_use_accuracy",
            "task_success_rate",
            "step_efficiency",
            "error_recovery_rate",
        ]
        for name in metric_names:
            assert metric_registry.has(name), f"{name} not registered"
