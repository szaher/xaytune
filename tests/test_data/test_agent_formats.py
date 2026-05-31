from xaytune.data.agent_formats import (
    AgentMessage,
    format_function_calling,
    format_react,
    format_trajectory,
)


class TestAgentMessage:
    def test_create_trainable(self):
        msg = AgentMessage(role="assistant", content="Hello", trainable=True)
        assert msg.role == "assistant"
        assert msg.content == "Hello"
        assert msg.trainable is True

    def test_create_non_trainable(self):
        msg = AgentMessage(role="user", content="Hi", trainable=False)
        assert msg.role == "user"
        assert msg.trainable is False

    def test_all_roles(self):
        for role in ("system", "user", "assistant", "tool"):
            msg = AgentMessage(role=role, content="test", trainable=False)
            assert msg.role == role


class TestFunctionCallingFormat:
    def test_basic_conversation(self):
        sample = {
            "messages": [
                {"role": "user", "content": "What's 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
            ]
        }
        messages = format_function_calling(sample)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].trainable is False
        assert messages[1].role == "assistant"
        assert messages[1].trainable is True

    def test_with_tool_call(self):
        sample = {
            "messages": [
                {"role": "user", "content": "Weather in London?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 18}'},
                {"role": "assistant", "content": "It's 18°C in London."},
            ]
        }
        messages = format_function_calling(sample)
        assert len(messages) == 4
        assert messages[1].role == "assistant"
        assert messages[1].trainable is True
        assert "<tool_call>" in messages[1].content
        assert "get_weather" in messages[1].content
        assert messages[2].role == "tool"
        assert messages[2].trainable is False
        assert "<tool_result>" in messages[2].content

    def test_with_system_message(self):
        sample = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        }
        messages = format_function_calling(sample)
        assert len(messages) == 3
        assert messages[0].role == "system"
        assert messages[0].trainable is False

    def test_with_tools_array(self):
        sample = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search the web",
                        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                    },
                }
            ],
        }
        messages = format_function_calling(sample)
        assert messages[0].role == "system"
        assert "search" in messages[0].content
        assert messages[0].trainable is False

    def test_multiple_tool_calls(self):
        sample = {
            "messages": [
                {"role": "user", "content": "Weather in London and Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
                        },
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                        },
                    ],
                },
            ]
        }
        messages = format_function_calling(sample)
        assistant_msg = messages[1]
        assert assistant_msg.content.count("<tool_call>") == 2

    def test_registered(self):
        from xaytune.data.registry import format_registry

        assert format_registry.has("function_calling")


class TestReactFormat:
    def test_single_step(self):
        sample = {
            "task": "What is 2+2?",
            "steps": [{"thought": "Simple math.", "action": "finish", "action_input": "4"}],
        }
        messages = format_react(sample)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "What is 2+2?"
        assert messages[0].trainable is False
        assert messages[1].role == "assistant"
        assert messages[1].trainable is True
        assert "Thought: Simple math." in messages[1].content
        assert "Action: finish" in messages[1].content
        assert "Action Input: 4" in messages[1].content

    def test_multi_step_with_observation(self):
        sample = {
            "task": "Find population of France",
            "steps": [
                {
                    "thought": "I need to search.",
                    "action": "search",
                    "action_input": "France population",
                    "observation": "68 million",
                },
                {"thought": "I have the answer.", "action": "finish", "action_input": "68 million"},
            ],
        }
        messages = format_react(sample)
        assert len(messages) == 4
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[1].trainable is True
        assert messages[2].role == "tool"
        assert messages[2].trainable is False
        assert "Observation: 68 million" in messages[2].content
        assert messages[3].role == "assistant"
        assert messages[3].trainable is True

    def test_step_without_observation(self):
        sample = {
            "task": "Say hello",
            "steps": [{"thought": "Easy.", "action": "finish", "action_input": "Hello!"}],
        }
        messages = format_react(sample)
        assert len(messages) == 2

    def test_registered(self):
        from xaytune.data.registry import format_registry

        assert format_registry.has("react")


class TestTrajectoryFormat:
    def test_basic(self):
        sample = {
            "goal": "Say hello",
            "turns": [
                {"role": "assistant", "content": "Hello!"},
            ],
        }
        messages = format_trajectory(sample)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Say hello"
        assert messages[0].trainable is False
        assert messages[1].role == "assistant"
        assert messages[1].trainable is True

    def test_with_system(self):
        sample = {
            "system": "You are a coding assistant.",
            "goal": "Write hello world",
            "turns": [
                {"role": "assistant", "content": "Done."},
            ],
        }
        messages = format_trajectory(sample)
        assert len(messages) == 3
        assert messages[0].role == "system"
        assert messages[0].trainable is False

    def test_with_tool_calls(self):
        sample = {
            "goal": "Create a file",
            "turns": [
                {
                    "role": "assistant",
                    "content": "I'll create it.",
                    "tool_calls": [
                        {"name": "write_file", "arguments": {"path": "a.py", "content": "x=1"}}
                    ],
                },
                {"role": "tool", "content": "File written."},
                {"role": "assistant", "content": "Done."},
            ],
        }
        messages = format_trajectory(sample)
        assert len(messages) == 4
        assert messages[1].role == "assistant"
        assert messages[1].trainable is True
        assert "<tool_call>" in messages[1].content
        assert "write_file" in messages[1].content
        assert messages[2].role == "tool"
        assert messages[2].trainable is False
        assert "<tool_result>" in messages[2].content
        assert messages[3].role == "assistant"
        assert messages[3].trainable is True

    def test_multi_turn(self):
        sample = {
            "goal": "Do two things",
            "turns": [
                {
                    "role": "assistant",
                    "content": "Step 1.",
                    "tool_calls": [{"name": "tool_a", "arguments": {}}],
                },
                {"role": "tool", "content": "Result 1"},
                {
                    "role": "assistant",
                    "content": "Step 2.",
                    "tool_calls": [{"name": "tool_b", "arguments": {}}],
                },
                {"role": "tool", "content": "Result 2"},
                {"role": "assistant", "content": "All done."},
            ],
        }
        messages = format_trajectory(sample)
        assert len(messages) == 6
        trainable_count = sum(1 for m in messages if m.trainable)
        assert trainable_count == 3

    def test_registered(self):
        from xaytune.data.registry import format_registry

        assert format_registry.has("trajectory")
