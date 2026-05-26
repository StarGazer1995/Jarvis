"""
Tests for the generic ReAct Agent
"""

import unittest
from unittest.mock import AsyncMock, MagicMock

from src.core.agent.react import ReActAgent
from src.core.agent.types import AgentState


class TestReActAgent(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.agent = ReActAgent(config={"enable_llm": True})

        # Mock LLM Manager
        self.agent.llm_manager = AsyncMock()
        self.agent.llm_manager.generate_response = AsyncMock()
        self.agent.llm_manager.initialize_default_client = AsyncMock(return_value=True)

        # Mock execute_tool
        self.agent.execute_tool = AsyncMock(return_value="Tool result")

        # Initialize
        await self.agent.initialize()

    async def test_react_loop_final_answer(self):
        """Test simple thought -> final answer loop."""
        response_obj = MagicMock()
        response_obj.content = (
            '{"thought": "I know the answer.", "type": "answer", '
            '"content": "The answer is 42."}'
        )
        self.agent.llm_manager.generate_response.return_value = response_obj

        result = await self.agent.process_input("What is the answer?")

        self.assertEqual(result, "The answer is 42.")
        self.assertEqual(self.agent.state, AgentState.READY)

    async def test_react_loop_with_tool(self):
        """Test thought -> tool -> response -> final answer."""

        # Step 1: Tool call
        response1 = MagicMock()
        response1.content = (
            '{"thought": "I need to calculate.", "type": "tool_call", '
            '"content": {"name": "calculator", "arguments": {"expr": "2+2"}}}'
        )

        # Step 2: Final answer (after tool execution)
        response2 = MagicMock()
        response2.content = (
            '{"thought": "The result is 4.", "type": "answer", "content": "4"}'
        )

        # Configure side effects for LLM calls
        # Note: The loop calls generate_response.
        # Call 1: Gets tool call
        # Agent executes tool -> appends <tool_response>
        # Call 2: Gets final answer
        self.agent.llm_manager.generate_response.side_effect = [response1, response2]

        # Configure tool execution
        self.agent.execute_tool = AsyncMock(return_value="4")

        result = await self.agent.process_input("Calculate 2+2")

        self.assertEqual(result, "4")
        self.agent.execute_tool.assert_called_with("calculator", {"expr": "2+2"})
        self.assertEqual(self.agent.llm_manager.generate_response.call_count, 2)

    async def test_max_steps(self):
        """Test that the loop respects max steps."""
        self.agent.max_steps = 2

        response = MagicMock()
        response.content = (
            '{"thought": "Thinking...", "type": "tool_call", '
            '"content": {"name": "wait", "arguments": "forever"}}'
        )
        self.agent.llm_manager.generate_response.return_value = response

        result = await self.agent.process_input("Wait")

        self.assertIn("Reached maximum steps", result)
        self.assertEqual(self.agent.llm_manager.generate_response.call_count, 2)

    async def test_parallel_tool_calls(self):
        """Test parallel tool execution."""
        response1 = MagicMock()
        response1.content = (
            '{"thought": "Two calcs", "type": "tool_calls", "content": ['
            '{"name": "calc", "arguments": {"x": 1}}, '
            '{"name": "calc", "arguments": {"x": 2}}'
            "]}"
        )
        response2 = MagicMock()
        response2.content = '{"thought": "Done", "type": "answer", "content": "Done"}'

        self.agent.llm_manager.generate_response.side_effect = [response1, response2]
        self.agent.execute_tool = AsyncMock(side_effect=["Res1", "Res2"])

        await self.agent.process_input("Do two things")

        self.assertEqual(self.agent.execute_tool.call_count, 2)
        # Verify both calls were made
        self.agent.execute_tool.assert_any_call("calc", {"x": 1})
        self.agent.execute_tool.assert_any_call("calc", {"x": 2})

    async def test_invalid_json_retries_without_legacy_fallback(self):
        """Test invalid JSON triggers retry instead of legacy markdown fallback."""
        bad_response = MagicMock()
        bad_response.content = "## Response\nLegacy fallback should not pass"
        good_response = MagicMock()
        good_response.content = (
            '{"thought": "Recovered", "type": "answer", "content": "Done"}'
        )

        self.agent.llm_manager.generate_response.side_effect = [
            bad_response,
            good_response,
        ]

        result = await self.agent.process_input("Recover from invalid JSON")

        self.assertEqual(result, "Done")
        self.assertEqual(self.agent.llm_manager.generate_response.call_count, 2)

    async def test_error_type_without_code_uses_plain_error_message(self):
        """Test error payload without code returns plain error prefix."""
        response_obj = MagicMock()
        response_obj.content = (
            '{"thought": "Bad request", "type": "error", '
            '"content": {"message": "Missing code"}}'
        )
        self.agent.llm_manager.generate_response.return_value = response_obj

        result = await self.agent.process_input("Trigger error")

        self.assertEqual(result, "Error: Missing code")

    async def test_error_type_with_non_object_content_returns_stringified_error(self):
        """Test non-object error content uses generic error formatting."""
        response_obj = MagicMock()
        response_obj.content = (
            '{"thought": "Bad request", "type": "error", "content": "Plain failure"}'
        )
        self.agent.llm_manager.generate_response.return_value = response_obj

        result = await self.agent.process_input("Trigger error")

        self.assertEqual(result, "Error: Plain failure")

    async def test_async_final_answer_validator_is_awaited(self):
        """Test async final-answer validation hooks are awaited by the loop."""
        bad_response = MagicMock()
        bad_response.content = (
            '{"thought": "First try", "type": "answer", "content": "wrong"}'
        )
        good_response = MagicMock()
        good_response.content = (
            '{"thought": "Second try", "type": "answer", "content": "fixed"}'
        )
        self.agent.llm_manager.generate_response.side_effect = [
            bad_response,
            good_response,
        ]

        async def validate(content, rendered_answer, steps):
            if rendered_answer == "wrong":
                return "Need a corrected answer."
            return None

        self.agent._validate_final_answer = validate

        result = await self.agent.process_input("Check async validation")

        self.assertEqual(result, "fixed")
        self.assertEqual(self.agent.llm_manager.generate_response.call_count, 2)


if __name__ == "__main__":
    unittest.main()
