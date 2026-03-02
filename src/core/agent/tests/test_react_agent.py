"""
Tests for the generic ReAct Agent
"""

import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock

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
        response_obj.content = """## Reasoning
I know the answer.

## Response
The answer is 42.
"""
        self.agent.llm_manager.generate_response.return_value = response_obj

        result = await self.agent.process_input("What is the answer?")

        self.assertEqual(result, "The answer is 42.")
        self.assertEqual(self.agent.state, AgentState.READY)

    async def test_react_loop_with_tool(self):
        """Test thought -> tool -> response -> final answer."""

        # Step 1: Tool call
        response1 = MagicMock()
        response1.content = """## Reasoning
I need to calculate.

## Response
{
  "name": "calculator", 
  "arguments": {"expr": "2+2"}
}
"""

        # Step 2: Final answer (after tool execution)
        response2 = MagicMock()
        response2.content = """## Reasoning
The result is 4.

## Response
4
"""

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
        response.content = """## Reasoning
Thinking...

## Response
{
  "name": "wait", 
  "arguments": "forever"
}
"""
        self.agent.llm_manager.generate_response.return_value = response

        result = await self.agent.process_input("Wait")

        self.assertIn("Reached maximum steps", result)
        self.assertEqual(self.agent.llm_manager.generate_response.call_count, 2)

    async def test_parallel_tool_calls(self):
        """Test parallel tool execution."""
        response1 = MagicMock()
        response1.content = """## Reasoning
Two calcs

## Response
[
  {"name": "calc", "arguments": {"x": 1}},
  {"name": "calc", "arguments": {"x": 2}}
]
"""
        response2 = MagicMock()
        response2.content = "## Response\nDone"

        self.agent.llm_manager.generate_response.side_effect = [response1, response2]
        self.agent.execute_tool = AsyncMock(side_effect=["Res1", "Res2"])

        await self.agent.process_input("Do two things")

        self.assertEqual(self.agent.execute_tool.call_count, 2)
        # Verify both calls were made
        self.agent.execute_tool.assert_any_call("calc", {"x": 1})
        self.agent.execute_tool.assert_any_call("calc", {"x": 2})


if __name__ == "__main__":
    unittest.main()
