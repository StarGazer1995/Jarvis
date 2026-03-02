import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import json

# Import from the new simple engine module
from src.core.ark.simple_engine import SimpleARKEngine, TaskStatus
from src.core.agent.types import AgentState


class TestSimpleARKEngine(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = SimpleARKEngine(config={"enable_llm": True})
        # Mock MCP Client
        self.engine.mcp_client = AsyncMock()
        self.engine.mcp_client.sessions = {}
        self.engine.mcp_client.execute_tool = AsyncMock(
            return_value="Tool executed successfully"
        )

        # Mock LLM Manager
        self.engine.llm_manager = AsyncMock()
        self.engine.llm_manager.generate_response = AsyncMock()

        # Manually set state to READY to skip full initialization in tests
        self.engine.state = AgentState.READY

    async def test_react_loop_add_task(self):
        """Test the ReAct loop handling a task addition."""

        # Simulate LLM responses for a 2-step process:
        # 1. Thought + Action (add task)
        # 2. Thought + Final Answer

        response1 = MagicMock()
        response1.content = """Thought: The user wants to add a task. I should use manage_tasks.
Action: manage_tasks
Action Input: {"action": "add", "description": "Buy groceries"}
"""

        response2 = MagicMock()
        response2.content = """Thought: Task added successfully.
Final Answer: I have added 'Buy groceries' to your todo list.
"""

        self.engine.llm_manager.generate_response.side_effect = [response1, response2]

        user_input = "Remind me to buy groceries"
        final_response = await self.engine.process_input(user_input)

        # Verify LLM was called twice
        self.assertEqual(self.engine.llm_manager.generate_response.call_count, 2)

        # Verify Todo List was updated
        self.assertEqual(len(self.engine.todo_list), 1)
        self.assertEqual(self.engine.todo_list[0].description, "Buy groceries")
        self.assertEqual(self.engine.todo_list[0].status, TaskStatus.PENDING)

        # Verify Final Response
        self.assertEqual(
            final_response, "I have added 'Buy groceries' to your todo list."
        )

    async def test_react_loop_direct_answer(self):
        """Test the ReAct loop providing a direct answer."""

        response = MagicMock()
        response.content = """Thought: This is a simple greeting.
Final Answer: Hello! How can I help you today?
"""

        self.engine.llm_manager.generate_response.return_value = response

        user_input = "Hi"
        final_response = await self.engine.process_input(user_input)

        # Verify LLM called once
        self.assertEqual(self.engine.llm_manager.generate_response.call_count, 1)

        # Verify Response
        self.assertEqual(final_response, "Hello! How can I help you today?")

        # Verify no tasks added
        self.assertEqual(len(self.engine.todo_list), 0)

    async def test_react_loop_tool_execution(self):
        """Test external tool execution."""

        # Mock available tools
        self.engine.available_tools = {"weather_tool": {"description": "Get weather"}}

        response1 = MagicMock()
        response1.content = """Thought: Need to check weather.
Action: weather_tool
Action Input: {"location": "New York"}
"""

        response2 = MagicMock()
        response2.content = """Thought: Weather is sunny.
Final Answer: It is sunny in New York.
"""

        self.engine.llm_manager.generate_response.side_effect = [response1, response2]
        self.engine.mcp_client.execute_tool.return_value = "Sunny, 25C"

        await self.engine.process_input("What's the weather in NY?")

        # Verify tool execution
        self.engine.mcp_client.execute_tool.assert_called_with(
            "weather_tool", {"location": "New York"}
        )


if __name__ == "__main__":
    unittest.main()
