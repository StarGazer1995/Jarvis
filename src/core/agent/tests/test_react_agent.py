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
        response_obj.content = """Thought: I know the answer.
Final Answer: The answer is 42.
"""
        self.agent.llm_manager.generate_response.return_value = response_obj
        
        result = await self.agent.process_input("What is the answer?")
        
        self.assertEqual(result, "The answer is 42.")
        self.assertEqual(self.agent.state, AgentState.READY)
        
    async def test_react_loop_with_tool(self):
        """Test thought -> action -> observation -> final answer."""
        
        # Step 1: Tool call
        response1 = MagicMock()
        response1.content = """Thought: I need to calculate.
Action: calculator
Action Input: {"expr": "2+2"}
"""
        
        # Step 2: Final answer
        response2 = MagicMock()
        response2.content = """Thought: The result is 4.
Final Answer: 4
"""
        
        self.agent.llm_manager.generate_response.side_effect = [response1, response2]
        self.agent.execute_tool.return_value = "4"
        
        result = await self.agent.process_input("Calculate 2+2")
        
        self.assertEqual(result, "4")
        self.agent.execute_tool.assert_called_with("calculator", {"expr": "2+2"})
        self.assertEqual(self.agent.llm_manager.generate_response.call_count, 2)

    async def test_max_steps(self):
        """Test that the loop respects max steps."""
        self.agent.max_steps = 2
        
        response = MagicMock()
        response.content = """Thought: Thinking...
Action: wait
Action Input: "forever"
"""
        self.agent.llm_manager.generate_response.return_value = response
        
        result = await self.agent.process_input("Wait")
        
        self.assertIn("Reached maximum steps", result)
        self.assertEqual(self.agent.llm_manager.generate_response.call_count, 2)

if __name__ == '__main__':
    unittest.main()
