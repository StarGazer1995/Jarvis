import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from src.core.agent.react import ReActAgent
from src.core.agent.types import AgentState, AgentStep
from src.core.llm.client import LLMResponse, LLMMessage

class TestReActAgent:

    @pytest.fixture
    def agent(self):
        with patch('src.core.agent.base.LLMManager'), \
             patch('src.core.agent.base.load_llm_config'):
            agent = ReActAgent()
            # Mock LLM manager
            agent.llm_manager = AsyncMock()
            return agent

    @pytest.mark.asyncio
    async def test_process_input_not_ready(self, agent):
        """Test process_input when agent is not ready."""
        agent.state = AgentState.ERROR
        response = await agent.process_input("Hi")
        assert "not ready" in response

    @pytest.mark.asyncio
    async def test_process_input_success(self, agent):
        """Test successful processing flow."""
        agent.state = AgentState.READY
        
        # Mock _run_loop
        agent._run_loop = AsyncMock(return_value="Final Response")
        
        response = await agent.process_input("Hi")
        
        assert response == "Final Response"
        assert agent.state == AgentState.READY
        agent._run_loop.assert_called_once_with("Hi")

    @pytest.mark.asyncio
    async def test_process_input_error(self, agent):
        """Test error handling in process_input."""
        agent.state = AgentState.READY
        agent._run_loop = AsyncMock(side_effect=Exception("Loop Failed"))
        
        response = await agent.process_input("Hi")
        
        assert "Error: Loop Failed" in response
        assert agent.state == AgentState.ERROR

    @pytest.mark.asyncio
    async def test_run_loop_final_answer(self, agent):
        """Test loop reaching Final Answer immediately."""
        agent.llm_manager.generate_response.return_value = LLMResponse(
            content="Thought: I know the answer.\nFinal Answer: 42",
            usage={}
        )
        
        response = await agent._run_loop("What is 6*7?")
        
        assert response == "42"
        agent.llm_manager.generate_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_loop_with_action(self, agent):
        """Test loop with one action execution."""
        # First response: Action
        # Second response: Final Answer
        agent.llm_manager.generate_response.side_effect = [
            LLMResponse(content='Thought: Check calculator.\nAction: calculator\nAction Input: {"expr": "6*7"}', usage={}),
            LLMResponse(content='Thought: Got it.\nFinal Answer: 42', usage={})
        ]
        
        agent.execute_tool = AsyncMock(return_value="42")
        
        response = await agent._run_loop("Calc 6*7")
        
        assert response == "42"
        assert agent.llm_manager.generate_response.call_count == 2
        agent.execute_tool.assert_called_once_with("calculator", {"expr": "6*7"})

    @pytest.mark.asyncio
    async def test_run_loop_max_steps(self, agent):
        """Test loop hitting max steps."""
        agent.max_steps = 2
        agent.llm_manager.generate_response.return_value = LLMResponse(
            content="Thought: Thinking...", usage={}
        )
        
        response = await agent._run_loop("Hi")
        
        assert "Reached maximum steps" in response
        assert agent.llm_manager.generate_response.call_count == 2

    @pytest.mark.asyncio
    async def test_run_loop_llm_error(self, agent):
        """Test LLM failure handling."""
        agent.llm_manager.generate_response.side_effect = Exception("API Error")
        
        response = await agent._run_loop("Hi")
        
        assert "Error communicating with LLM" in response

    def test_parse_response_final_answer(self, agent):
        """Test parsing Final Answer."""
        text = "Thought: Done.\nFinal Answer: Result"
        step = agent._parse_response(text)
        
        assert step.action == "Final Answer"
        assert step.observation == "Result"
        assert step.thought == "Done."

    def test_parse_response_action_json(self, agent):
        """Test parsing Action with JSON input."""
        text = 'Thought: Run tool.\nAction: test_tool\nAction Input: {"key": "value"}'
        step = agent._parse_response(text)
        
        assert step.action == "test_tool"
        assert step.action_input == {"key": "value"}
        assert step.thought == "Run tool."

    def test_parse_response_action_string(self, agent):
        """Test parsing Action with string input."""
        text = 'Thought: Run tool.\nAction: search\nAction Input: query string'
        step = agent._parse_response(text)
        
        assert step.action == "search"
        assert step.action_input == "query string"

    def test_parse_response_action_code_block(self, agent):
        """Test parsing Action with code block input."""
        text = 'Thought: Run code.\nAction: python\nAction Input: ```python\nprint("hi")\n```'
        step = agent._parse_response(text)
        
        assert step.action == "python"
        assert step.action_input == 'print("hi")'

    def test_parse_response_invalid(self, agent):
        """Test parsing invalid format."""
        text = "Just some random text."
        step = agent._parse_response(text)
        
        assert step.action is None
        assert "Error: Could not parse" in step.observation

    def test_build_prompt(self, agent):
        """Test prompt construction."""
        steps = [
            AgentStep(thought="t1", action="a1", action_input="i1", observation="o1"),
            AgentStep(thought="t2", action=None, action_input=None, observation="o2")
        ]
        
        messages = agent._build_prompt("User Input", steps)
        
        assert len(messages) == 3 # System, User, Assistant (History)
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert "User Input" in messages[1].content
        assert messages[2].role == "assistant"
        
        history = messages[2].content
        assert "Thought: t1" in history
        assert "Action: a1" in history
        assert "Observation: o1" in history
        assert "Thought: t2" in history
        assert "Observation: o2" in history

    @pytest.mark.asyncio
    async def test_execute_tool_default(self, agent):
        """Test default execute_tool raises/returns error."""
        result = await agent.execute_tool("any", "params")
        assert "not implemented" in result
