import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from src.core.agent.react import ReActAgent
from src.core.agent.types import AgentState, AgentStep
from src.core.llm.client import LLMResponse, LLMMessage


class TestReActAgent:
    @pytest.fixture
    def agent(self):
        with (
            patch("src.core.agent.base.LLMManager"),
            patch("src.core.agent.base.load_llm_config"),
        ):
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
        agent._run_loop.assert_called_once_with("Hi", None)

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

        async def mock_stream(*args, **kwargs):
            yield '{"thought": "I know the answer.", "type": "answer", "content": "42"}'

        agent.llm_manager.stream_response = MagicMock(side_effect=mock_stream)

        response = await agent._run_loop("What is 6*7?")

        assert response == "42"
        agent.llm_manager.stream_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_loop_with_action(self, agent):
        """Test loop with one action execution."""
        # First response: Action
        # Second response: Final Answer
        responses = [
            '{"thought": "Check calculator.", "type": "tool_call", "content": {"name": "calculator", "arguments": {"expr": "6*7"}}}',
            '{"thought": "Got it.", "type": "answer", "content": "42"}',
        ]

        async def mock_stream(*args, **kwargs):
            if responses:
                yield responses.pop(0)

        agent.llm_manager.stream_response = MagicMock(side_effect=mock_stream)

        agent.execute_tool = AsyncMock(return_value="42")

        response = await agent._run_loop("Calc 6*7")

        assert response == "42"
        assert agent.llm_manager.stream_response.call_count == 2
        agent.execute_tool.assert_called_once_with("calculator", {"expr": "6*7"})

    @pytest.mark.asyncio
    async def test_run_loop_max_steps(self, agent):
        """Test loop hitting max steps."""
        agent.max_steps = 2

        async def mock_stream(*args, **kwargs):
            yield '{"thought": "Thinking...", "type": "unknown", "content": ""}'

        agent.llm_manager.stream_response = MagicMock(side_effect=mock_stream)

        response = await agent._run_loop("Hi")

        assert "Reached maximum steps" in response
        assert agent.llm_manager.stream_response.call_count == 2

    @pytest.mark.asyncio
    async def test_run_loop_llm_error(self, agent):
        """Test LLM failure handling."""
        agent.llm_manager.stream_response = MagicMock(
            side_effect=Exception("API Error")
        )

        response = await agent._run_loop("Hi")

        assert "Error communicating with LLM" in response

    @pytest.mark.asyncio
    async def test_execute_tool_default(self, agent):
        """Test default execute_tool raises/returns error."""
        result = await agent.execute_tool("any", "params")
        assert "not implemented" in result

    # The following tests are for deprecated methods (_parse_response, _build_prompt)
    # which are no longer used in the new XML-based loop.
    # We remove them or replace them with tests for internal helper methods if needed.
    # For now, we rely on the integration tests above (run_loop) to cover parsing logic.
