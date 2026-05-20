"""
Tests for graph guardrails: iteration limits, state handling, and recovery.

Verifies that the LangGraph iteration limit prevents infinite loops and
that the state carries the expected guardrail fields.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage

from src.core.ark.graph import DEFAULT_GRAPH_ITERATION_LIMIT, create_ark_graph
from src.core.ark.state import JarvisState
from src.core.llm.client import LLMManager
from src.core.mcp.client import ARKMCPClient


@pytest.mark.asyncio
async def create_stream_response(content: str):
    """Helper to create an async generator that yields the content in chunks."""
    chunk_size = 5
    for i in range(0, len(content), chunk_size):
        yield content[i : i + chunk_size]
        await asyncio.sleep(0.01)


class TestGraphIterationLimit:
    """Tests for the graph iteration guardrail."""

    @pytest.mark.asyncio
    async def test_default_iteration_limit_exists(self):
        """The module should export a default iteration limit."""
        assert DEFAULT_GRAPH_ITERATION_LIMIT > 0
        assert isinstance(DEFAULT_GRAPH_ITERATION_LIMIT, int)

    @pytest.mark.asyncio
    async def test_state_has_iteration_count(self):
        """JarvisState should include iteration_count field."""
        # Verify the TypedDict structure by inspecting annotations
        assert "iteration_count" in JarvisState.__annotations__

    @pytest.mark.asyncio
    async def test_state_has_termination_reason(self):
        """JarvisState should include termination_reason field."""
        assert "termination_reason" in JarvisState.__annotations__

    @pytest.mark.asyncio
    async def test_master_node_increments_iteration(self):
        """The MasterNode should increment iteration_count each call."""
        mock_llm = MagicMock(spec=LLMManager)
        # Return a simple answer to stop the loop
        mock_llm.stream_response.return_value = create_stream_response(
            '{"thought": "Done", "type": "answer", "content": "Finished"}'
        )

        mock_mcp = MagicMock(spec=ARKMCPClient)

        graph = create_ark_graph(mock_llm, mock_mcp, iteration_limit=10)

        initial_state = {
            "messages": [HumanMessage(content="Hello")],
            "user_input": "Hello",
            "todo_list": [],
            "scratchpad": {},
            "sender": "user",
            "iteration_count": 0,
            "termination_reason": None,
        }

        result = await graph.ainvoke(initial_state, {"recursion_limit": 20})

        # The final state should have iteration_count >= 1
        assert result.get("iteration_count", 0) >= 1

    @pytest.mark.asyncio
    async def test_iteration_limit_forced_termination(self):
        """When iteration_count exceeds the limit, the graph should stop."""
        mock_llm = MagicMock(spec=LLMManager)
        # Keep returning tool calls to drive iteration
        mock_llm.stream_response.return_value = create_stream_response(
            '{"thought": "Need tool", "type": "tool_call", "content": {"name": "test_tool", "arguments": {}}}'
        )

        mock_mcp = MagicMock(spec=ARKMCPClient)
        mock_mcp.execute_tool = AsyncMock(return_value="tool_result")

        # Set a very low iteration limit so we hit it quickly
        graph = create_ark_graph(mock_llm, mock_mcp, iteration_limit=3)

        initial_state = {
            "messages": [HumanMessage(content="Loop")],
            "user_input": "Loop",
            "todo_list": [],
            "scratchpad": {},
            "sender": "user",
            "iteration_count": 0,
            "termination_reason": None,
        }

        result = await graph.ainvoke(initial_state, {"recursion_limit": 20})

        # The graph should have terminated without error
        assert result is not None
        assert "messages" in result

        # Iteration count should be <= limit + some tolerance for tool rounds
        iteration_count = result.get("iteration_count", 0)
        assert iteration_count <= 10, (
            f"Iteration count {iteration_count} is too high; limit was 3"
        )

    @pytest.mark.asyncio
    async def test_normal_completion_before_limit(self):
        """Graph should complete normally if it reaches a final answer before the limit."""
        mock_llm = MagicMock(spec=LLMManager)
        # Return a valid JSON response that MasterNode can parse
        mock_llm.stream_response.return_value = create_stream_response(
            '{"thought": "Done", "type": "answer", "content": "All good"}'
        )

        mock_mcp = MagicMock(spec=ARKMCPClient)

        graph = create_ark_graph(mock_llm, mock_mcp, iteration_limit=25)

        initial_state = {
            "messages": [HumanMessage(content="Hi")],
            "user_input": "Hi",
            "todo_list": [],
            "scratchpad": {},
            "sender": "user",
            "iteration_count": 0,
            "termination_reason": None,
        }

        result = await graph.ainvoke(initial_state, {"recursion_limit": 30})

        # Should complete with a single iteration and a final answer
        last_content = result["messages"][-1].content
        assert "All good" in last_content, (
            f"Expected 'All good' in response, got: {last_content}"
        )
        assert result.get("iteration_count", 0) >= 1

    @pytest.mark.asyncio
    async def test_termination_reason_final_answer(self):
        """The termination_reason should be 'final_answer' when answering."""
        mock_llm = MagicMock(spec=LLMManager)
        mock_llm.stream_response.return_value = create_stream_response(
            '{"thought": "Done", "type": "answer", "content": "OK"}'
        )

        mock_mcp = MagicMock(spec=ARKMCPClient)

        graph = create_ark_graph(mock_llm, mock_mcp)

        initial_state = {
            "messages": [HumanMessage(content="Hi")],
            "user_input": "Hi",
            "todo_list": [],
            "scratchpad": {},
            "sender": "user",
            "iteration_count": 0,
            "termination_reason": None,
        }

        result = await graph.ainvoke(initial_state, {"recursion_limit": 10})

        # Should have 'final_answer' termination reason
        termination = result.get("termination_reason")
        assert termination == "final_answer", (
            f"Expected 'final_answer', got '{termination}'. "
            f"Messages: {[m.content[:50] for m in result['messages']]}"
        )
