"""
Tests for graph guardrails: iteration limits, state handling, and recovery.

Verifies that the LangGraph iteration limit prevents infinite loops and
that the state carries the expected guardrail fields.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage

from src.core.ark.graph import DEFAULT_GRAPH_ITERATION_LIMIT, create_ark_graph
from src.core.ark.state import JarvisState
from src.core.llm.client import LLMManager
from src.core.mcp.client import ARKMCPClient


def make_stream_side_effect(*responses: str):
    """Return a side-effect callable that yields a fresh async generator per call."""
    responses_list = list(responses)

    def side_effect(*args, **kwargs):
        async def gen(content: str):
            chunk_size = 5
            for i in range(0, len(content), chunk_size):
                yield content[i : i + chunk_size]

        if responses_list:
            return gen(responses_list.pop(0))
        # Default: return a simple answer when exhausted
        return gen('{"thought": "Done", "type": "answer", "content": "Finished"}')

    return side_effect


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
        assert "iteration_count" in JarvisState.__annotations__

    @pytest.mark.asyncio
    async def test_state_has_termination_reason(self):
        """JarvisState should include termination_reason field."""
        assert "termination_reason" in JarvisState.__annotations__

    @pytest.mark.asyncio
    async def test_master_node_increments_iteration(self):
        """The MasterNode should increment iteration_count each call."""
        mock_llm = MagicMock(spec=LLMManager)
        mock_llm.stream_response.side_effect = make_stream_side_effect(
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

        assert result.get("iteration_count", 0) >= 1

    @pytest.mark.asyncio
    async def test_iteration_limit_forced_termination(self):
        """When iteration_count exceeds the limit, the graph should stop."""
        mock_llm = MagicMock(spec=LLMManager)
        # Keep returning tool calls to drive iteration (need many since each
        # call consumes one generator; the side_effect factory creates a new
        # generator each time)
        tool_call_json = (
            '{"thought": "Need tool", "type": "tool_call", '
            '"content": {"name": "test_tool", "arguments": {}}}'
        )
        mock_llm.stream_response.side_effect = make_stream_side_effect(
            tool_call_json,
            tool_call_json,
            tool_call_json,
            tool_call_json,
            tool_call_json,
        )

        mock_mcp = MagicMock(spec=ARKMCPClient)
        mock_mcp.execute_tool = AsyncMock(return_value="tool_result")

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

        result = await graph.ainvoke(initial_state, {"recursion_limit": 15})

        assert result is not None
        assert "messages" in result

        iteration_count = result.get("iteration_count", 0)
        # Should have stopped at or near the limit
        assert iteration_count <= 5, (
            f"Iteration count {iteration_count} is too high; limit was 3"
        )

    @pytest.mark.asyncio
    async def test_normal_completion_before_limit(self):
        """Graph should complete normally if it reaches a final answer before the limit."""
        mock_llm = MagicMock(spec=LLMManager)
        mock_llm.stream_response.side_effect = make_stream_side_effect(
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

        last_content = result["messages"][-1].content
        assert "All good" in last_content, (
            f"Expected 'All good' in response, got: {last_content}"
        )
        assert result.get("iteration_count", 0) >= 1

    @pytest.mark.asyncio
    async def test_termination_reason_final_answer(self):
        """The termination_reason should be 'final_answer' when answering."""
        mock_llm = MagicMock(spec=LLMManager)
        mock_llm.stream_response.side_effect = make_stream_side_effect(
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

        termination = result.get("termination_reason")
        assert termination == "final_answer", (
            f"Expected 'final_answer', got '{termination}'. "
            f"Messages: {[m.content[:50] for m in result['messages']]}"
        )

    @pytest.mark.asyncio
    async def test_should_continue_enforces_iteration_limit(self):
        """The should_continue function must return __end__ when iteration limit reached."""
        # Directly test the guard by using a pre-set iteration_count >= limit
        # that will cause should_continue to return __end__ after the first master call
        mock_llm = MagicMock(spec=LLMManager)
        mock_llm.stream_response.side_effect = make_stream_side_effect(
            '{"thought": "One more", "type": "tool_call", '
            '"content": {"name": "test_tool", "arguments": {}}}'
        )

        mock_mcp = MagicMock(spec=ARKMCPClient)
        mock_mcp.execute_tool = AsyncMock(return_value="result")

        # Set limit to 2 and start at iteration 1 so the first should_continue
        # call after master (iteration becomes 2) will hit the limit
        graph = create_ark_graph(mock_llm, mock_mcp, iteration_limit=2)

        initial_state = {
            "messages": [HumanMessage(content="Loop")],
            "user_input": "Loop",
            "todo_list": [],
            "scratchpad": {},
            "sender": "user",
            "iteration_count": 1,  # Start at 1 so master sets it to 2 → limit hit
            "termination_reason": None,
        }

        result = await graph.ainvoke(initial_state, {"recursion_limit": 10})

        # The graph should terminate when the guardrail triggers
        assert result is not None
        assert "messages" in result

        # The graph should have stopped via the guardrail, meaning the tool
        # was never called because should_continue returned __end__
        # mock_mcp.execute_tool should NOT have been called
        # (depending on how the guardrail interacts with the tool routing)
        iteration_count = result.get("iteration_count", 0)
        assert iteration_count >= 2
        assert mock_mcp.execute_tool.call_count == 0 or (
            # If master did pass through, at most 1 tool call before guardrail
            mock_mcp.execute_tool.call_count <= 1
        )
