"""
Direct unit tests for MasterNode termination_reason and iteration_count.

Tests MasterNode in isolation (not through the full graph) to ensure
the termination_reason and iteration_count code paths are covered.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from src.core.ark.nodes.master import MasterNode
from src.core.llm.client import LLMManager


@pytest.fixture
def mock_llm():
    mock = MagicMock(spec=LLMManager)
    return mock


def _make_async_gen(content: str):
    """Return a coroutine function that yields an async generator."""

    async def _gen():
        yield content

    return _gen()


class TestMasterNodeTerminationReason:
    """Verify termination_reason is set correctly for different response types."""

    @pytest.mark.asyncio
    async def test_termination_reason_answer(self, mock_llm):
        """When response_type is 'answer', termination_reason should be 'final_answer'."""
        mock_llm.stream_response = MagicMock(
            return_value=_make_async_gen(
                '{"thought": "Done", "type": "answer", "content": "OK"}'
            )
        )
        node = MasterNode(mock_llm)
        state: dict[str, Any] = {
            "messages": [HumanMessage(content="Hi")],
            "user_input": "Hi",
            "todo_list": [],
            "available_tools": {},
            "scratchpad": {},
            "sender": "user",
            "iteration_count": 0,
            "termination_reason": None,
        }
        result = await node(state, config=None)
        assert result.get("termination_reason") == "final_answer"
        assert result.get("iteration_count", 0) >= 1

    @pytest.mark.asyncio
    async def test_iteration_count_increments(self, mock_llm):
        """iteration_count should be incremented by 1."""
        mock_llm.stream_response = MagicMock(
            return_value=_make_async_gen(
                '{"thought": "Done", "type": "answer", "content": "OK"}'
            )
        )
        node = MasterNode(mock_llm)
        state: dict[str, Any] = {
            "messages": [HumanMessage(content="Hi")],
            "user_input": "Hi",
            "todo_list": [],
            "available_tools": {},
            "scratchpad": {},
            "sender": "user",
            "iteration_count": 5,
            "termination_reason": None,
        }
        result = await node(state, config=None)
        assert result.get("iteration_count") == 6  # 5 + 1

    @pytest.mark.asyncio
    async def test_llm_failure_still_returns_iteration(self, mock_llm):
        """Even when LLM call fails, iteration_count should be returned."""
        mock_llm.stream_response = MagicMock(side_effect=RuntimeError("LLM failure"))
        node = MasterNode(mock_llm)
        state: dict[str, Any] = {
            "messages": [HumanMessage(content="Hi")],
            "user_input": "Hi",
            "todo_list": [],
            "available_tools": {},
            "scratchpad": {},
            "sender": "user",
            "iteration_count": 3,
            "termination_reason": None,
        }
        result = await node(state, config=None)
        assert result.get("iteration_count") == 4

    @pytest.mark.asyncio
    async def test_tool_call_returns_iteration(self, mock_llm):
        """Tool call responses should also include iteration_count."""
        mock_llm.stream_response = MagicMock(
            return_value=_make_async_gen(
                '{"thought": "Need data", "type": "tool_call", '
                '"content": {"name": "search", "arguments": {"q": "test"}}}'
            )
        )
        node = MasterNode(mock_llm)
        state: dict[str, Any] = {
            "messages": [HumanMessage(content="Search")],
            "user_input": "Search",
            "todo_list": [],
            "available_tools": {},
            "scratchpad": {},
            "sender": "user",
            "iteration_count": 0,
            "termination_reason": None,
        }
        result = await node(state, config=None)
        assert result.get("iteration_count") == 1
        # Tool calls should not set termination_reason (tool path continues)
        assert result.get("termination_reason") is None

    @pytest.mark.asyncio
    async def test_multi_tool_call_returns_iteration(self, mock_llm):
        """Multiple tool calls should include iteration_count."""
        mock_llm.stream_response = MagicMock(
            return_value=_make_async_gen(
                '{"thought": "Need data", "type": "tool_calls", '
                '"content": [{"name": "search", "arguments": {"q": "a"}}, '
                '{"name": "fetch", "arguments": {"url": "b"}}]}'
            )
        )
        node = MasterNode(mock_llm)
        state: dict[str, Any] = {
            "messages": [HumanMessage(content="Do stuff")],
            "user_input": "Do stuff",
            "todo_list": [],
            "available_tools": {},
            "scratchpad": {},
            "sender": "user",
            "iteration_count": 0,
            "termination_reason": None,
        }
        result = await node(state, config=None)
        assert result.get("iteration_count") == 1
        assert "sender" in result

    @pytest.mark.asyncio
    async def test_iteration_limit_termination_reason(self, mock_llm):
        """When iteration_count >= 25, termination_reason should be 'iteration_limit'."""
        mock_llm.stream_response = MagicMock(
            return_value=_make_async_gen(
                '{"thought": "Still going", "type": "answer", "content": "Done"}'
            )
        )
        node = MasterNode(mock_llm)
        state: dict[str, Any] = {
            "messages": [HumanMessage(content="Hi")],
            "user_input": "Hi",
            "todo_list": [],
            "available_tools": {},
            "scratchpad": {},
            "sender": "user",
            "iteration_count": 25,  # Hit the hardcoded iteration_limit check
            "termination_reason": None,
        }
        result = await node(state, config=None)
        assert result.get("termination_reason") == "iteration_limit"
        assert result.get("iteration_count") == 26

    @pytest.mark.asyncio
    async def test_no_tool_call_termination_reason(self, mock_llm):
        """When response_type is 'no_tool_call', termination_reason should be 'no_tool_call'."""
        mock_llm.stream_response = MagicMock(
            return_value=_make_async_gen(
                '{"thought": "Nothing needed", "type": "no_tool_call", '
                '"content": "No tools required for this request."}'
            )
        )
        node = MasterNode(mock_llm)
        state: dict[str, Any] = {
            "messages": [HumanMessage(content="Hi")],
            "user_input": "Hi",
            "todo_list": [],
            "available_tools": {},
            "scratchpad": {},
            "sender": "user",
            "iteration_count": 0,
            "termination_reason": None,
        }
        result = await node(state, config=None)
        assert result.get("termination_reason") == "no_tool_call"
        assert result.get("iteration_count") == 1
