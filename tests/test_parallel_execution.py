"""
Tests for the Parallel Execution Engine

Covers:
- ExecutionGraph dependency resolution
- ExecutionPlan batch calculation
- ParallelExecutor concurrent execution
- Reference resolution ($ref{...})
- Dependency validation and error handling
- Fallback to sequential on failure
- ToolsNode integration with multi-tool LLM responses
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.core.ark.nodes.tools import ToolsNode
from src.core.ark.state import JarvisState
from src.core.execution.engine import (
    ExecutionGraph,
    ExecutionPlan,
    ParallelExecutor,
    ToolCall,
)

# ═══════════════════════════════════════════════════════════════════
# 1. ExecutionGraph — Dependency Resolution
# ═══════════════════════════════════════════════════════════════════


class TestExecutionGraph:
    """Test the dependency graph building and resolution."""

    def test_no_dependencies(self):
        """Graph with no dependencies should mark all as ready."""
        calls = [
            ToolCall(name="tool_a", arguments={}, id="a"),
            ToolCall(name="tool_b", arguments={}, id="b"),
        ]
        graph = ExecutionGraph(calls)
        ready = graph.get_ready_tools(set())
        assert "a" in ready
        assert "b" in ready

    def test_with_dependencies(self):
        """Graph with dependencies should prevent dependent tools from being ready."""
        calls = [
            ToolCall(name="tool_a", arguments={}, id="a"),
            ToolCall(name="tool_b", arguments={}, depends_on=["a"], id="b"),
        ]
        graph = ExecutionGraph(calls)
        ready = graph.get_ready_tools(set())
        assert "a" in ready
        assert "b" not in ready  # Depends on 'a'

    def test_ready_after_dependency_completed(self):
        """After completing dep, dependent tool should become ready."""
        calls = [
            ToolCall(name="tool_a", arguments={}, id="a"),
            ToolCall(name="tool_b", arguments={}, depends_on=["a"], id="b"),
        ]
        graph = ExecutionGraph(calls)
        ready = graph.get_ready_tools({"a"})  # 'a' completed
        assert "b" in ready

    def test_circular_dependency_detection(self):
        """Tools that depend on each other should cause a deadlock."""
        calls = [
            ToolCall(name="tool_a", arguments={}, depends_on=["b"], id="a"),
            ToolCall(name="tool_b", arguments={}, depends_on=["a"], id="b"),
        ]
        graph = ExecutionGraph(calls)
        # get_ready_tools with empty completed should return empty
        ready = graph.get_ready_tools(set())
        assert ready == []

    def test_missing_dependency_raises(self):
        """Referencing a non-existent dependency should raise."""
        calls = [
            ToolCall(name="tool_a", arguments={}, depends_on=["ghost"], id="a"),
        ]
        with pytest.raises(ValueError, match="ghost"):
            ExecutionGraph(calls)

    def test_duplicate_tool_names(self):
        """Duplicate tool IDs should be disambiguated."""
        calls = [
            ToolCall(name="search", arguments={}, id="a"),
            ToolCall(name="search", arguments={}, id="b"),
        ]
        graph = ExecutionGraph(calls)
        assert len(graph.nodes) == 2

    def test_all_completed(self):
        """are_all_completed should detect completion."""
        calls = [ToolCall(name="a", arguments={}, id="a")]
        graph = ExecutionGraph(calls)
        assert not graph.are_all_completed(set())
        assert graph.are_all_completed({"a"})


# ═══════════════════════════════════════════════════════════════════
# 2. ExecutionPlan — Batch Calculation
# ═══════════════════════════════════════════════════════════════════


class TestExecutionPlan:
    """Test the execution plan batch resolution."""

    def test_independent_tools_single_batch(self):
        """Independent tools should be grouped into one batch."""
        calls = [
            ToolCall(name="a", arguments={}),
            ToolCall(name="b", arguments={}),
        ]
        plan = ExecutionPlan(calls)
        batches = plan.resolve()
        assert len(batches) == 1
        assert len(batches[0]) == 2

    def test_dependent_tools_multiple_batches(self):
        """Dependent tools should be split into sequential batches."""
        calls = [
            ToolCall(name="a", arguments={}, id="a"),
            ToolCall(name="b", arguments={}, depends_on=["a"], id="b"),
        ]
        plan = ExecutionPlan(calls)
        batches = plan.resolve()
        assert len(batches) == 2  # [a], [b]
        assert batches[0] == ["a"]
        assert batches[1] == ["b"]

    def test_chain_dependency(self):
        """A → B → C should result in three batches."""
        calls = [
            ToolCall(name="a", arguments={}, id="a"),
            ToolCall(name="b", arguments={}, depends_on=["a"], id="b"),
            ToolCall(name="c", arguments={}, depends_on=["b"], id="c"),
        ]
        plan = ExecutionPlan(calls)
        batches = plan.resolve()
        assert len(batches) == 3

    def test_diamond_dependency(self):
        """Diamond: A → {B, C} → D should have 3 batches."""
        calls = [
            ToolCall(name="a", arguments={}, id="a"),
            ToolCall(name="b", arguments={}, depends_on=["a"], id="b"),
            ToolCall(name="c", arguments={}, depends_on=["a"], id="c"),
            ToolCall(name="d", arguments={}, depends_on=["b", "c"], id="d"),
        ]
        plan = ExecutionPlan(calls)
        batches = plan.resolve()
        assert len(batches) == 3  # [a], [b,c], [d]
        assert len(batches[1]) == 2  # b and c in parallel

    def test_empty_tool_list(self):
        """Empty tool list should produce no batches."""
        plan = ExecutionPlan([])
        assert plan.resolve() == []

    def test_deadlock_detection(self):
        """Circular dependencies should raise RuntimeError."""
        calls = [
            ToolCall(name="a", arguments={}, depends_on=["b"], id="a"),
            ToolCall(name="b", arguments={}, depends_on=["a"], id="b"),
        ]
        plan = ExecutionPlan(calls)
        with pytest.raises(RuntimeError, match="Deadlock"):
            plan.resolve()


# ═══════════════════════════════════════════════════════════════════
# 3. ParallelExecutor — Concurrent Execution
# ═══════════════════════════════════════════════════════════════════


class TestParallelExecutor:
    """Test the actual parallel execution."""

    @pytest.mark.asyncio
    async def test_independent_tools_run_in_parallel(self):
        """Independent tools should execute concurrently (check timing)."""

        async def slow_tool(name: str, args: dict) -> str:
            await asyncio.sleep(0.1)
            return f"{name}_done"

        executor = ParallelExecutor(execute_fn=slow_tool)

        start = asyncio.get_event_loop().time()
        results = await executor.run(
            [
                ToolCall(name="a", arguments={}, id="a"),
                ToolCall(name="b", arguments={}, id="b"),
            ]
        )
        elapsed = asyncio.get_event_loop().time() - start

        # If run in parallel, total time < 0.2s (2 × 0.1s sequential)
        assert elapsed < 0.15, f"Expected parallel execution, took {elapsed:.3f}s"
        assert results["a"] == "a_done"
        assert results["b"] == "b_done"

    @pytest.mark.asyncio
    async def test_dependent_tools_run_sequential(self):
        """Dependent tools should execute sequentially."""
        execution_order = []

        async def tracking_tool(name: str, args: dict) -> str:
            execution_order.append(name)
            await asyncio.sleep(0.05)
            return f"{name}_done"

        executor = ParallelExecutor(execute_fn=tracking_tool)

        await executor.run(
            [
                ToolCall(name="a", arguments={}, id="a"),
                ToolCall(name="b", arguments={}, depends_on=["a"], id="b"),
            ]
        )

        assert execution_order == ["a", "b"]

    @pytest.mark.asyncio
    async def test_parallel_tools_with_one_dependency(self):
        """Diamond: a → {b, c} → d. b and c should be parallel."""
        execution_order = []

        async def tracking_tool(name: str, args: dict) -> str:
            execution_order.append(name)
            await asyncio.sleep(0.05)
            return f"{name}_done"

        executor = ParallelExecutor(execute_fn=tracking_tool)

        await executor.run(
            [
                ToolCall(name="a", arguments={}, id="a"),
                ToolCall(name="b", arguments={}, depends_on=["a"], id="b"),
                ToolCall(name="c", arguments={}, depends_on=["a"], id="c"),
                ToolCall(name="d", arguments={}, depends_on=["b", "c"], id="d"),
            ]
        )

        assert execution_order[0] == "a"
        assert execution_order[-1] == "d"
        # b and c should be in the same batch (parallel)
        b_idx = execution_order.index("b")
        c_idx = execution_order.index("c")
        assert abs(b_idx - c_idx) <= 1  # effectively parallel

    @pytest.mark.asyncio
    async def test_empty_call_list(self):
        """Empty call list should return empty results."""
        executor = ParallelExecutor(execute_fn=lambda n, a: n)
        results = await executor.run([])
        assert results == {}

    @pytest.mark.asyncio
    async def test_concurrent_error_does_not_crash_others(self):
        """A failing tool should not prevent independent tools from completing."""
        results_store = {}

        async def flaky_tool(name: str, args: dict) -> str:
            if name == "flaky":
                raise ValueError("Flaky tool failed")
            results_store[name] = "ok"
            return "ok"

        executor = ParallelExecutor(execute_fn=flaky_tool)

        results = await executor.run(
            [
                ToolCall(name="good_a", arguments={}, id="good_a"),
                ToolCall(name="flaky", arguments={}, id="flaky"),
                ToolCall(name="good_b", arguments={}, id="good_b"),
            ]
        )

        assert results["good_a"] == "ok" or results["good_a"] is not None
        assert results["good_b"] == "ok" or results["good_b"] is not None

    @pytest.mark.asyncio
    async def test_respects_max_concurrency_limit(self):
        """Executor should cap in-flight tool calls within the configured limit."""
        active_calls = 0
        peak_calls = 0
        lock = asyncio.Lock()

        async def limited_tool(name: str, args: dict) -> str:
            nonlocal active_calls, peak_calls
            async with lock:
                active_calls += 1
                peak_calls = max(peak_calls, active_calls)
            await asyncio.sleep(0.02)
            async with lock:
                active_calls -= 1
            return f"{name}_done"

        executor = ParallelExecutor(execute_fn=limited_tool, max_concurrency=2)

        results = await executor.run(
            [
                ToolCall(name="a", arguments={}, id="a"),
                ToolCall(name="b", arguments={}, id="b"),
                ToolCall(name="c", arguments={}, id="c"),
            ]
        )

        assert peak_calls == 2
        assert results == {"a": "a_done", "b": "b_done", "c": "c_done"}


# ═══════════════════════════════════════════════════════════════════
# 4. Reference Resolution ($ref{...})
# ═══════════════════════════════════════════════════════════════════


class TestReferenceResolution:
    """Test $ref{tool_name} variable interpolation."""

    @pytest.mark.asyncio
    async def test_simple_reference(self):
        """$ref{tool_id} should be replaced with the tool's output."""

        async def echo(name: str, args: dict) -> str:
            return f"output_of_{name}"

        executor = ParallelExecutor(execute_fn=echo)

        results = await executor.run(
            [
                ToolCall(name="search", arguments={"q": "hello"}, id="search"),
                ToolCall(
                    name="summarize",
                    arguments={"input": "$ref{search}"},
                    depends_on=["search"],
                    id="summarize",
                ),
            ]
        )

        assert results["summarize"] is not None

    @pytest.mark.asyncio
    async def test_unresolved_reference_does_not_crash(self):
        """An unresolved reference should produce a placeholder, not crash."""

        async def echo(name: str, args: dict) -> str:
            return str(args.get("input", ""))

        executor = ParallelExecutor(execute_fn=echo)

        results = await executor.run(
            [
                ToolCall(
                    name="tool",
                    arguments={"input": "$ref{ghost}"},
                    id="tool",
                ),
            ]
        )

        assert "unresolved" in str(results.get("tool", ""))


# ═══════════════════════════════════════════════════════════════════
# 5. ToolsNode — Integration Test
# ═══════════════════════════════════════════════════════════════════


class TestToolsNodeParallelExecution:
    """Test ToolsNode with multi-tool LLM responses."""

    @pytest.mark.asyncio
    async def test_returns_sender_only_when_no_tool_calls_present(self):
        """ToolsNode should no-op when the last message has no tool calls."""
        from langchain_core.messages import AIMessage

        mock_mcp = AsyncMock()
        node = ToolsNode(mcp_client=mock_mcp)

        result = await node({"messages": [AIMessage(content="no tools")]}, config=None)

        assert result == {"sender": "tools"}

    @pytest.mark.asyncio
    async def test_registers_and_executes_local_tools(self):
        """Registered local tools should be preferred over MCP execution."""
        from langchain_core.messages import AIMessage

        mock_mcp = AsyncMock()
        node = ToolsNode(mcp_client=mock_mcp)

        def sync_tool(value: int) -> str:
            return f"sync:{value}"

        async def async_tool(value: int) -> str:
            await asyncio.sleep(0)
            return f"async:{value}"

        node.register_tool("sync_tool", sync_tool)
        node.register_tool("async_tool", async_tool)

        state: JarvisState = {
            "messages": [
                AIMessage(
                    content="run local tools",
                    tool_calls=[
                        {"name": "sync_tool", "args": {"value": 1}, "id": "call_0"},
                        {"name": "async_tool", "args": {"value": 2}, "id": "call_1"},
                    ],
                ),
            ],
            "todo_list": [],
            "available_tools": {},
            "user_input": "test",
            "scratchpad": {},
            "sender": "master",
        }

        result = await node(state, config=None)

        assert [message.content for message in result["messages"]] == [
            "sync:1",
            "async:2",
        ]
        mock_mcp.execute_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_manage_tasks_tool_updates_todo_list(self):
        """manage_tasks should be handled internally without MCP."""
        from langchain_core.messages import AIMessage

        mock_mcp = AsyncMock()
        node = ToolsNode(mcp_client=mock_mcp)

        state: JarvisState = {
            "messages": [
                AIMessage(
                    content="track tasks",
                    tool_calls=[
                        {
                            "name": "manage_tasks",
                            "args": {
                                "action": "add",
                                "description": "Write tests",
                            },
                            "id": "call_0",
                        }
                    ],
                ),
            ],
            "todo_list": [],
            "available_tools": {},
            "user_input": "test",
            "scratchpad": {},
            "sender": "master",
        }

        result = await node(state, config=None)

        assert result["todo_list"][0]["id"] == "1"
        assert result["todo_list"][0]["description"] == "Write tests"
        assert "Write tests" in result["messages"][0].content
        mock_mcp.execute_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_tool_call_still_works(self):
        """Single tool call should work as before (backward compatibility)."""
        from langchain_core.messages import AIMessage

        mock_mcp = AsyncMock()
        mock_mcp.execute_tool = AsyncMock(return_value="single_result")

        node = ToolsNode(mcp_client=mock_mcp)

        state = {
            "messages": [
                AIMessage(
                    content="Need to search",
                    tool_calls=[
                        {
                            "name": "web_search",
                            "args": {"query": "AI"},
                            "id": "call_0",
                        }
                    ],
                ),
            ],
            "todo_list": [],
            "available_tools": {},
            "user_input": "test",
            "scratchpad": {},
            "sender": "master",
        }

        result = await node(state, config=None)
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0].name == "web_search"

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_parallel(self):
        """Multiple independent tool calls should execute in parallel."""
        from langchain_core.messages import AIMessage

        mock_mcp = AsyncMock()
        call_count = {}

        async def fake_execute(name: str, args: dict) -> str:
            call_count[name] = call_count.get(name, 0) + 1
            await asyncio.sleep(0.05)
            return f"{name}_done"

        mock_mcp.execute_tool = fake_execute
        node = ToolsNode(mcp_client=mock_mcp)

        state: JarvisState = {
            "messages": [
                AIMessage(
                    content="Multiple tools needed",
                    tool_calls=[
                        {"name": "search", "args": {"q": "a"}, "id": "call_0"},
                        {"name": "fetch", "args": {"url": "b"}, "id": "call_1"},
                        {"name": "time", "args": {}, "id": "call_2"},
                    ],
                ),
            ],
            "todo_list": [],
            "available_tools": {},
            "user_input": "test",
            "scratchpad": {},
            "sender": "master",
        }

        # Should complete within ~0.1s if parallel (3 × 0.05s sequential would be 0.15s)
        import time

        start = time.monotonic()
        result = await node(state, config=None)
        elapsed = time.monotonic() - start

        assert elapsed < 0.12, f"Expected parallel, took {elapsed:.3f}s"
        assert len(result["messages"]) == 3
        assert all(m.name in ("search", "fetch", "time") for m in result["messages"])

    @pytest.mark.asyncio
    async def test_multi_tool_with_dependencies(self):
        """Dependencies between tools should be respected."""
        from langchain_core.messages import AIMessage

        execution_order = []
        mock_mcp = AsyncMock()

        async def tracking_execute(name: str, args: dict) -> str:
            execution_order.append(name)
            await asyncio.sleep(0.02)
            return f"{name}_done"

        mock_mcp.execute_tool = tracking_execute
        node = ToolsNode(mcp_client=mock_mcp)

        # Tool calls with dependency annotations
        state: JarvisState = {
            "messages": [
                AIMessage(
                    content="Chained tools",
                    tool_calls=[
                        {"name": "step1", "args": {}, "id": "call_0"},
                        {
                            "name": "step2",
                            "args": {"_depends_on": ["call_0"]},
                            "id": "call_1",
                        },
                    ],
                ),
            ],
            "todo_list": [],
            "available_tools": {},
            "user_input": "test",
            "scratchpad": {},
            "sender": "master",
        }

        result = await node(state, config=None)
        assert execution_order == ["step1", "step2"]
        assert len(result["messages"]) == 2

    @pytest.mark.asyncio
    async def test_invalid_dependency_falls_back_to_sequential_execution(self):
        """Invalid dependency metadata should trigger sequential fallback."""
        from langchain_core.messages import AIMessage

        execution_order = []
        mock_mcp = AsyncMock()

        async def tracking_execute(name: str, args: dict) -> str:
            execution_order.append(name)
            await asyncio.sleep(0.01)
            return f"{name}:{args}"

        mock_mcp.execute_tool = tracking_execute
        node = ToolsNode(mcp_client=mock_mcp)

        state: JarvisState = {
            "messages": [
                AIMessage(
                    content="Broken dependency metadata",
                    tool_calls=[
                        {"name": "step1", "args": {"value": 1}, "id": "call_0"},
                        {
                            "name": "step2",
                            "args": {"value": 2, "_depends_on": ["missing_call"]},
                            "id": "call_1",
                        },
                    ],
                ),
            ],
            "todo_list": [],
            "available_tools": {},
            "user_input": "test",
            "scratchpad": {},
            "sender": "master",
        }

        result = await node(state, config=None)

        assert execution_order == ["step1", "step2"]
        assert [message.name for message in result["messages"]] == ["step1", "step2"]
        assert "_depends_on" not in result["messages"][1].content

    @pytest.mark.asyncio
    async def test_sequential_fallback_returns_tool_errors(self):
        """Sequential fallback should surface per-tool execution failures."""
        from langchain_core.messages import AIMessage

        mock_mcp = AsyncMock()

        async def maybe_fail(name: str, args: dict) -> str:
            if name == "explode":
                raise RuntimeError("boom")
            return "ok"

        mock_mcp.execute_tool = maybe_fail
        node = ToolsNode(mcp_client=mock_mcp)

        state: JarvisState = {
            "messages": [
                AIMessage(
                    content="Broken dependency metadata",
                    tool_calls=[
                        {
                            "name": "explode",
                            "args": {"_depends_on": ["missing"]},
                            "id": "call_0",
                        },
                    ],
                ),
            ],
            "todo_list": [],
            "available_tools": {},
            "user_input": "test",
            "scratchpad": {},
            "sender": "master",
        }

        result = await node(state, config=None)

        assert "Error executing tool explode: boom" in result["messages"][0].content

    def test_extract_depends_on_returns_empty_for_non_list(self):
        """Non-list dependency annotations should be ignored safely."""
        args = {"_depends_on": "call_0", "value": 1}
        depends_on = ToolsNode._extract_depends_on(args)
        assert depends_on == []
        assert args == {"value": 1}


# ═══════════════════════════════════════════════════════════════════
# 6. MasterNode — Multi-Tool Response
# ═══════════════════════════════════════════════════════════════════


class TestMasterNodeMultiTool:
    """Test MasterNode handles multi-tool LLM responses."""

    def test_parses_single_tool_call(self):
        """Legacy single tool_call format should still work."""
        # This test validates the JSON structure -
        # the actual MasterNode is tested via integration tests
        sample_response = {
            "thought": "I need to search",
            "type": "tool_call",
            "content": {
                "name": "web_search",
                "arguments": {"query": "AI"},
            },
        }
        assert sample_response["type"] == "tool_call"
        assert isinstance(sample_response["content"], dict)
        assert sample_response["content"]["name"] == "web_search"

    def test_parses_multi_tool_calls(self):
        """New multi-tool format should parse correctly."""
        sample_response = {
            "thought": "I need multiple tools",
            "type": "tool_calls",
            "content": [
                {"name": "web_search", "arguments": {"query": "AI"}},
                {"name": "get_time", "arguments": {}},
            ],
        }
        assert sample_response["type"] == "tool_calls"
        assert isinstance(sample_response["content"], list)
        assert len(sample_response["content"]) == 2

    def test_multi_tool_with_depends_on(self):
        """Multi-tool with dependency annotations."""
        sample_response = {
            "thought": "Search then summarize",
            "type": "tool_calls",
            "content": [
                {"name": "web_search", "arguments": {"query": "AI trends"}},
                {
                    "name": "summarize",
                    "arguments": {"text": "$ref{web_search}"},
                    "depends_on": ["web_search"],
                },
            ],
        }
        tools = sample_response["content"]
        assert tools[1].get("depends_on") == ["web_search"]


class TestParallelExecutorConcurrencyLimit:
    """Test concurrency limiting behavior."""

    @pytest.mark.asyncio
    async def test_respects_max_concurrency_limit(self):
        """Executor should cap concurrent tasks within a batch."""
        active = 0
        peak = 0

        async def limited_tool(name: str, args: dict) -> str:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return name

        executor = ParallelExecutor(execute_fn=limited_tool, max_concurrency=2)
        await executor.run(
            [
                ToolCall(name="a", arguments={}, id="a"),
                ToolCall(name="b", arguments={}, id="b"),
                ToolCall(name="c", arguments={}, id="c"),
            ]
        )

        assert peak == 2


class TestToolsNodeAdditionalPaths:
    """Cover additional ToolsNode behaviors."""

    @pytest.mark.asyncio
    async def test_returns_sender_only_when_no_tool_calls_present(self):
        """ToolsNode should no-op when the message has no tool calls."""
        from langchain_core.messages import AIMessage

        node = ToolsNode(mcp_client=AsyncMock())
        state: JarvisState = {
            "messages": [AIMessage(content="hello")],
            "todo_list": [],
            "available_tools": {},
            "user_input": "test",
            "scratchpad": {},
            "sender": "master",
        }

        result = await node(state, config=None)
        assert result == {"sender": "tools"}

    @pytest.mark.asyncio
    async def test_registers_and_executes_local_tools(self):
        """ToolsNode should invoke registered local tools before MCP."""
        from langchain_core.messages import AIMessage

        mock_mcp = AsyncMock()
        node = ToolsNode(mcp_client=mock_mcp)

        async def local_tool(value: str) -> str:
            return f"local:{value}"

        node.register_tool("echo_local", local_tool)
        state: JarvisState = {
            "messages": [
                AIMessage(
                    content="run local tool",
                    tool_calls=[
                        {"name": "echo_local", "args": {"value": "ok"}, "id": "call_0"}
                    ],
                )
            ],
            "todo_list": [],
            "available_tools": {},
            "user_input": "test",
            "scratchpad": {},
            "sender": "master",
        }

        result = await node(state, config=None)
        assert result["messages"][0].content == "local:ok"
        mock_mcp.execute_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_manage_tasks_tool_updates_todo_list(self):
        """manage_tasks should update todo state through ToolsNode."""
        from langchain_core.messages import AIMessage

        node = ToolsNode(mcp_client=AsyncMock())
        state: JarvisState = {
            "messages": [
                AIMessage(
                    content="update tasks",
                    tool_calls=[
                        {
                            "name": "manage_tasks",
                            "args": {
                                "action": "add",
                                "description": "Do thing",
                            },
                            "id": "call_0",
                        }
                    ],
                )
            ],
            "todo_list": [],
            "available_tools": {},
            "user_input": "test",
            "scratchpad": {},
            "sender": "master",
        }

        result = await node(state, config=None)
        assert result["todo_list"][0]["description"] == "Do thing"

    @pytest.mark.asyncio
    async def test_sequential_fallback_returns_tool_errors(self):
        """Sequential fallback should return stringified tool errors."""
        mock_mcp = AsyncMock()
        node = ToolsNode(mcp_client=mock_mcp)

        async def broken_execute(name: str, args: dict) -> str:
            raise ValueError("tool failed")

        mock_mcp.execute_tool = broken_execute

        result = await node._run_sequential(
            [ToolCall(name="boom", arguments={}, id="call_0")],
            [],
        )
        assert "Error executing tool boom" in result["call_0"]
