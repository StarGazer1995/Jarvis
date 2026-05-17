"""
ToolsNode 单元测试

测试 src/core/ark/nodes/tools.py 的 ToolsNode 类。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from src.core.ark.nodes.tools import ToolsNode


@pytest.fixture
def mock_mcp_client():
    client = MagicMock()
    client.execute_tool = AsyncMock(return_value="mcp result")
    return client


@pytest.fixture
def tools_node(mock_mcp_client):
    return ToolsNode(mcp_client=mock_mcp_client, max_concurrency=10)


@pytest.fixture
def config():
    return RunnableConfig()


class TestToolsNode:
    """测试 ToolsNode"""

    def test_init(self, tools_node):
        assert tools_node.max_concurrency == 10
        assert tools_node.mcp_client is not None
        assert tools_node.local_tools == {}

    def test_register_tool(self, tools_node):
        def my_tool():
            return "done"

        tools_node.register_tool("my_tool", my_tool)
        assert "my_tool" in tools_node.local_tools
        assert tools_node.local_tools["my_tool"] == my_tool

    @pytest.mark.asyncio
    async def test_no_tool_calls(self, tools_node, config):
        """测试没有 tool_calls 的情况"""
        state = {"messages": [HumanMessage(content="Hello")], "todo_list": []}
        result = await tools_node(state, config)
        assert result["sender"] == "tools"

    @pytest.mark.asyncio
    async def test_manage_tasks_tool(self, tools_node, config):
        """测试 manage_tasks tool"""
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "manage_tasks",
                            "args": {"action": "add", "description": "Test task"},
                            "id": "call_1",
                        }
                    ],
                )
            ],
            "todo_list": [],
        }
        result = await tools_node(state, config)
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0].tool_call_id == "call_1"
        assert "added" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_local_tool_execution(self, tools_node, config):
        """测试本地工具执行"""

        async def my_async_tool(**kwargs):
            return f"processed {kwargs}"

        tools_node.register_tool("my_tool", my_async_tool)

        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "my_tool",
                            "args": {"data": "test"},
                            "id": "call_1",
                        }
                    ],
                )
            ],
            "todo_list": [],
        }
        result = await tools_node(state, config)
        assert len(result["messages"]) == 1
        assert "processed" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_mcp_tool_execution(self, tools_node, mock_mcp_client, config):
        """测试 MCP 工具执行"""
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "mcp_tool",
                            "args": {"query": "test"},
                            "id": "call_1",
                        }
                    ],
                )
            ],
            "todo_list": [],
        }
        result = await tools_node(state, config)
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "mcp result"
        mock_mcp_client.execute_tool.assert_called_once_with(
            "mcp_tool", {"query": "test"}
        )

    @pytest.mark.asyncio
    async def test_extract_depends_on(self, tools_node):
        """测试提取依赖信息"""
        args = {"_depends_on": ["tool_1", "tool_2"], "actual_param": "value"}
        depends = tools_node._extract_depends_on(args)
        assert depends == ["tool_1", "tool_2"]
        assert "_depends_on" not in args  # Should be removed

    @pytest.mark.asyncio
    async def test_extract_depends_on_non_list(self, tools_node):
        """测试提取非列表依赖"""
        args = {"_depends_on": "not_a_list"}
        depends = tools_node._extract_depends_on(args)
        assert depends == []

    @pytest.mark.asyncio
    async def test_extract_depends_on_missing(self, tools_node):
        """测试提取缺失的依赖"""
        args = {"param": "value"}
        depends = tools_node._extract_depends_on(args)
        assert depends == []

    @pytest.mark.asyncio
    async def test_run_sequential_fallback(self, tools_node):
        """测试 _run_sequential 直接"""
        tool_calls_list = [
            MagicMock(id="call_1", name="failing_tool", arguments={}),
        ]

        # Directly test _run_sequential with a failing execute
        with patch.object(
            tools_node,
            "_execute_tool_call",
            new=AsyncMock(side_effect=Exception("fail")),
        ):
            result = await tools_node._run_sequential(tool_calls_list, [])
            assert "Error executing tool" in result["call_1"]

    @pytest.mark.asyncio
    async def test_sequential_with_multiple_tools(self, tools_node):
        """测试顺序执行多个工具"""
        tool_calls_list = [
            MagicMock(id="call_1", name="tool_a", arguments={}),
            MagicMock(id="call_2", name="tool_b", arguments={}),
        ]

        # Directly test _run_sequential with mocked execution
        mock_exec = AsyncMock()
        mock_exec.side_effect = ["result_a", "result_b"]

        with patch.object(tools_node, "_execute_tool_call", mock_exec):
            result = await tools_node._run_sequential(tool_calls_list, [])
            assert result["call_1"] == "result_a"
            assert result["call_2"] == "result_b"

    def test_repr(self):
        """测试 repr"""
        node = ToolsNode(mcp_client=MagicMock(), max_concurrency=5)
        r = repr(node)
        assert isinstance(r, str)
