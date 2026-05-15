"""
Simple ARK Engine 单元测试

测试 SimpleARKEngine 特有的方法：initialize, _discover_tools,
execute_tool, _handle_manage_tasks, shutdown。
process_input 继承自 ReActAgent，已在 test_react.py 中测试。
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.core.agent.types import AgentState
from src.core.ark.simple_engine import SimpleARKEngine
from src.core.ark.tasks import TaskStatus


@pytest.fixture
def engine():
    """创建 SimpleARKEngine 测试实例"""
    eng = SimpleARKEngine(config={"enable_llm": True})
    eng.mcp_client = AsyncMock()
    eng.mcp_client.sessions = {}
    eng.mcp_client.execute_tool = AsyncMock(return_value="executed")
    eng.mcp_client.get_server_tools = AsyncMock(return_value={})
    eng.mcp_client.list_tools = AsyncMock(return_value=[])
    eng.llm_manager = AsyncMock()
    eng.state = AgentState.READY
    eng.available_tools = {}
    return eng


@pytest.mark.asyncio
async def test_initialize_success(engine):
    """测试 initialize 成功路径"""
    with patch("src.core.ark.simple_engine.connect_servers") as mock_connect:
        with patch.object(engine, "_discover_tools") as mock_discover:
            engine.state = AgentState.INITIALIZING
            result = await engine.initialize(mcp_servers=None)

            assert result is True
            assert engine.state == AgentState.READY
            mock_connect.assert_called_once()
            mock_discover.assert_called_once()


@pytest.mark.asyncio
async def test_initialize_failure(engine):
    """测试 initialize 失败路径"""
    with patch("src.core.ark.simple_engine.connect_servers") as mock_connect:
        mock_connect.side_effect = Exception("Connection failed")
        engine.state = AgentState.INITIALIZING
        result = await engine.initialize(mcp_servers=None)

        assert result is False
        assert engine.state == AgentState.ERROR


@pytest.mark.asyncio
async def test_discover_tools_success(engine):
    """测试工具发现成功"""
    with patch("src.core.ark.simple_engine.discover_tools") as mock_discover:
        mock_discover.return_value = {"tool1": {"name": "tool1"}}
        await engine._discover_tools()

        assert "tool1" in engine.available_tools
        assert "manage_tasks" in engine.available_tools
        assert engine.available_tools["manage_tasks"]["name"] == "manage_tasks"


@pytest.mark.asyncio
async def test_discover_tools_failure(engine):
    """测试工具发现失败"""
    with patch("src.core.ark.simple_engine.discover_tools") as mock_discover:
        mock_discover.side_effect = Exception("Discovery failed")
        await engine._discover_tools()

        assert engine.available_tools == {}


@pytest.mark.asyncio
async def test_execute_tool_manage_tasks(engine):
    """测试执行内置任务管理工具"""
    with patch.object(engine, "_handle_manage_tasks") as mock_handle:
        mock_handle.return_value = "Task managed"
        result = await engine.execute_tool("manage_tasks", {"action": "add"})

        assert result == "Task managed"
        mock_handle.assert_called_once_with({"action": "add"})


@pytest.mark.asyncio
async def test_execute_tool_mcp(engine):
    """测试执行 MCP 外部工具"""
    result = await engine.execute_tool("weather", {"city": "NYC"})

    assert result == "executed"
    engine.mcp_client.execute_tool.assert_called_once_with("weather", {"city": "NYC"})


@pytest.mark.asyncio
async def test_shutdown(engine):
    """测试 shutdown 清理资源"""
    with patch("src.core.ark.simple_engine.close_mcp_client") as mock_close:
        with patch.object(engine, "_discover_tools"):
            engine.state = AgentState.READY
            await engine.shutdown()
            mock_close.assert_called_once_with(engine.mcp_client)


def test_handle_manage_tasks(engine):
    """测试内置任务管理"""
    engine.todo_list = []
    result = engine._handle_manage_tasks({"action": "add", "description": "Test task"})

    assert "added" in result
    assert len(engine.todo_list) == 1
    assert engine.todo_list[0].description == "Test task"
    assert engine.todo_list[0].status == TaskStatus.PENDING
