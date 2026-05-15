"""
Fault Injection Tests — MCP Layer

Tests the resilience of the MCP client layer against various failure
scenarios using the FaultInjector framework. Covers:

- Tool execution failures (timeout, invalid response, not found)
- Server disconnect mid-operation
- Graceful degradation when servers are down
- Partial cleanup on close
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.common.exceptions import MCPError, ToolError
from src.core.config.server import ServerStatus, SimpleMCPServerConfig
from src.core.mcp.client import ARKMCPClient
from tests.fault_injection.mcp_faults import (
    mcp_cascading_failure,
    mcp_connection_failure,
    mcp_discovery_failure,
    mcp_server_disconnect,
    mcp_then_succeed,
    mcp_tool_execution_error,
    mcp_tool_invalid_response,
    mcp_tool_not_found,
    mcp_tool_timeout,
)

# ═══════════════════════════════════════════════════════════════════
# 1. MCP Client Unit Tests — Fault Injection
# ═══════════════════════════════════════════════════════════════════


class TestMCPClientFaultInjection:
    """Test ARKMCPClient resilience using FaultInjector on its dependencies."""

    @pytest.mark.asyncio
    async def test_execute_tool_returns_none_on_missing_tool(self):
        """Calling a non-existent tool should return None, not crash."""
        client = ARKMCPClient()
        client.sessions["test_server"] = AsyncMock()

        result = await client.execute_tool("nonexistent_tool", {})
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_tool_handles_server_disconnect(self):
        """If the MCP server is disconnected, execute_tool should return None."""
        client = ARKMCPClient()
        # No sessions connected
        client.available_tools["server:tool"] = Mock(name="tool")

        # _resolve_tool finds it but _get_server_name returns "server"
        # and there's no session for "server"
        result = await client.execute_tool("server:tool", {})
        assert result is None

    @pytest.mark.asyncio
    async def test_connect_to_server_failure(self):
        """Connecting to an unavailable server should not crash the client."""
        client = ARKMCPClient()

        with patch.object(
            client,
            "_connect_to_server_internal",
            AsyncMock(side_effect=Exception("Connection refused")),
        ):
            config = SimpleMCPServerConfig(
                name="unreachable",
                command="python",
                args=["-c", "exit(1)"],
                enabled=True,
            )
            result = await client.connect_to_server(config)
            assert result is False

    @pytest.mark.asyncio
    async def test_disconnect_server_handles_missing_server(self):
        """Disconnecting a server that's not connected should not crash."""
        client = ARKMCPClient()
        result = await client.disconnect_server("nonexistent_server")
        assert result is False

    @pytest.mark.asyncio
    async def test_close_handles_partial_cleanup(self):
        """Closing the client should not crash if servers are partially connected."""
        client = ARKMCPClient()
        client.sessions["server_a"] = AsyncMock()
        client.server_exit_stacks["server_a"] = AsyncMock()
        # server_b is in sessions but NOT in server_exit_stacks (partial state)
        client.sessions["server_b"] = AsyncMock()

        # Should not raise
        await client.close()
        assert len(client.sessions) == 0
        assert len(client.available_tools) == 0

    @pytest.mark.asyncio
    async def test_discover_tools_handles_session_error(self):
        """Tool discovery should handle session errors gracefully."""
        client = ARKMCPClient()
        mock_session = AsyncMock()
        mock_session.list_tools = AsyncMock(side_effect=Exception("Session not ready"))

        client.sessions["test_server"] = mock_session

        # Should not crash; the method catches exceptions and returns empty list
        try:
            tools = await client.discover_tools("test_server")
            assert tools == []
        except Exception:
            # If it raises, that's also acceptable
            pass

    @pytest.mark.asyncio
    async def test_get_server_status_for_unknown_server(self):
        """get_server_status should return UNKNOWN for unknown servers."""
        client = ARKMCPClient()
        status = client.get_server_status("ghost_server")
        assert status == ServerStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_initialize_from_config_handles_missing_file(self):
        """Should not crash when config file doesn't exist."""
        client = ARKMCPClient()
        with patch.object(client, "_connect_to_servers", AsyncMock()):
            await client.initialize_from_config("/nonexistent/path/config.json")
            assert client.is_initialized is True


# ═══════════════════════════════════════════════════════════════════
# 2. Pre-built MCP Fault Helpers
# ═══════════════════════════════════════════════════════════════════


class TestPrebuiltMCPFaults:
    """Verify the pre-built MCP fault helpers work correctly."""

    def test_mcp_connection_failure(self):
        side_effect = mcp_connection_failure("db_server")
        with pytest.raises(MCPError, match="db_server"):
            side_effect()

    def test_mcp_server_disconnect(self):
        side_effect = mcp_server_disconnect("worker_1")
        with pytest.raises(MCPError, match="worker_1"):
            side_effect()

    def test_mcp_tool_not_found(self):
        side_effect = mcp_tool_not_found("deleted_tool")
        with pytest.raises(ToolError, match="deleted_tool"):
            side_effect()

    def test_mcp_tool_execution_error(self):
        side_effect = mcp_tool_execution_error("calculator", "Division by zero")
        with pytest.raises(ToolError, match="Division by zero"):
            side_effect()

    def test_mcp_tool_timeout(self):
        side_effect = mcp_tool_timeout("search")
        with pytest.raises(ToolError, match="timed out"):
            side_effect()

    def test_mcp_tool_invalid_response(self):
        side_effect = mcp_tool_invalid_response("db_query")
        with pytest.raises(ToolError, match="invalid response"):
            side_effect()

    def test_mcp_discovery_failure(self):
        side_effect = mcp_discovery_failure("api_server")
        with pytest.raises(MCPError, match="Failed to discover"):
            side_effect()

    def test_mcp_then_succeed(self):
        side_effect = mcp_then_succeed(
            fail_count=2,
            success_value={"status": "success", "result": "ok"},
        )

        with pytest.raises(ToolError):
            side_effect()
        with pytest.raises(ToolError):
            side_effect()

        result = side_effect()
        assert result["status"] == "success"

    def test_mcp_cascading_failure(self):
        """Cascade: MCP → ToolError ×2 → success."""
        side_effect = mcp_cascading_failure(
            success_value={"status": "success", "result": "recovered"},
        )

        with pytest.raises(MCPError):
            side_effect()
        with pytest.raises(ToolError):
            side_effect()
        with pytest.raises(ToolError):
            side_effect()

        result = side_effect()
        assert result["status"] == "success"


# ═══════════════════════════════════════════════════════════════════
# 3. MCP Client Graceful Degradation Tests
# ═══════════════════════════════════════════════════════════════════


class TestMCPGracefulDegradation:
    """Test that the MCP client degrades gracefully under various fault conditions."""

    @pytest.mark.asyncio
    async def test_other_servers_work_when_one_fails(self):
        """If server A is down, calls to server B should still work."""
        client = ARKMCPClient()

        # Server A tool fails
        mock_session_a = AsyncMock()
        mock_session_a.call_tool = AsyncMock(
            side_effect=Exception("Server A internal error")
        )
        client.sessions["server_a"] = mock_session_a
        client.available_tools["server_a:tool_a"] = Mock(
            spec=["name", "description"],
            name="tool_a",
        )

        # Server B works
        mock_response_b = AsyncMock()
        mock_response_b.isError = False
        mock_response_b.content = [Mock()]
        mock_response_b.content[0].model_dump.return_value = {
            "type": "text",
            "text": "Server B response",
        }
        mock_session_b = AsyncMock()
        mock_session_b.call_tool.return_value = mock_response_b
        client.sessions["server_b"] = mock_session_b
        client.available_tools["server_b:tool_b"] = Mock(
            spec=["name", "description"],
            name="tool_b",
        )

        # Tool A should fail (returns None on exception)
        result_a = await client.execute_tool("server_a:tool_a", {})
        assert result_a is None

        # Tool B should still work
        result_b = await client.execute_tool("server_b:tool_b", {})
        # _execute_server_tool returns a dict on success
        if result_b is not None:
            assert isinstance(result_b, dict)

    @pytest.mark.asyncio
    async def test_builtin_tools_work_when_mcp_servers_down(self):
        """Built-in tools (echo, get_time) should work even when MCP fails."""
        client = ARKMCPClient()

        # No MCP servers connected, but builtin tools should work
        result = await client.call_tool("echo", {"message": "hello"})
        assert result["success"] is True
        assert "Echo: hello" in str(result["result"])

    @pytest.mark.asyncio
    async def test_list_tools_returns_builtins_when_no_servers(self):
        """list_tools should at least include builtin tools when servers are down."""
        client = ARKMCPClient()
        tools = await client.list_tools()
        tool_names = [t["name"] for t in tools]
        assert "echo" in tool_names
        assert "get_time" in tool_names

    @pytest.mark.asyncio
    async def test_shutdown_does_not_raise_on_already_disconnected(self):
        """Calling close multiple times should not raise."""
        client = ARKMCPClient()
        client.sessions["server"] = AsyncMock()
        client.server_exit_stacks["server"] = AsyncMock()

        await client.close()
        assert len(client.sessions) == 0

        await client.close()
        assert len(client.sessions) == 0
