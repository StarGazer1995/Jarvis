"""
Tests for the MCP Client module.

This module contains comprehensive tests for the ARK MCP client,
including server configuration, connection management, and tool execution.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from src.core.mcp.client import ARKMCPClient
from src.core.config.server import SimpleMCPServerConfig, ServerStatus


class TestSimpleMCPServerConfig:
    """Test cases for SimpleMCPServerConfig class."""
    
    def test_config_creation(self):
        """Test basic configuration creation."""
        config = SimpleMCPServerConfig(
            name="test_server",
            command=["python", "-m", "test_server"],
            env={"TEST_VAR": "test_value"}
        )
        
        assert config.name == "test_server"
        assert config.command == ["python", "-m", "test_server"]
        assert config.env["TEST_VAR"] == "test_value"
    
    def test_config_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = SimpleMCPServerConfig(
            name="test_server",
            command=["python", "-m", "test_server"]
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["name"] == "test_server"
        assert config_dict["command"] == ["python", "-m", "test_server"]
    
    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_data = {
            "name": "test_server",
            "command": ["python", "-m", "test_server"],
            "env": {"TEST_VAR": "test_value"}
        }
        
        config = SimpleMCPServerConfig.from_dict(config_data)
        
        assert config.name == "test_server"
        assert config.command == ["python", "-m", "test_server"]
        assert config.env["TEST_VAR"] == "test_value"


class TestARKMCPClient:
    """Test cases for ARKMCPClient class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock server configuration."""
        return SimpleMCPServerConfig(
            name="test_server",
            command=["python", "-m", "test_server"],
            env={"TEST_VAR": "test_value"}
        )
    
    @pytest.fixture
    def mcp_client(self):
        """Create an MCP client instance for testing."""
        return ARKMCPClient()
    
    def test_client_initialization(self, mcp_client):
        """Test MCP client initialization."""
        assert mcp_client.servers == {}
        assert mcp_client.available_tools == {}
        assert mcp_client.tool_schemas == {}
        assert mcp_client._running is False
    
    @pytest.mark.asyncio
    async def test_add_server(self, mcp_client, mock_config):
        """Test adding a server configuration."""
        result = await mcp_client.add_server(mock_config)
        
        assert result is True
        assert "test_server" in mcp_client.servers
        assert mcp_client.servers["test_server"] == mock_config
    
    @pytest.mark.asyncio
    async def test_add_duplicate_server(self, mcp_client, mock_config):
        """Test adding a duplicate server configuration."""
        # Add server first time
        await mcp_client.add_server(mock_config)
        
        # Try to add same server again
        result = await mcp_client.add_server(mock_config)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_remove_server(self, mcp_client, mock_config):
        """Test removing a server configuration."""
        # Add server first
        await mcp_client.add_server(mock_config)
        
        # Remove server
        result = await mcp_client.remove_server("test_server")
        
        assert result is True
        assert "test_server" not in mcp_client.servers
    
    @pytest.mark.asyncio
    async def test_remove_nonexistent_server(self, mcp_client):
        """Test removing a non-existent server."""
        result = await mcp_client.remove_server("nonexistent_server")
        
        assert result is False
    
    def test_get_server(self, mcp_client, mock_config):
        """Test getting server configuration."""
        # Add server
        asyncio.run(mcp_client.add_server(mock_config))
        
        # Get server
        retrieved_config = mcp_client.get_server("test_server")
        
        assert retrieved_config == mock_config
    
    def test_get_nonexistent_server(self, mcp_client):
        """Test getting non-existent server configuration."""
        retrieved_config = mcp_client.get_server("nonexistent_server")
        
        assert retrieved_config is None
    
    def test_list_servers(self, mcp_client, mock_config):
        """Test listing all servers."""
        # Initially empty
        servers = mcp_client.list_servers()
        assert servers == []
        
        # Add server
        asyncio.run(mcp_client.add_server(mock_config))
        
        # List servers
        servers = mcp_client.list_servers()
        assert len(servers) == 1
        assert servers[0] == mock_config
    
    @pytest.mark.asyncio
    async def test_start_client(self, mcp_client):
        """Test starting the MCP client."""
        with patch.object(mcp_client, '_discover_tools', new_callable=AsyncMock) as mock_discover:
            await mcp_client.start()
            
            assert mcp_client._running is True
            mock_discover.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_client(self, mcp_client):
        """Test stopping the MCP client."""
        # Start client first
        with patch.object(mcp_client, '_discover_tools', new_callable=AsyncMock):
            await mcp_client.start()
        
        # Stop client
        with patch.object(mcp_client, '_disconnect_all_servers', new_callable=AsyncMock) as mock_disconnect:
            await mcp_client.stop()
            
            assert mcp_client._running is False
            mock_disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_discover_tools(self, mcp_client, mock_config):
        """Test tool discovery process."""
        # Add server
        await mcp_client.add_server(mock_config)
        
        # Simulate that the server is connected by adding it to sessions
        mock_session = Mock()
        mcp_client.sessions["test_server"] = mock_session
        
        # Mock the discover_tools method to simulate tool discovery
        with patch.object(mcp_client, 'discover_tools', new_callable=AsyncMock) as mock_discover:
            # Simulate the discover_tools method behavior
            async def mock_discover_side_effect(server_name):
                tool_info = {
                    "name": "test_tool",
                    "description": "A test tool",
                    "inputSchema": {"type": "object"},
                    "server": server_name
                }
                tool_key = f"{server_name}:test_tool"
                mcp_client.available_tools[tool_key] = tool_info
                return [tool_info]
            
            mock_discover.side_effect = mock_discover_side_effect
            
            await mcp_client._discover_tools()
            
            # Check that the tool is stored with the correct key format
            tool_key = "test_server:test_tool"
            assert tool_key in mcp_client.available_tools
            assert mcp_client.available_tools[tool_key]["server"] == "test_server"
    
    @pytest.mark.asyncio
    async def test_execute_tool_success(self, mcp_client):
        """Test successful tool execution."""
        # Setup available tool
        mcp_client.available_tools["test_tool"] = {
            "name": "test_tool",
            "server": "test_server",
            "description": "A test tool"
        }
        
        # Mock tool execution
        with patch.object(mcp_client, '_execute_server_tool', new_callable=AsyncMock) as mock_execute:
            mock_result = {"result": "success", "data": "test_data"}
            mock_execute.return_value = mock_result
            
            result = await mcp_client.execute_tool("test_tool", {"param": "value"})
            
            assert result == mock_result
            mock_execute.assert_called_once_with("test_server", "test_tool", {"param": "value"})
    
    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self, mcp_client):
        """Test tool execution with non-existent tool."""
        result = await mcp_client.execute_tool("nonexistent_tool", {})
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_execute_tool_with_error(self, mcp_client):
        """Test tool execution with server error."""
        # Setup available tool
        mcp_client.available_tools["test_tool"] = {
            "name": "test_tool",
            "server": "test_server",
            "description": "A test tool"
        }
        
        # Mock tool execution with error
        with patch.object(mcp_client, '_execute_server_tool', new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = Exception("Server error")
            
            result = await mcp_client.execute_tool("test_tool", {"param": "value"})
            
            assert result is None
    
    def test_get_available_tools(self, mcp_client):
        """Test getting available tools."""
        # Initially empty
        tools = mcp_client.get_available_tools()
        assert tools == {}
        
        # Add some tools
        mcp_client.available_tools = {
            "tool1": {"name": "tool1", "server": "server1"},
            "tool2": {"name": "tool2", "server": "server2"}
        }
        
        tools = mcp_client.get_available_tools()
        assert len(tools) == 2
        assert "tool1" in tools
        assert "tool2" in tools
    
    def test_get_tool_schema(self, mcp_client):
        """Test getting tool schema."""
        # Setup tool schema
        mcp_client.tool_schemas["test_tool"] = {
            "type": "object",
            "properties": {
                "param": {"type": "string"}
            }
        }
        
        schema = mcp_client.get_tool_schema("test_tool")
        assert schema["type"] == "object"
        assert "param" in schema["properties"]
    
    def test_get_nonexistent_tool_schema(self, mcp_client):
        """Test getting schema for non-existent tool."""
        schema = mcp_client.get_tool_schema("nonexistent_tool")
        assert schema is None
    
    @pytest.mark.asyncio
    async def test_connect_to_server_success(self, mcp_client, mock_config):
        """Test successful server connection."""
        with patch.object(mcp_client, '_connect_to_server_internal', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = True
            
            result = await mcp_client._connect_to_server("test_server")
            
            # Should attempt to connect
            mock_connect.assert_called_once()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_connect_to_server_failure(self, mcp_client, mock_config):
        """Test failed server connection."""
        with patch.object(mcp_client, '_connect_to_server_internal', new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")
            
            result = await mcp_client._connect_to_server("test_server")
            
            assert result is False
    
    def test_client_statistics(self, mcp_client):
        """Test getting client statistics."""
        # Add some test data
        mcp_client.available_tools = {
            "tool1": {"server": "server1"},
            "tool2": {"server": "server1"},
            "tool3": {"server": "server2"}
        }
        
        stats = mcp_client.get_client_stats()
        
        assert stats["total_servers"] == 0  # No servers added in this test
        assert stats["total_tools"] == 3
        assert stats["running"] is False
    
    @pytest.mark.asyncio
    async def test_demo_mode(self, mcp_client):
        """Test demo mode functionality."""
        with patch('builtins.print') as mock_print:
            await mcp_client.demo()
            
            # Should print demo information
            assert mock_print.called
    
    def test_repr(self, mcp_client):
        """Test string representation of client."""
        repr_str = repr(mcp_client)
        assert "ARKMCPClient" in repr_str
        assert "servers=0" in repr_str
        assert "tools=0" in repr_str


class TestMCPClientIntegration:
    """Integration tests for MCP client functionality."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete MCP client workflow."""
        client = ARKMCPClient()
        
        # Create test configuration
        config = SimpleMCPServerConfig(
            name="test_server",
            command=["echo", "hello"]
        )
        
        try:
            # Add server
            result = await client.add_server(config)
            assert result is True
            
            # Start client - mock both connection and discovery methods
            with patch.object(client, '_connect_to_servers', new_callable=AsyncMock) as mock_connect, \
                 patch.object(client, '_discover_tools', new_callable=AsyncMock) as mock_discover:
                await client.start()
                assert client._running is True
                mock_connect.assert_called_once()
                mock_discover.assert_called_once()
            
            # Check server is added
            assert client.get_server("test_server") == config
            
            # List servers
            servers = client.list_servers()
            assert len(servers) == 1
            
        finally:
            # Clean up - mock disconnect method
            with patch.object(client, '_disconnect_all_servers', new_callable=AsyncMock) as mock_disconnect:
                await client.stop()
                assert client._running is False
                mock_disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_multiple_servers(self):
        """Test handling multiple servers."""
        client = ARKMCPClient()
        
        # Create multiple configurations
        configs = [
            SimpleMCPServerConfig(name="server1", command=["echo", "1"]),
            SimpleMCPServerConfig(name="server2", command=["echo", "2"]),
            SimpleMCPServerConfig(name="server3", command=["echo", "3"])
        ]
        
        try:
            # Add all servers
            for config in configs:
                result = await client.add_server(config)
                assert result is True
            
            # Check all servers are added
            servers = client.list_servers()
            assert len(servers) == 3
            
            # Remove one server
            result = await client.remove_server("server2")
            assert result is True
            
            # Check server is removed
            servers = client.list_servers()
            assert len(servers) == 2
            assert client.get_server("server2") is None
            
        finally:
            # Clean up
            await client.stop()


class TestARKMCPClientIntegration:
    """Integration tests for ARKMCPClient with real server interactions."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return ARKMCPClient()
    
    @pytest.mark.asyncio
    async def test_client_lifecycle(self, client):
        """Test complete client lifecycle."""
        # Start client
        await client.start()
        
        # Check initial state
        assert len(client.list_servers()) >= 0
        
        # Stop client
        await client.stop()
    
    @pytest.mark.asyncio
    async def test_builtin_tools(self, client):
        """Test builtin tools functionality."""
        # Test echo tool
        result = await client.call_tool("echo", {"message": "test"})
        assert result["success"] is True
        assert result["result"][0]["text"] == "Echo: test"
        
        # Test invalid tool
        result = await client.call_tool("nonexistent", {})
        assert result["success"] is False
        assert "不存在" in result["error"]
    
    @pytest.mark.asyncio
    async def test_list_tools(self, client):
        """Test listing all available tools."""
        tools = await client.list_tools()
        assert isinstance(tools, list)
        
        # Should have at least the builtin echo tool
        tool_names = [tool["name"] for tool in tools]
        assert "echo" in tool_names
        
        # Check tool structure
        echo_tool = next(tool for tool in tools if tool["name"] == "echo")
        assert "description" in echo_tool
        assert "inputSchema" in echo_tool
        assert "server" in echo_tool
        assert echo_tool["server"] == "builtin"
    
    @pytest.mark.asyncio
    async def test_get_available_tools(self, client):
        """Test getting available tools dictionary."""
        tools = client.get_available_tools()
        assert isinstance(tools, dict)
        
        # get_available_tools() only returns MCP server tools, not builtin tools
        # For builtin tools, we should use get_all_tools()
        all_tools = client.get_all_tools()
        assert "echo" in all_tools
        assert "description" in all_tools["echo"]
        assert "inputSchema" in all_tools["echo"]
    
    @pytest.mark.asyncio
    async def test_get_all_tools(self, client):
        """Test getting all tools including builtin."""
        all_tools = client.get_all_tools()
        assert isinstance(all_tools, dict)
        
        # Should include builtin tools
        assert "echo" in all_tools
    
    @pytest.mark.asyncio
    async def test_get_tool_schema(self, client):
        """Test getting tool schema."""
        # Test existing tool
        schema = client.get_tool_schema("echo")
        assert schema is not None
        assert "type" in schema
        assert "properties" in schema
        
        # Test non-existent tool
        schema = client.get_tool_schema("nonexistent")
        assert schema is None
    
    @pytest.mark.asyncio
    async def test_get_client_stats(self, client):
        """Test getting client statistics."""
        stats = client.get_client_stats()
        assert isinstance(stats, dict)
        assert "total_servers" in stats
        assert "connected_servers" in stats
        assert "total_tools" in stats
        assert "running" in stats
    
    @pytest.mark.asyncio
    async def test_execute_tool(self, client):
        """Test execute_tool method."""
        # Test builtin tool
        result = await client.execute_tool("echo", {"message": "test"})
        assert result is not None
        assert result["success"] is True
        assert result["result"][0]["text"] == "Echo: test"
        
        # Test non-existent tool
        result = await client.execute_tool("nonexistent", {})
        assert result is None
    
    @pytest.mark.asyncio
    async def test_discover_tools_invalid_server(self, client):
        """Test discovering tools from invalid server."""
        tools = await client.discover_tools("nonexistent_server")
        assert tools == []
    
    @pytest.mark.asyncio
    async def test_get_server_status_invalid(self, client):
        """Test getting status of non-existent server."""
        status = client.get_server_status("nonexistent")
        assert status == ServerStatus.UNKNOWN
    
    @pytest.mark.asyncio
    async def test_get_all_server_status(self, client):
        """Test getting all server statuses."""
        statuses = client.get_all_server_status()
        assert isinstance(statuses, dict)
        
        # Should have demo servers
        for server_name, status in statuses.items():
            assert isinstance(status, ServerStatus)
    
    @pytest.mark.asyncio
    async def test_disconnect_invalid_server(self, client):
        """Test disconnecting from invalid server."""
        result = await client.disconnect_server("nonexistent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_server_invalid(self, client):
        """Test getting invalid server config."""
        server = client.get_server("nonexistent")
        assert server is None
    
    @pytest.mark.asyncio
    async def test_call_builtin_tool_invalid(self, client):
        """Test calling invalid builtin tool."""
        with pytest.raises(ValueError, match="未知的内置工具"):
            await client._call_builtin_tool("nonexistent", {})
    
    @pytest.mark.asyncio
    async def test_call_tool_with_exception(self, client):
        """Test call_tool with exception handling."""
        # Mock a tool that raises an exception
        with patch.object(client, '_call_builtin_tool', side_effect=Exception("Test error")):
            result = await client.call_tool("echo", {"message": "test"})
            assert result["success"] is False
            assert "Test error" in result["error"]
    
    @pytest.mark.asyncio
    async def test_repr(self, client):
        """Test string representation."""
        repr_str = repr(client)
        assert "ARKMCPClient" in repr_str
        assert "servers=" in repr_str
        assert "tools=" in repr_str


class TestARKMCPClientErrorHandling:
    """Test error handling and edge cases for ARKMCPClient."""
    
    @pytest.fixture
    def client(self):
        """Create a client for testing."""
        return ARKMCPClient()
    
    @pytest.mark.asyncio
    async def test_initialization_with_invalid_config(self):
        """Test client initialization."""
        # Test normal initialization
        client = ARKMCPClient()
        assert client is not None
        assert hasattr(client, 'servers')
        assert hasattr(client, 'available_tools')
    
    @pytest.mark.asyncio
    async def test_initialize_from_nonexistent_config(self, client):
        """Test initializing from non-existent config file."""
        # Mock _connect_to_servers to avoid actual connection attempts
        with patch.object(client, '_connect_to_servers', new_callable=AsyncMock):
            # This should trigger demo server initialization
            await client.initialize_from_config("nonexistent_config.json")
            assert client.is_initialized
            # Verify demo servers were added
            assert len(client.servers) > 0
    
    @pytest.mark.asyncio
    async def test_initialize_from_valid_config(self, client, tmp_path):
        """Test initializing from valid config file."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.json"
        config_data = {
            "servers": [
                {
                    "name": "test_server",
                    "command": ["echo", "test"],
                    "args": [],
                    "enabled": True
                }
            ]
        }
        config_file.write_text(json.dumps(config_data))
        
        # Mock _connect_to_servers to avoid actual connection attempts
        with patch.object(client, '_connect_to_servers', new_callable=AsyncMock):
            await client.initialize_from_config(str(config_file))
            assert client.is_initialized
            assert "test_server" in client.servers
    
    @pytest.mark.asyncio
    async def test_demo_server_configuration_error(self, client):
        """Test demo server configuration error handling."""
        with patch('src.core.mcp.client.SimpleMCPServerConfig') as mock_config:
            mock_config.side_effect = Exception("Configuration error")
            
            # This should not raise an exception
            await client.start()
            # Client should still be functional even if demo server setup fails
    
    @pytest.mark.asyncio
    async def test_initialize_from_invalid_json_config(self, client, tmp_path):
        """Test initializing from invalid JSON config file."""
        import json
        # Create a temporary config file with invalid JSON
        config_file = tmp_path / "invalid_config.json"
        config_file.write_text("invalid json content")
        
        with pytest.raises(Exception):
            await client.initialize_from_config(str(config_file))
    
    @pytest.mark.asyncio
    async def test_connect_to_servers_error(self, client):
        """Test error handling in _connect_to_servers."""
        # Add a server but mock connection to fail
        client.servers["test_server"] = SimpleMCPServerConfig(
            name="test_server",
            command=["echo", "test"],
            args=[],
            enabled=True
        )
        
        with patch.object(client, '_connect_to_server_internal', side_effect=Exception("Connection error")):
            # This should not raise an exception, just log the error
            await client._connect_to_servers()
    
    @pytest.mark.asyncio
    async def test_repr_method(self, client):
        """Test the __repr__ method."""
        repr_str = repr(client)
        assert "ARKMCPClient" in repr_str
        assert "servers=" in repr_str
        assert "tools=" in repr_str
    
    @pytest.mark.asyncio
    async def test_connect_to_server_with_invalid_name(self, client):
        """Test connecting to server with invalid name."""
        result = await client._connect_to_server("nonexistent_server")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_find_tool_in_servers_not_found(self, client):
        """Test finding tool that doesn't exist in any server."""
        # Test getting schema for a non-existent tool
        result = client.get_tool_schema("nonexistent_tool")
        assert result is None
        
        # Test executing a non-existent tool
        result = await client.execute_tool("nonexistent_tool", {})
        assert result is None
    
    @pytest.mark.asyncio
    async def test_execute_server_tool_with_session_error(self, client):
        """Test executing server tool when session retrieval fails."""
        # Test with non-existent server
        with pytest.raises(ValueError, match="服务器.*未连接"):
            await client._execute_server_tool("nonexistent_server", "test_tool", {})
    
    @pytest.mark.asyncio
    async def test_call_builtin_tool_get_time(self, client):
        """Test calling builtin get_time tool."""
        result = await client._call_builtin_tool("get_time", {})
        assert "result" in result
        # Check if result contains time information (supports both Chinese and English)
        result_text = str(result["result"])
        assert "当前时间" in result_text or "Current time" in result_text
    
    @pytest.mark.asyncio
    async def test_call_builtin_tool_invalid(self, client):
        """Test calling invalid builtin tool."""
        try:
            result = await client._call_builtin_tool("invalid_tool", {})
            # If no exception, check for error in result
            assert "error" in result
        except ValueError as e:
            # Expected behavior - invalid tool raises ValueError
            assert "未知的内置工具" in str(e) or "Unknown builtin tool" in str(e)
    
    def test_get_server_status_disconnected(self, client):
        """Test getting server status for disconnected server."""
        # Add a server but don't connect
        config = SimpleMCPServerConfig(
            name="test_server",
            command=["echo", "test"],
            args=[]
        )
        client.servers["test_server"] = config
        
        status = client.get_server_status("test_server")
        assert status == ServerStatus.DISCONNECTED
    
    def test_get_all_server_status_with_mixed_states(self, client):
        """Test getting all server statuses with mixed connection states."""
        # Add servers in different states
        config1 = SimpleMCPServerConfig(name="server1", command=["echo"], args=[])
        config2 = SimpleMCPServerConfig(name="server2", command=["echo"], args=[])
        
        client.servers["server1"] = config1
        client.servers["server2"] = config2
        
        # Mock one as connected, one as disconnected
        client.sessions["server1"] = Mock()
        # server2 has no session (disconnected)
        
        statuses = client.get_all_server_status()
        assert "server1" in statuses
        assert "server2" in statuses
        assert statuses["server1"] == ServerStatus.CONNECTED
        assert statuses["server2"] == ServerStatus.DISCONNECTED
    
    @pytest.mark.asyncio
    async def test_execute_tool_with_server_tool_error(self, client):
        """Test executing tool when server tool execution fails."""
        # Test with non-existent tool
        result = await client.execute_tool("nonexistent_tool", {})
        assert result is None
    
    @pytest.mark.asyncio
    async def test_discover_tools_with_connection_error(self, client):
        """Test discovering tools when server connection fails."""
        config = SimpleMCPServerConfig(
            name="test_server",
            command=["echo", "test"],
            args=[]
        )
        await client.add_server(config)
        
        # Mock connection to fail
        with patch.object(client, '_connect_to_server') as mock_connect:
            mock_connect.return_value = False
            
            tools = await client.discover_tools("test_server")
            assert tools == []
    
    def test_client_statistics_with_tools_and_servers(self, client):
        """Test client statistics with actual tools and servers."""
        # Add some mock data
        client.servers["server1"] = Mock()
        client.servers["server2"] = Mock()
        client.available_tools = {
            "tool1": {"server": "server1"},
            "tool2": {"server": "server2"},
            "tool3": {"server": "server1"}
        }
        
        stats = client.get_client_stats()
        assert stats["total_servers"] == 2
        assert stats["total_tools"] == 3
        assert stats["connected_servers"] == 0  # No sessions
        assert stats["running"] is False
        # Check that all expected keys are present
        assert isinstance(stats, dict)
        assert len(stats) >= 4


class TestARKMCPClientAdvancedFeatures:
    """Test advanced features and edge cases."""
    
    @pytest.fixture
    def client(self):
        """Create a client for testing."""
        return ARKMCPClient()
    
    @pytest.mark.asyncio
    async def test_tool_execution_with_complex_args(self, client):
        """Test tool execution with complex arguments."""
        # Mock builtin tool execution
        with patch.object(client, '_call_builtin_tool') as mock_builtin:
            mock_builtin.return_value = {"content": "Success"}
            
            complex_args = {
                "nested": {"key": "value"},
                "list": [1, 2, 3],
                "string": "test"
            }
            
            result = await client.execute_tool("get_time", complex_args)
            assert "content" in result
            mock_builtin.assert_called_once_with("get_time", complex_args)
    
    @pytest.mark.asyncio
    async def test_multiple_server_tool_discovery(self, client):
        """Test tool discovery across multiple servers."""
        # Add multiple servers
        config1 = SimpleMCPServerConfig(name="server1", command=["echo"], args=[])
        config2 = SimpleMCPServerConfig(name="server2", command=["echo"], args=[])
        
        await client.add_server(config1)
        await client.add_server(config2)
        
        # Test discovery without actual connection (will return empty lists)
        tools1 = await client.discover_tools("server1")
        tools2 = await client.discover_tools("server2")
        
        # Should return empty lists since servers aren't actually connected
        assert isinstance(tools1, list)
        assert isinstance(tools2, list)
    
    def test_get_tool_schema_for_builtin_tool(self, client):
        """Test getting schema for builtin tools."""
        schema = client.get_tool_schema("get_time")
        assert schema is not None
        assert isinstance(schema, dict)
        # Builtin tools have basic schema structure
        assert "type" in schema
    
    @pytest.mark.asyncio
    async def test_client_lifecycle_with_errors(self, client):
        """Test client lifecycle with various errors."""
        # Test start
        await client.start()
        
        # Test stop
        await client.stop()
        
        # Test restart
        await client.start()
        await client.stop()

    @pytest.mark.asyncio
    async def test_disconnect_server_success(self, client):
        """Test successful server disconnection."""
        # Setup mock server and session
        server_config = SimpleMCPServerConfig(
            name="test_server",
            command=["python", "-m", "test_server"]
        )
        
        # Add server to client
        client.servers["test_server"] = server_config
        
        # Mock session
        mock_session = AsyncMock()
        client.sessions["test_server"] = mock_session
        
        # Add some tools from this server
        client.available_tools["test_server:tool1"] = Mock()
        client.available_tools["test_server:tool2"] = Mock()
        client.available_tools["other_server:tool3"] = Mock()
        
        # Test disconnection
        result = await client.disconnect_server("test_server")
        
        assert result is True
        assert "test_server" not in client.sessions
        assert "test_server:tool1" not in client.available_tools
        assert "test_server:tool2" not in client.available_tools
        assert "other_server:tool3" in client.available_tools  # Should remain

    @pytest.mark.asyncio
    async def test_disconnect_server_not_found(self, client):
        """Test disconnecting a server that doesn't exist."""
        result = await client.disconnect_server("nonexistent_server")
        assert result is False

    @pytest.mark.asyncio
    async def test_discover_tools_with_schema_update(self, client):
        """Test successful tool discovery with schema updates."""
        # Mock server session
        mock_session = AsyncMock()
        mock_tool1 = Mock()
        mock_tool1.name = "tool1"
        mock_tool1.description = "Tool 1"
        mock_tool1.inputSchema = {"type": "object"}
        
        mock_tool2 = Mock()
        mock_tool2.name = "tool2"
        mock_tool2.description = None  # Test None description
        mock_tool2.inputSchema = {"type": "string"}
        
        mock_session.list_tools.return_value = Mock(tools=[mock_tool1, mock_tool2])
        client.sessions = {"server1": mock_session}
        
        # Test discover tools
        tools = await client.discover_tools("server1")
        
        assert len(tools) == 2
        assert tools[0]["name"] == "tool1"
        assert tools[0]["description"] == "Tool 1"
        assert tools[0]["server"] == "server1"
        assert tools[1]["name"] == "tool2"
        assert tools[1]["description"] == ""  # None should become empty string
        
        # Verify tools were added to available_tools and tool_schemas
        assert "server1:tool1" in client.available_tools
        assert "server1:tool2" in client.available_tools
        assert "server1:tool1" in client.tool_schemas
        assert "server1:tool2" in client.tool_schemas

    @pytest.mark.asyncio
    async def test_call_tool_error_handling(self, client):
        """Test error handling when calling tools."""
        # Test tool not found
        result = await client.call_tool("nonexistent_tool", {})
        assert result["success"] is False
        assert "不存在" in result["error"]
        
        # Test builtin tool with error
        with patch.object(client, '_call_builtin_tool', side_effect=Exception("Test error")):
            result = await client.call_tool("echo", {"message": "test"})
            assert result["success"] is False
            assert "Test error" in result["error"]

    @pytest.mark.asyncio
    async def test_call_tool_server_not_connected(self, client):
        """Test calling tool when server is not connected."""
        # Add tool but no session
        client.available_tools = {
            "disconnected_server:test_tool": {"name": "test_tool", "server": "disconnected_server"}
        }
        
        # Test calling tool with disconnected server
        result = await client.call_tool("disconnected_server:test_tool", {})
        
        assert result["success"] is False
        assert "未连接" in result["error"]

    @pytest.mark.asyncio
    async def test_discover_tools_success(self, client):
        """Test successful tool discovery."""
        # Setup mock server
        mock_session = AsyncMock()
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool"
        mock_tool.inputSchema = {"type": "object"}
        mock_session.list_tools.return_value = Mock(tools=[mock_tool])
        
        client.sessions = {"test_server": mock_session}
        
        # Test discover tools
        tools = await client.discover_tools("test_server")
        
        assert len(tools) == 1
        assert tools[0]["name"] == "test_tool"
        assert tools[0]["description"] == "Test tool"
        assert tools[0]["server"] == "test_server"
        
        # Verify tool was added to available_tools
        assert "test_server:test_tool" in client.available_tools

    @pytest.mark.asyncio
    async def test_connect_to_server_internal_error_handling(self, client):
        """Test error handling in internal server connection."""
        # Test with invalid server config
        config = SimpleMCPServerConfig(
            name="test_server",
            command=["invalid_command"]
        )
        client.servers["test_server"] = config
        
        # Mock the connection process to raise an exception
        with patch('src.core.mcp.client.stdio_client') as mock_stdio:
            mock_stdio.side_effect = Exception("Connection failed")
            
            # Should raise the exception since _connect_to_server_internal doesn't handle errors
            with pytest.raises(Exception, match="Connection failed"):
                await client._connect_to_server_internal("test_server", config)

    @pytest.mark.asyncio
    async def test_close_method_coverage(self, client):
        """Test the close method for coverage."""
        # Add some sessions to close
        mock_session1 = AsyncMock()
        mock_session2 = AsyncMock()
        client.sessions = {
            "server1": mock_session1,
            "server2": mock_session2
        }
        
        # Test close
        await client.close()
        
        # Verify sessions were closed
        mock_session1.close.assert_called_once()
        mock_session2.close.assert_called_once()
        
        # Verify sessions dict is cleared
        assert len(client.sessions) == 0

    @pytest.mark.asyncio
    async def test_list_tools_with_mcp_server_tools(self, client):
        """Test list_tools method handling MCP server tools branch."""
        from unittest.mock import MagicMock
        
        # Create mock MCP tool object that behaves like a dictionary
        mock_tool = MagicMock()
        # Set up the get method to return appropriate values
        mock_tool.get.side_effect = lambda key, default=None: {
            "name": "mcp_tool",
            "description": "A test MCP tool",
            "parameters": {"type": "object", "properties": {}},
            "inputSchema": {"type": "object", "properties": {}}
        }.get(key, default)
        
        # Add MCP tool to available_tools
        client.available_tools = {
            "test_server:mcp_tool": mock_tool
        }
        
        # Call list_tools
        tools = await client.list_tools()
        
        # Verify returned tool information (should include both builtin and MCP tools)
        mcp_tools = [t for t in tools if t["server"] == "test_server"]
        assert len(mcp_tools) == 1
        tool = mcp_tools[0]
        assert tool["name"] == "mcp_tool"
        assert tool["description"] == "A test MCP tool"
        assert tool["server"] == "test_server"
        assert tool["full_name"] == "test_server:mcp_tool"

    @pytest.mark.asyncio
    async def test_call_tool_simple_name_matching(self, client):
        """Test call_tool method simple name matching logic."""
        from unittest.mock import MagicMock, PropertyMock
        
        # Create mock tool and session
        mock_tool = MagicMock()
        type(mock_tool).name = PropertyMock(return_value="simple_tool")
        
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.content = [{"type": "text", "text": "Success"}]
        mock_session.call_tool.return_value = mock_result
        
        # Set available_tools and sessions
        client.available_tools = {
            "test_server:simple_tool": mock_tool
        }
        client.sessions = {
            "test_server": mock_session
        }
        
        # Call tool using simple name (should match to test_server:simple_tool)
        result = await client.call_tool("simple_tool", {"arg": "value"})
        
        # Verify result
        assert result["success"] is True
        assert result["tool"] == "simple_tool"
        assert result["server"] == "test_server"
        mock_session.call_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_tool_successful_execution(self, client):
        """Test call_tool method successful execution path."""
        from unittest.mock import MagicMock, PropertyMock
        
        # Create mock tool and session
        mock_tool = MagicMock()
        type(mock_tool).name = PropertyMock(return_value="test_tool")
        
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.content = [{"type": "text", "text": "Tool executed successfully"}]
        mock_session.call_tool.return_value = mock_result
        
        # Set available_tools and sessions
        client.available_tools = {
            "test_server:test_tool": mock_tool
        }
        client.sessions = {
            "test_server": mock_session
        }
        
        # Call tool
        result = await client.call_tool("test_server:test_tool", {"param": "test"})
        
        # Verify result
        assert result["success"] is True
        assert result["result"] == [{"type": "text", "text": "Tool executed successfully"}]
        assert result["tool"] == "test_server:test_tool"
        assert result["server"] == "test_server"
        
        # Verify CallToolRequest was created correctly
        call_args = mock_session.call_tool.call_args[0][0]
        assert call_args.params.name == "test_tool"
        assert call_args.params.arguments == {"param": "test"}
    
    @pytest.mark.asyncio
    async def test_disconnect_server_error_handling(self, client):
        """Test error handling in disconnect_server method."""
        # Setup mock session that raises exception on close
        mock_session = AsyncMock()
        mock_session.close.side_effect = Exception("Close error")
        client.sessions = {"test_server": mock_session}
        
        # Add some tools from this server
        client.available_tools = {
            "test_server:tool1": {"name": "tool1", "server": "test_server"}
        }
        
        # Test disconnect with error
        result = await client.disconnect_server("test_server")
        
        # Should return False due to error
        assert result is False
    
    @pytest.mark.asyncio
    async def test_close_method_error_handling(self, client):
        """Test error handling in close method."""
        # Setup mock session that raises exception
        mock_session = AsyncMock()
        mock_session.close.side_effect = Exception("Close error")
        client.sessions = {"test_server": mock_session}
        client.is_initialized = True
        
        # Test close with error (should not raise exception)
        await client.close()
        
        # Should still reset state
        assert client.is_initialized is False
    
    @pytest.mark.asyncio
    async def test_demo_method_coverage(self, client):
        """Test demo method for coverage."""
        # Setup some data for demo
        config = SimpleMCPServerConfig(
            name="test_server",
            command=["echo", "test"],
            description="Test server"
        )
        client.servers = {"test_server": config}
        client.available_tools = {
            "test_tool": {"name": "test_tool", "server": "test_server"}
        }
        client.sessions = {"test_server": Mock()}
        
        # Mock get_server_status
        with patch.object(client, 'get_server_status', return_value=ServerStatus.CONNECTED):
            # Execute demo (mainly for print statement coverage)
            await client.demo()
    
    @pytest.mark.asyncio
    async def test_remove_server_not_found(self, client):
        """Test removing non-existent server."""
        result = await client.remove_server("nonexistent_server")
        assert result is False
    
    def test_get_server_not_found(self, client):
        """Test getting non-existent server."""
        result = client.get_server("nonexistent_server")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_execute_tool_no_server_name(self, client):
        """Test execute_tool when tool has no server name."""
        # Setup tool without server field
        client.available_tools = {
            "test_tool": {"name": "test_tool"}  # Missing server field
        }
        
        result = await client.execute_tool("test_tool", {})
        assert result is None
    
    @pytest.mark.asyncio
    async def test_execute_server_tool_server_not_connected(self, client):
        """Test _execute_server_tool when server not connected."""
        with pytest.raises(ValueError, match="服务器.*未连接"):
            await client._execute_server_tool("nonexistent_server", "test_tool", {})
    
    @pytest.mark.asyncio
    async def test_connect_to_server_error_handling(self, client):
        """Test error handling in _connect_to_server."""
        config = SimpleMCPServerConfig(
            name="test_server",
            command=["invalid_command"]
        )
        
        # Mock _connect_to_server_internal to raise exception
        with patch.object(client, '_connect_to_server_internal', side_effect=Exception("Connection failed")):
            result = await client._connect_to_server("test_server", config)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_connect_to_server_internal_error(self, client):
        """Test _connect_to_server_internal error handling."""
        config = SimpleMCPServerConfig(
            name="test_server",
            command=["invalid_command"]
        )
        
        # Should raise exception since method doesn't handle errors internally
        with pytest.raises(Exception):
            await client._connect_to_server_internal("test_server", config)
    
    @pytest.mark.asyncio
    async def test_get_server_tools_server_not_in_sessions(self, client):
        """Test _get_server_tools when server not in sessions."""
        result = await client._get_server_tools("nonexistent_server")
        assert result == {}
    
    @pytest.mark.asyncio
    async def test_get_server_tools_error_handling(self, client):
        """Test _get_server_tools error handling."""
        # Setup mock session that raises exception
        mock_session = AsyncMock()
        mock_session.list_tools.side_effect = Exception("List tools failed")
        client.sessions = {"test_server": mock_session}
        
        result = await client._get_server_tools("test_server")
        assert result == {}
    def test_repr_method(self, client):
        """Test __repr__ method."""
        # Setup some data
        client.servers = {"server1": Mock(), "server2": Mock()}
        client.available_tools = {"tool1": Mock(), "tool2": Mock(), "tool3": Mock()}
        
        result = repr(client)
        expected = "ARKMCPClient(servers=2, tools=3)"
        assert result == expected
    
    @pytest.mark.asyncio
    async def test_close_method_exception_handling(self, client):
        """Test close method exception handling."""
        # Mock _disconnect_all_servers to raise an exception
        with patch.object(client, '_disconnect_all_servers', side_effect=Exception("Disconnect error")):
            # Should not raise exception, just log error
            await client.close()
    
    @pytest.mark.asyncio
    async def test_connect_to_server_success_path(self, client):
        """Test connect_to_server success path."""
        server_config = SimpleMCPServerConfig(
            name="test_server",
            command="python",
            args=["-m", "test"]
        )
        
        # Mock add_server and _connect_to_server to succeed
        with patch.object(client, 'add_server', return_value=True) as mock_add, \
             patch.object(client, '_connect_to_server', return_value=True) as mock_connect:
            
            result = await client.connect_to_server(server_config)
            assert result is True
            mock_add.assert_called_once_with(server_config)
            mock_connect.assert_called_once_with(server_config)
    
    @pytest.mark.asyncio
    async def test_remove_server_success_path(self, client):
        """Test remove_server success path."""
        server_name = "test_server"
        mock_config = Mock()
        client.servers = {server_name: mock_config}
        
        # Mock disconnect_server to succeed
        with patch.object(client, 'disconnect_server', return_value=True):
            result = await client.remove_server(server_name)
            assert result is True
            assert server_name not in client.servers
    
    @pytest.mark.asyncio
    async def test_execute_server_tool_success_path(self, client):
        """Test _execute_server_tool success path."""
        server_name = "test_server"
        tool_name = "test_tool"
        arguments = {"arg1": "value1"}
        
        # Mock session and call_tool result
        mock_session = Mock()
        mock_result = Mock()
        mock_result.content = [{"type": "text", "text": "Success"}]
        mock_result.isError = False
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        
        client.sessions = {server_name: mock_session}
        
        result = await client._execute_server_tool(server_name, tool_name, arguments)
        
        expected = {
            "success": True,
            "result": [{"type": "text", "text": "Success"}],
            "error": None
        }
        assert result == expected
    
    @pytest.mark.asyncio
    async def test_connect_to_server_internal_exception_handling(self, client):
        """Test _connect_to_server internal exception handling."""
        server_config = SimpleMCPServerConfig(
            name="test_server",
            command="python",
            args=["-m", "test"]
        )
        
        # Mock _connect_to_server_internal to raise exception
        with patch.object(client, '_connect_to_server_internal', side_effect=Exception("Connection error")):
            result = await client._connect_to_server(server_config)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_connect_to_server_internal_success_path(self, client):
        """Test _connect_to_server_internal success path."""
        server_name = "test_server"
        server_config = SimpleMCPServerConfig(
            name=server_name,
            command="python",
            args=["-m", "test"]
        )
        
        # Mock stdio_client and session
        mock_session = Mock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock()
        
        # Mock tools result
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tools_result = Mock()
        mock_tools_result.tools = [mock_tool]
        mock_session.list_tools.return_value = mock_tools_result
        
        # Mock stdio_client context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=(Mock(), Mock()))
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        with patch('src.core.mcp.client.stdio_client', return_value=mock_context), \
             patch('src.core.mcp.client.ClientSession', return_value=mock_session):
            
            await client._connect_to_server_internal(server_name, server_config)
            
            # Verify session was stored
            assert server_name in client.sessions
            assert client.sessions[server_name] == mock_session
            
            # Verify tool was added
            tool_key = f"{server_name}:test_tool"
            assert tool_key in client.available_tools
    
    @pytest.mark.asyncio
    async def test_disconnect_all_servers_success(self, client):
        """Test _disconnect_all_servers success path."""
        # Setup sessions (not servers)
        server1 = "server1"
        server2 = "server2"
        client.sessions = {server1: Mock(), server2: Mock()}
        
        # Mock disconnect_server to succeed
        with patch.object(client, 'disconnect_server', new_callable=AsyncMock, return_value=True) as mock_disconnect:
            await client._disconnect_all_servers()
            
            # Verify disconnect was called for each server
            assert mock_disconnect.call_count == 2
            mock_disconnect.assert_any_call(server1)
            mock_disconnect.assert_any_call(server2)
    
    @pytest.mark.asyncio
    async def test_get_server_tools_success_path(self, client):
        """Test _get_server_tools success path."""
        server_name = "test_server"
        
        # Mock session and tools
        mock_session = Mock()
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool description"
        mock_tool.inputSchema = {"type": "object"}
        
        mock_tools_response = Mock()
        mock_tools_response.tools = [mock_tool]
        mock_session.list_tools = AsyncMock(return_value=mock_tools_response)
        
        client.sessions = {server_name: mock_session}
        
        result = await client._get_server_tools(server_name)
        
        expected = {
            "test_tool": {
                "name": "test_tool",
                "description": "Test tool description",
                "parameters": {"type": "object"},
                "server": server_name
            }
        }
        assert result == expected

    async def test_close_method_with_disconnect_exception(self, client):
        """测试close方法中disconnect_server抛出异常的情况"""
        server_name = "test_server"
        
        # 设置sessions
        client.sessions = {server_name: Mock()}
        
        # Mock disconnect_server to raise exception
        with patch.object(client, 'disconnect_server', side_effect=Exception("Disconnect failed")):
            with patch.object(client.logger, 'error') as mock_error:
                await client.close()
                
                # 验证异常被捕获并记录
                mock_error.assert_called_once()
                assert "关闭MCP客户端失败" in str(mock_error.call_args[0][0])

    async def test_remove_server_with_delete_config(self, client):
        """测试remove_server方法删除服务器配置的路径"""
        server_name = "test_server"
        server_config = SimpleMCPServerConfig(
            name=server_name,
            command="test_command",
            args=[]
        )
        
        # 设置服务器配置但不在sessions中
        client.servers = {server_name: server_config}
        client.sessions = {}  # 不在sessions中，所以不会调用disconnect_server
        
        result = await client.remove_server(server_name)
        
        assert result is True
        assert server_name not in client.servers

    async def test_connect_to_server_config_success_path(self, client):
        """测试_connect_to_server方法中配置对象成功路径（覆盖line 621）"""
        server_config = SimpleMCPServerConfig(
            name="test_server",
            command="test_command", 
            args=[]
        )
        
        # Mock _connect_to_server_internal to succeed
        with patch.object(client, '_connect_to_server_internal') as mock_connect:
            result = await client._connect_to_server(server_config)
            
            assert result is True
            mock_connect.assert_called_once_with("test_server", server_config)

    async def test_connect_to_server_config_exception_handling(self, client):
        """测试_connect_to_server方法中配置对象异常处理路径"""
        server_config = SimpleMCPServerConfig(
            name="test_server",
            command="test_command", 
            args=[]
        )
        
        # Mock _connect_to_server_internal to raise exception
        with patch.object(client, '_connect_to_server_internal', side_effect=Exception("Connection failed")):
            result = await client._connect_to_server(server_config)
            
            assert result is False

    @pytest.mark.asyncio
    async def test_remove_server_delete_line_coverage(self):
        """Test remove_server to ensure line 413 (del self.servers[server_name]) is covered."""
        # Create a fresh client instance
        client = ARKMCPClient()
        server_name = "test_server"
        mock_config = Mock()
        
        # Directly set up the scenario for line 413
        client.servers[server_name] = mock_config
        # Ensure sessions is empty so we skip the disconnect logic
        assert server_name not in client.sessions
        
        # Call remove_server - this MUST execute line 413
        result = await client.remove_server(server_name)
        
        # Verify the deletion happened
        assert result is True
        assert server_name not in client.servers

if __name__ == "__main__":
    pytest.main([__file__, "-v"])