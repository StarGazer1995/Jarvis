"""
Tests for the MCP Client module.

This module contains comprehensive tests for the ARK MCP client,
including server configuration, connection management, and tool execution.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from src.core.mcp_client import ARKMCPClient
from src.core.server_config import SimpleMCPServerConfig


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
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = Mock()
            mock_process.stdin = Mock()
            mock_process.stdout = Mock()
            mock_process.stderr = Mock()
            mock_subprocess.return_value = mock_process
            
            result = await mcp_client._connect_to_server(mock_config)
            
            # Should attempt to create subprocess
            mock_subprocess.assert_called_once()
            # Result depends on implementation details
    
    @pytest.mark.asyncio
    async def test_connect_to_server_failure(self, mcp_client, mock_config):
        """Test failed server connection."""
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_subprocess.side_effect = Exception("Connection failed")
            
            result = await mcp_client._connect_to_server(mock_config)
            
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])