"""
Test module for Jarvis Agent functionality.

This module contains comprehensive tests for the JarvisAgent class and its configuration.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import dataclasses

from src.jarvis_agent import JarvisConfig, JarvisAgent
from src.core.config.server import SimpleMCPServerConfig


class TestJarvisConfig:
    """Test cases for JarvisConfig class."""
    
    def test_config_creation_minimal(self):
        """Test configuration creation with minimal parameters."""
        config = JarvisConfig()
        
        assert config.name == "Jarvis"
        assert config.version == "1.0.0"
        assert config.log_level == "INFO"
        assert config.max_conversation_history == 100
        assert config.enable_tool_chaining is True
        assert config.confidence_threshold == 0.7
        assert config.mcp_servers == []
    
    def test_config_creation_full(self):
        """Test configuration creation with all parameters."""
        # Create a mock MCP server config
        mock_server = SimpleMCPServerConfig(name="test_server", command=["test_command"])
        
        config = JarvisConfig(
            name="TestJarvis",
            version="2.0.0",
            log_level="DEBUG",
            max_conversation_history=50,
            enable_tool_chaining=False,
            confidence_threshold=0.8,
            mcp_servers=[mock_server]
        )
        
        assert config.name == "TestJarvis"
        assert config.version == "2.0.0"
        assert config.log_level == "DEBUG"
        assert config.max_conversation_history == 50
        assert config.enable_tool_chaining is False
        assert config.confidence_threshold == 0.8
        assert len(config.mcp_servers) == 1
        assert config.mcp_servers[0].name == "test_server"
    
    def test_config_to_dict(self):
        """Test configuration conversion to dictionary."""
        config = JarvisConfig(
            name="TestJarvis",
            log_level="DEBUG",
            max_conversation_history=50
        )
        
        config_dict = dataclasses.asdict(config)
        
        assert config_dict["name"] == "TestJarvis"
        assert config_dict["log_level"] == "DEBUG"
        assert config_dict["max_conversation_history"] == 50
    
    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_data = {
            "name": "TestJarvis",
            "version": "2.0.0",
            "log_level": "DEBUG",
            "max_conversation_history": 50,
            "enable_tool_chaining": False,
            "confidence_threshold": 0.8,
            "mcp_servers": []
        }
        
        config = JarvisConfig(**config_data)
        
        assert config.name == "TestJarvis"
        assert config.version == "2.0.0"
        assert config.log_level == "DEBUG"
        assert config.max_conversation_history == 50
        assert config.enable_tool_chaining is False
        assert config.confidence_threshold == 0.8


class TestJarvisAgent:
    """Test cases for JarvisAgent class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return JarvisConfig(
            name="TestJarvis",
            version="2.0.0",
            log_level="DEBUG",
            max_conversation_history=50
        )
    
    @pytest.fixture
    def agent(self, config):
        """Create a test agent."""
        return JarvisAgent(config)
    
    def test_agent_creation_with_default_config(self):
        """Test agent creation with default configuration."""
        agent = JarvisAgent()
        assert agent.config.name == "Jarvis"
        assert agent.config.version == "1.0.0"
        assert agent.config.log_level == "INFO"
        assert agent.config.max_conversation_history == 100
        assert not agent.is_running
        assert not agent.is_initialized

    def test_agent_creation_with_custom_config(self, config):
        """Test agent creation with custom configuration."""
        agent = JarvisAgent(config)
        assert agent.config.name == "TestJarvis"
        assert agent.config.version == "2.0.0"
        assert agent.config.log_level == "DEBUG"
        assert agent.config.max_conversation_history == 50
        assert not agent.is_running
        assert not agent.is_initialized

    def test_add_mcp_server(self, agent):
        """Test adding MCP server configuration."""
        server_config = SimpleMCPServerConfig(name="test_server", command=["test_command"])
        agent.add_mcp_server(server_config)
        assert "test_server" in [s.name for s in agent.config.mcp_servers]

    def test_remove_mcp_server(self, agent):
        """Test removing MCP server configuration."""
        server_config = SimpleMCPServerConfig(name="test_server", command=["test_command"])
        agent.add_mcp_server(server_config)
        result = agent.remove_mcp_server("test_server")
        assert result is True
        assert "test_server" not in [s.name for s in agent.config.mcp_servers]

    def test_remove_nonexistent_mcp_server(self, agent):
        """Test removing non-existent MCP server."""
        result = agent.remove_mcp_server("nonexistent")
        assert result is False

    def test_add_callbacks(self, agent):
        """Test adding various callbacks."""
        def dummy_callback():
            pass
        
        agent.add_startup_callback(dummy_callback)
        agent.add_shutdown_callback(dummy_callback)
        agent.add_message_callback(dummy_callback)
        
        assert dummy_callback in agent.on_startup_callbacks
        assert dummy_callback in agent.on_shutdown_callbacks
        assert dummy_callback in agent.on_message_callbacks

    def test_get_status(self, agent):
        """Test getting agent status."""
        status = agent.get_status()
        assert "agent" in status
        assert "configuration" in status
        assert "ark_engine" in status
        assert status["agent"]["name"] == "TestJarvis"
        assert status["agent"]["version"] == "2.0.0"
        assert "is_running" in status["agent"]

    def test_agent_context_manager_methods(self, agent):
        """Test agent context manager protocol."""
        assert hasattr(agent, '__aenter__')
        assert hasattr(agent, '__aexit__')

    @pytest.mark.asyncio
    async def test_initialize_agent(self, agent):
        """Test agent initialization."""
        with patch.object(agent.ark_engine, 'initialize', return_value=True):
            result = await agent.initialize()
            assert result is True
            assert agent.is_initialized

    @pytest.mark.asyncio
    async def test_start_agent(self, agent):
        """Test agent start."""
        with patch.object(agent, 'initialize', return_value=True):
            await agent.start()
            assert agent.is_running

    @pytest.mark.asyncio
    async def test_stop_agent(self, agent):
        """Test agent stop."""
        agent.is_running = True
        with patch.object(agent.ark_engine, 'shutdown', return_value=True):
            await agent.stop()
            assert not agent.is_running

    @pytest.mark.asyncio
    async def test_process_message(self, agent):
        """Test message processing."""
        agent.is_running = True
        agent.is_initialized = True
        
        with patch.object(agent.ark_engine, 'process_input', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = "Test response"
            
            response = await agent.process_message("Hello")
            
            assert response == "Test response"
            mock_process.assert_called_once_with("Hello")

    @pytest.mark.asyncio
    async def test_health_check(self, agent):
        """Test health check functionality."""
        with patch.object(agent.ark_engine.mcp_client, 'discover_tools', return_value=["tool1", "tool2"]):
            health = await agent.health_check()
            assert "overall" in health
            assert "components" in health
            assert "timestamp" in health
    
    @pytest.mark.asyncio
    async def test_start_conversation(self, agent):
        """Test starting a conversation."""
        with patch.object(agent.ark_engine.context_manager, 'reset_session', return_value=None):
            result = await agent.start_conversation("user123")
            assert "testjarvis" in result.lower() or "ai assistant" in result.lower()
    
    @pytest.mark.asyncio
    async def test_end_conversation(self, agent):
        """Test ending a conversation."""
        with patch.object(agent.ark_engine.context_manager, 'get_session_stats', return_value={"turn_count": 5, "duration_minutes": 10.5}):
            result = await agent.end_conversation("user123")
            assert "5 messages" in result
            assert "10.5 minutes" in result
    
    @pytest.mark.asyncio
    async def test_get_conversation_export(self, agent):
        """Test conversation export."""
        mock_history = [{"role": "user", "content": "Hello"}]
        with patch.object(agent.ark_engine.context_manager, 'export_conversation', return_value='{"conversation": [{"user": "Hello", "agent": "Hi there!"}]}'):
            # Test JSON export
            json_export = agent.get_conversation_export("json")
            assert "Hello" in json_export
            
            # Test text export
            text_export = agent.get_conversation_export("text")
            assert "Hello" in text_export
    
    @pytest.mark.asyncio
    async def test_get_available_tools(self, agent):
        """Test getting available tools."""
        mock_tools = {"echo": {"description": "Echo tool"}}
        with patch.object(agent.ark_engine, 'available_tools', mock_tools):
            tools = agent.get_available_tools()
            assert "echo" in tools
    
    @pytest.mark.asyncio
    async def test_get_tool_usage_stats(self, agent):
        """Test getting tool usage statistics."""
        stats = agent.get_tool_usage_stats()
        assert isinstance(stats, dict)
    
    def test_set_get_user_preference(self, agent):
        """Test setting and getting user preferences."""
        agent.set_user_preference("theme", "dark")
        assert agent.get_user_preference("theme") == "dark"
        assert agent.get_user_preference("nonexistent", "default") == "default"
    
    @pytest.mark.asyncio
    async def test_context_manager(self, agent):
        """Test agent as async context manager."""
        with patch.object(agent, 'initialize', return_value=True), \
             patch.object(agent, 'start'), \
             patch.object(agent, 'stop'):
            
            async with agent as ctx_agent:
                assert ctx_agent is agent
    
    def test_timestamp_generation(self, agent):
        """Test timestamp generation."""
        timestamp = agent._get_timestamp()
        assert isinstance(timestamp, str)
        assert len(timestamp) > 0
    
    def test_repr(self, agent):
        """Test string representation."""
        repr_str = repr(agent)
        assert "JarvisAgent" in repr_str
        assert agent.config.name in repr_str


class TestJarvisAgentAdvanced:
    """Advanced test cases for JarvisAgent."""
    
    @pytest.fixture
    def agent_with_servers(self):
        """Create agent with MCP servers."""
        server_config = SimpleMCPServerConfig(name="test_server", command=["test"])
        config = JarvisConfig(mcp_servers=[server_config])
        return JarvisAgent(config)
    
    @pytest.mark.asyncio
    async def test_initialization_with_servers(self, agent_with_servers):
        """Test initialization with MCP servers."""
        with patch.object(agent_with_servers.ark_engine.mcp_client, 'connect_to_server', return_value=True) as mock_connect, \
             patch.object(agent_with_servers.ark_engine, '_discover_tools'):
            await agent_with_servers.initialize()
            mock_connect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_message_with_tool_usage(self):
        """Test message processing with tool usage tracking."""
        agent = JarvisAgent()
        agent.is_running = True  # Set agent to running state
        
        with patch.object(agent.ark_engine, 'process_input', return_value="response"), \
             patch.object(agent.ark_engine.context_manager, 'add_exchange'), \
             patch.object(agent.ark_engine, 'tool_usage_stats', {"total_messages": 1}):
            
            result = await agent.process_message("test message", "user123")
            assert result == "response"
            stats = agent.get_tool_usage_stats()
            assert stats.get("total_messages", 0) >= 1
    
    @pytest.mark.asyncio
    async def test_health_check_with_errors(self):
        """Test health check with component errors."""
        agent = JarvisAgent()
        agent.is_initialized = True
        agent.is_running = True
        
        with patch.object(agent.ark_engine.mcp_client, 'list_tools', side_effect=Exception("MCP error")):
            health = await agent.health_check()
            assert health["overall"] == "degraded"
            assert health["components"]["mcp_client"]["status"] == "unhealthy"
    
    def test_callback_management(self):
        """Test callback management."""
        agent = JarvisAgent()
        
        startup_callback = Mock()
        shutdown_callback = Mock()
        message_callback = Mock()
        
        agent.add_startup_callback(startup_callback)
        agent.add_shutdown_callback(shutdown_callback)
        agent.add_message_callback(message_callback)
        
        assert startup_callback in agent.on_startup_callbacks
        assert shutdown_callback in agent.on_shutdown_callbacks
        assert message_callback in agent.on_message_callbacks
    
    @pytest.mark.asyncio
    async def test_startup_callbacks_execution(self):
        """Test startup callbacks are executed."""
        agent = JarvisAgent()
        callback = AsyncMock()
        agent.add_startup_callback(callback)
        
        with patch.object(agent.ark_engine, 'initialize', return_value=True), \
             patch.object(agent.ark_engine.mcp_client, 'start'):
            await agent.start()
            callback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown_callbacks_execution(self):
        """Test shutdown callbacks are executed."""
        agent = JarvisAgent()
        callback = AsyncMock()
        agent.add_shutdown_callback(callback)
        
        with patch.object(agent.ark_engine.mcp_client, 'stop'), \
             patch.object(agent.ark_engine, 'shutdown'):
            await agent.stop()
            callback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_message_callbacks_execution(self):
        """Test message callbacks are executed."""
        agent = JarvisAgent()
        agent.is_running = True  # Set agent as running
        callback = Mock()
        agent.add_message_callback(callback)
        
        with patch.object(agent.ark_engine, 'process_input', return_value="response"), \
             patch.object(agent.ark_engine.context_manager, 'add_exchange'):
            
            await agent.process_message("test", "user123")
            callback.assert_called()
    
    @pytest.mark.asyncio
    async def test_initialization_failure(self):
        """Test initialization failure when ARK engine fails to initialize."""
        agent = JarvisAgent()
        
        # Mock ARK engine initialization to fail
        with patch.object(agent.ark_engine, 'initialize', return_value=False):
            # Test initialization
            result = await agent.initialize()
            
            # Verify initialization failed
            assert result is False
            assert not agent.is_initialized
    
    @pytest.mark.asyncio
    async def test_startup_callbacks_execution(self):
        """Test startup callbacks execution during initialization."""
        agent = JarvisAgent()
        
        # Add sync and async callbacks
        sync_callback = Mock()
        async_callback = AsyncMock()
        
        agent.add_startup_callback(sync_callback)
        agent.add_startup_callback(async_callback)
        
        # Mock ARK engine initialization to succeed
        with patch.object(agent.ark_engine, 'initialize', return_value=True):
            # Test initialization
            result = await agent.initialize()
            
            # Verify initialization succeeded
            assert result is True
            assert agent.is_initialized
            
            # Verify callbacks were called
            sync_callback.assert_called_once_with(agent)
            async_callback.assert_called_once_with(agent)
    
    @pytest.mark.asyncio
    async def test_startup_callbacks_exception_handling(self):
        """Test startup callbacks exception handling."""
        agent = JarvisAgent()
        
        # Add a callback that raises an exception
        def failing_callback(agent):
            raise Exception("Callback failed")
        
        agent.add_startup_callback(failing_callback)
        
        # Mock ARK engine initialization to succeed
        with patch.object(agent.ark_engine, 'initialize', return_value=True):
            # Test initialization - should still succeed despite callback failure
            result = await agent.initialize()
            
            # Verify initialization succeeded despite callback failure
            assert result is True
            assert agent.is_initialized
    
    @pytest.mark.asyncio
    async def test_initialization_exception_handling(self):
        """Test initialization exception handling."""
        agent = JarvisAgent()
        
        # Mock ARK engine initialization to raise an exception
        with patch.object(agent.ark_engine, 'initialize', side_effect=Exception("Initialization error")):
            # Test initialization
            result = await agent.initialize()
            
            # Verify initialization failed due to exception
            assert result is False
            assert not agent.is_initialized
    
    @pytest.mark.asyncio
    async def test_shutdown_callbacks_execution(self):
        """Test shutdown callbacks execution during stop."""
        agent = JarvisAgent()
        agent.is_running = True
        
        # Add sync and async callbacks
        sync_callback = Mock()
        async_callback = AsyncMock()
        
        agent.add_shutdown_callback(sync_callback)
        agent.add_shutdown_callback(async_callback)
        
        # Mock ARK engine shutdown
        with patch.object(agent.ark_engine, 'shutdown', new_callable=AsyncMock):
            # Test stop
            await agent.stop()
            
            # Verify callbacks were called
            sync_callback.assert_called_once_with(agent)
            async_callback.assert_called_once_with(agent)
    
    @pytest.mark.asyncio
    async def test_shutdown_callbacks_exception_handling(self):
        """Test shutdown callbacks exception handling."""
        agent = JarvisAgent()
        agent.is_running = True
        
        # Add a callback that raises an exception
        def failing_callback(agent):
            raise Exception("Shutdown callback failed")
        
        agent.add_shutdown_callback(failing_callback)
        
        # Mock ARK engine shutdown
        with patch.object(agent.ark_engine, 'shutdown', new_callable=AsyncMock):
            # Test stop - should complete despite callback failure
            await agent.stop()
            
            # Verify agent stopped despite callback failure
            assert not agent.is_running
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager functionality."""
        agent = JarvisAgent()
        
        # Mock ARK engine methods
        with patch.object(agent.ark_engine, 'initialize', return_value=True), \
             patch.object(agent.ark_engine, 'shutdown', new_callable=AsyncMock):
            
            # Test async context manager
            async with agent:
                # Verify agent is started
                assert agent.is_running
                assert agent.is_initialized
            
            # Verify agent is stopped after context exit
            assert not agent.is_running


if __name__ == "__main__":
    pytest.main([__file__, "-v"])