"""
Test module for Jarvis Agent functionality.

This module contains comprehensive tests for the JarvisAgent class and its configuration.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import dataclasses

from src.jarvis_agent import JarvisConfig, JarvisAgent
from src.core.server_config import SimpleMCPServerConfig


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
        health = await agent.health_check()
        
        assert "overall" in health
        assert "timestamp" in health
        assert "components" in health


if __name__ == "__main__":
    pytest.main([__file__, "-v"])