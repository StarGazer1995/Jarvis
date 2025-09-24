"""
Pytest configuration and shared fixtures for the Jarvis project tests.

This module provides common test fixtures, configuration, and utilities
that are shared across all test modules in the project.
"""

import pytest
import asyncio
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, Optional

from src.jarvis_agent import JarvisConfig, JarvisAgent
from src.core.server_config import SimpleMCPServerConfig
from src.core.mcp_client import ARKMCPClient
from src.core.context_manager import ConversationContext
from src.core.intent_engine import ARKIntentEngine
from src.core.ark_engine import ARKEngine
from src.core.tool_registry import ARKToolRegistry
from src.core.server_config import ARKServerConfigManager


@pytest.fixture(scope="session")
def event_loop():
    """
    Create an event loop for the entire test session.
    
    This fixture ensures that async tests can run properly
    and share the same event loop across the session.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """
    Create a temporary directory for test files.
    
    Returns:
        str: Path to the temporary directory
    """
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_config_file(temp_dir):
    """
    Create a temporary configuration file for testing.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        str: Path to the temporary config file
    """
    config_path = os.path.join(temp_dir, "test_config.json")
    config_data = {
        "name": "TestJarvis",
        "version": "1.0.0",
        "debug": True,
        "max_conversation_length": 100,
        "log_level": "DEBUG"
    }
    
    import json
    with open(config_path, 'w') as f:
        json.dump(config_data, f)
    
    return config_path


@pytest.fixture
def temp_config_dir():
    """
    Create a temporary directory for config files.
    
    Returns:
        Path: Path to the temporary config directory
    """
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_jarvis_config():
    """
    Create a sample JarvisConfig for testing.
    
    Returns:
        JarvisConfig: A configured JarvisConfig instance
    """
    return JarvisConfig(
        name="TestJarvis",
        version="1.0.0",
        debug=True,
        max_conversation_length=50,
        log_level="DEBUG",
        config_file=None
    )


@pytest.fixture
def sample_mcp_server_config():
    """
    Create a sample SimpleMCPServerConfig for testing.
    
    Returns:
        SimpleMCPServerConfig: A configured SimpleMCPServerConfig instance
    """
    return SimpleMCPServerConfig(
        name="test_server",
        command=["python", "-m", "test_server"],
        env={"TEST_ENV": "true"}
    )


@pytest.fixture
def mock_mcp_client():
    """
    Create a mock MCP client for testing.
    
    Returns:
        Mock: A mock ARKMCPClient instance
    """
    client = Mock(spec=ARKMCPClient)
    client.startup = AsyncMock(return_value=True)
    client.shutdown = AsyncMock(return_value=True)
    client.add_server = AsyncMock(return_value=True)
    client.remove_server = AsyncMock(return_value=True)
    client.discover_tools = AsyncMock(return_value=[])
    client.execute_tool = AsyncMock(return_value={
        "status": "success",
        "result": "Mock tool result"
    })
    client.get_server_status = Mock(return_value={
        "connected": True,
        "tools_count": 0
    })
    return client


@pytest.fixture
def mock_context_manager():
    """
    Create a mock conversation context manager for testing.
    
    Returns:
        Mock: A mock ConversationContext instance
    """
    context = Mock(spec=ConversationContext)
    context.add_turn = Mock()
    context.get_recent_turns = Mock(return_value=[])
    context.get_context_summary = Mock(return_value="Mock context summary")
    context.clear_conversation = Mock()
    context.save_to_file = Mock(return_value=True)
    context.load_from_file = Mock(return_value=True)
    context.get_conversation_stats = Mock(return_value={
        "total_turns": 0,
        "user_messages": 0,
        "assistant_messages": 0
    })
    return context


@pytest.fixture
def mock_intent_engine():
    """
    Create a mock intent engine for testing.
    
    Returns:
        Mock: A mock ARKIntentEngine instance
    """
    engine = Mock(spec=ARKIntentEngine)
    engine.analyze_intent = AsyncMock(return_value={
        "intent": "general_query",
        "confidence": 0.8,
        "entities": [],
        "requires_tools": False
    })
    engine.extract_entities = Mock(return_value=[])
    engine.get_intent_confidence = Mock(return_value=0.8)
    return engine


@pytest.fixture
def mock_tool_registry():
    """
    Create a mock tool registry for testing.
    
    Returns:
        Mock: A mock ARKToolRegistry instance
    """
    registry = Mock(spec=ARKToolRegistry)
    registry.register_tool = Mock(return_value=True)
    registry.unregister_tool = Mock(return_value=True)
    registry.discover_tools = AsyncMock(return_value=[])
    registry.get_tool = Mock(return_value=None)
    registry.list_tools = Mock(return_value=[])
    registry.get_tools_by_category = Mock(return_value=[])
    registry.get_registry_stats = Mock(return_value={
        "total_tools": 0,
        "active_tools": 0,
        "categories": []
    })
    return registry


@pytest.fixture
def mock_server_config_manager():
    """
    Create a mock server configuration manager for testing.
    
    Returns:
        Mock: A mock ARKServerConfigManager instance
    """
    manager = Mock(spec=ARKServerConfigManager)
    manager.load_config = AsyncMock(return_value=True)
    manager.save_config = AsyncMock(return_value=True)
    manager.add_server = Mock(return_value=True)
    manager.remove_server = Mock(return_value=True)
    manager.get_server = Mock(return_value=None)
    manager.list_servers = Mock(return_value=[])
    manager.validate_config = Mock(return_value=True)
    manager.health_check = AsyncMock(return_value={
        "status": "healthy",
        "servers_count": 0
    })
    return manager


@pytest.fixture
def mock_ark_engine():
    """
    Create a mock ARK engine for testing.
    
    Returns:
        Mock: A mock ARKEngine instance
    """
    engine = Mock(spec=ARKEngine)
    engine.initialize = AsyncMock(return_value=True)
    engine.shutdown = AsyncMock(return_value=True)
    engine.process_input = AsyncMock(return_value={
        "status": "success",
        "response": "Mock ARK response",
        "decision": {
            "action": "respond",
            "confidence": 0.9,
            "tools_used": []
        }
    })
    engine.get_status = Mock(return_value={
        "state": "ready",
        "initialized": True
    })
    return engine


@pytest.fixture
def mock_jarvis_agent(
    sample_jarvis_config,
    mock_ark_engine,
    mock_context_manager
):
    """
    Create a mock Jarvis agent for testing.
    
    Args:
        sample_jarvis_config: Sample configuration fixture
        mock_ark_engine: Mock ARK engine fixture
        mock_context_manager: Mock context manager fixture
        
    Returns:
        Mock: A mock JarvisAgent instance
    """
    agent = Mock(spec=JarvisAgent)
    agent.config = sample_jarvis_config
    agent.ark_engine = mock_ark_engine
    agent.context_manager = mock_context_manager
    
    agent.startup = AsyncMock(return_value=True)
    agent.shutdown = AsyncMock(return_value=True)
    agent.process_message = AsyncMock(return_value={
        "status": "success",
        "response": "Mock agent response",
        "conversation_id": "mock_conv_123"
    })
    agent.get_status = Mock(return_value={
        "is_running": True,
        "conversation_id": "mock_conv_123",
        "config": sample_jarvis_config.to_dict()
    })
    agent.clear_conversation = Mock(return_value=True)
    agent.get_conversation_stats = Mock(return_value={
        "total_messages": 0,
        "session_duration": 0
    })
    
    return agent


@pytest.fixture
def sample_conversation_data():
    """
    Create sample conversation data for testing.
    
    Returns:
        Dict[str, Any]: Sample conversation data
    """
    return {
        "conversation_id": "test_conv_123",
        "user_id": "test_user",
        "turns": [
            {
                "turn_id": "turn_1",
                "timestamp": "2024-01-01T10:00:00Z",
                "user_message": "Hello",
                "assistant_response": "Hi there! How can I help you?",
                "intent": "greeting",
                "entities": [],
                "tools_used": []
            },
            {
                "turn_id": "turn_2",
                "timestamp": "2024-01-01T10:01:00Z",
                "user_message": "What's the weather like?",
                "assistant_response": "I'd be happy to help with weather information.",
                "intent": "weather_query",
                "entities": [{"type": "query_type", "value": "weather"}],
                "tools_used": ["weather_tool"]
            }
        ],
        "metadata": {
            "created_at": "2024-01-01T10:00:00Z",
            "last_updated": "2024-01-01T10:01:00Z",
            "total_turns": 2
        }
    }


@pytest.fixture
def sample_tool_metadata():
    """
    Create sample tool metadata for testing.
    
    Returns:
        Dict[str, Any]: Sample tool metadata
    """
    return {
        "name": "test_tool",
        "description": "A test tool for unit testing",
        "version": "1.0.0",
        "category": "testing",
        "parameters": {
            "input": {
                "type": "string",
                "description": "Test input parameter"
            }
        },
        "returns": {
            "type": "string",
            "description": "Test output"
        },
        "examples": [
            {
                "input": {"input": "test"},
                "output": "test result"
            }
        ]
    }


@pytest.fixture
def sample_server_configs():
    """
    Create sample server configurations for testing.
    
    Returns:
        List[SimpleMCPServerConfig]: List of sample server configurations
    """
    return [
        SimpleMCPServerConfig(
            name="weather_server",
            command=["python", "-m", "weather_server"],
            env={"API_KEY": "test_key"}
        ),
        SimpleMCPServerConfig(
            name="calendar_server",
            command=["node", "calendar_server.js"],
            env={"NODE_ENV": "test"}
        ),
        SimpleMCPServerConfig(
            name="file_server",
            command=["python", "-m", "file_server"],
            env={}
        )
    ]


@pytest.fixture(autouse=True)
def reset_singletons():
    """
    Reset singleton instances between tests.
    
    This fixture automatically runs before each test to ensure
    that singleton instances don't carry state between tests.
    """
    # Reset any singleton instances here if needed
    yield
    # Cleanup after test if needed


@pytest.fixture
def capture_logs(caplog):
    """
    Capture and provide access to log messages during tests.
    
    Args:
        caplog: Pytest's built-in log capture fixture
        
    Returns:
        caplog: The log capture fixture
    """
    return caplog


class AsyncContextManager:
    """
    Helper class for creating async context managers in tests.
    """
    
    def __init__(self, return_value=None):
        self.return_value = return_value
    
    async def __aenter__(self):
        return self.return_value
    
    async def __aaenter__(self):
        pass


@pytest.fixture
def async_context_manager():
    """
    Create an async context manager for testing.
    
    Returns:
        AsyncContextManager: Helper for async context management
    """
    return AsyncContextManager


# Test utilities
def assert_config_equal(config1: JarvisConfig, config2: JarvisConfig):
    """
    Assert that two JarvisConfig instances are equal.
    
    Args:
        config1: First configuration
        config2: Second configuration
    """
    assert config1.name == config2.name
    assert config1.version == config2.version
    assert config1.debug == config2.debug
    assert config1.max_conversation_length == config2.max_conversation_length
    assert config1.log_level == config2.log_level
    assert config1.config_file == config2.config_file


def create_test_file(path: str, content: str) -> None:
    """
    Create a test file with the given content.
    
    Args:
        path: File path
        content: File content
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)


def read_test_file(path: str) -> str:
    """
    Read content from a test file.
    
    Args:
        path: File path
        
    Returns:
        str: File content
    """
    with open(path, 'r') as f:
        return f.read()


# Pytest configuration
def pytest_configure(config):
    """
    Configure pytest with custom markers and settings.
    
    Args:
        config: Pytest configuration object
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "asyncio: marks tests as async tests"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers automatically.
    
    Args:
        config: Pytest configuration object
        items: List of test items
    """
    for item in items:
        # Add asyncio marker to async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)
        
        # Add unit marker to unit tests
        if "test_" in item.name and "integration" not in item.name:
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to integration tests
        if "integration" in item.name:
            item.add_marker(pytest.mark.integration)