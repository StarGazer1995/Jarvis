"""
Tests for the Tool Registry module.

This module contains comprehensive tests for the MCP tool registration and
discovery system, including tool metadata, filtering, discovery providers,
and the central registry functionality.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.core.tool_registry import (
    ToolStatus, ToolCategory, ToolMetadata, ToolFilter,
    ToolDiscoveryProvider, MCPToolDiscoveryProvider, FileToolDiscoveryProvider, ARKToolRegistry
)


class TestToolStatus:
    """Test cases for ToolStatus enum."""
    
    def test_tool_status_values(self):
        """Test that all expected tool statuses are defined."""
        expected_statuses = ["AVAILABLE", "UNAVAILABLE", "DEPRECATED", "EXPERIMENTAL"]
        
        for status in expected_statuses:
            assert hasattr(ToolStatus, status)
        
        # Test specific values
        assert ToolStatus.AVAILABLE.value == "available"
        assert ToolStatus.UNAVAILABLE.value == "unavailable"
        assert ToolStatus.DEPRECATED.value == "deprecated"
        assert ToolStatus.EXPERIMENTAL.value == "experimental"


class TestToolCategory:
    """Test cases for ToolCategory enum."""
    
    def test_tool_category_values(self):
        """Test that all expected tool categories are defined."""
        expected_categories = [
            "GENERAL", "WEATHER", "TIME", "SEARCH", "COMMUNICATION",
            "FILE_SYSTEM", "CALCULATION", "ENTERTAINMENT", "PRODUCTIVITY"
        ]
        
        for category in expected_categories:
            assert hasattr(ToolCategory, category)
        
        # Test specific values
        assert ToolCategory.GENERAL.value == "general"
        assert ToolCategory.WEATHER.value == "weather"
        assert ToolCategory.TIME.value == "time"
        assert ToolCategory.SEARCH.value == "search"


class TestToolMetadata:
    """Test cases for ToolMetadata class."""
    
    def test_metadata_creation(self):
        """Test basic tool metadata creation."""
        metadata = ToolMetadata(
            name="weather_tool",
            description="Get weather information",
            category=ToolCategory.WEATHER,
            status=ToolStatus.AVAILABLE,
            version="1.0.0",
            provider="weather_service"
        )
        
        assert metadata.name == "weather_tool"
        assert metadata.description == "Get weather information"
        assert metadata.category == ToolCategory.WEATHER
        assert metadata.status == ToolStatus.AVAILABLE
        assert metadata.version == "1.0.0"
        assert metadata.provider == "weather_service"
        assert isinstance(metadata.created_at, datetime)
        assert metadata.parameters == {}
        assert metadata.tags == []
    
    def test_metadata_with_parameters(self):
        """Test tool metadata with parameters."""
        parameters = {
            "location": {"type": "string", "required": True},
            "units": {"type": "string", "default": "celsius"}
        }
        
        metadata = ToolMetadata(
            name="weather_tool",
            description="Get weather",
            category=ToolCategory.WEATHER,
            parameters=parameters
        )
        
        assert metadata.parameters == parameters
        assert "location" in metadata.parameters
        assert metadata.parameters["location"]["required"] is True
    
    def test_metadata_with_tags(self):
        """Test tool metadata with tags."""
        tags = ["weather", "forecast", "temperature"]
        
        metadata = ToolMetadata(
            name="weather_tool",
            description="Get weather",
            category=ToolCategory.WEATHER,
            tags=tags
        )
        
        assert metadata.tags == tags
        assert "weather" in metadata.tags
        assert "forecast" in metadata.tags
    
    def test_metadata_to_dict(self):
        """Test tool metadata serialization."""
        metadata = ToolMetadata(
            name="test_tool",
            description="Test tool",
            category=ToolCategory.GENERAL,
            status=ToolStatus.AVAILABLE,
            version="2.0.0",
            provider="test_provider",
            tags=["test", "demo"]
        )
        
        metadata_dict = metadata.to_dict()
        
        assert metadata_dict["name"] == "test_tool"
        assert metadata_dict["description"] == "Test tool"
        assert metadata_dict["category"] == "general"
        assert metadata_dict["status"] == "available"
        assert metadata_dict["version"] == "2.0.0"
        assert metadata_dict["provider"] == "test_provider"
        assert metadata_dict["tags"] == ["test", "demo"]
        assert "created_at" in metadata_dict
    
    def test_metadata_from_dict(self):
        """Test tool metadata deserialization."""
        metadata_data = {
            "name": "search_tool",
            "description": "Search for information",
            "category": "search",
            "status": "available",
            "version": "1.5.0",
            "provider": "search_service",
            "parameters": {"query": {"type": "string", "required": True}},
            "tags": ["search", "information"],
            "created_at": "2024-01-01T12:00:00"
        }
        
        metadata = ToolMetadata.from_dict(metadata_data)
        
        assert metadata.name == "search_tool"
        assert metadata.description == "Search for information"
        assert metadata.category == ToolCategory.SEARCH
        assert metadata.status == ToolStatus.AVAILABLE
        assert metadata.version == "1.5.0"
        assert metadata.provider == "search_service"
        assert "query" in metadata.parameters
        assert metadata.tags == ["search", "information"]
    
    def test_metadata_string_representation(self):
        """Test tool metadata string representation."""
        metadata = ToolMetadata(
            name="time_tool",
            description="Get current time",
            category=ToolCategory.TIME
        )
        
        repr_str = repr(metadata)
        assert "ToolMetadata" in repr_str
        assert "time_tool" in repr_str
        assert "time" in repr_str


class TestToolFilter:
    """Test cases for ToolFilter class."""
    
    def test_filter_creation(self):
        """Test basic tool filter creation."""
        tool_filter = ToolFilter(
            categories=[ToolCategory.WEATHER, ToolCategory.TIME],
            status=ToolStatus.AVAILABLE,
            provider="test_provider",
            tags=["weather", "time"]
        )
        
        assert ToolCategory.WEATHER in tool_filter.categories
        assert ToolCategory.TIME in tool_filter.categories
        assert tool_filter.status == ToolStatus.AVAILABLE
        assert tool_filter.provider == "test_provider"
        assert "weather" in tool_filter.tags
        assert "time" in tool_filter.tags
    
    def test_filter_matches_category(self):
        """Test filter matching by category."""
        tool_filter = ToolFilter(categories=[ToolCategory.WEATHER])
        
        weather_tool = ToolMetadata(
            name="weather_tool",
            description="Weather",
            category=ToolCategory.WEATHER
        )
        
        time_tool = ToolMetadata(
            name="time_tool",
            description="Time",
            category=ToolCategory.TIME
        )
        
        assert tool_filter.matches(weather_tool)
        assert not tool_filter.matches(time_tool)
    
    def test_filter_matches_status(self):
        """Test filter matching by status."""
        tool_filter = ToolFilter(status=ToolStatus.AVAILABLE)
        
        available_tool = ToolMetadata(
            name="available_tool",
            description="Available",
            category=ToolCategory.GENERAL,
            status=ToolStatus.AVAILABLE
        )
        
        deprecated_tool = ToolMetadata(
            name="deprecated_tool",
            description="Deprecated",
            category=ToolCategory.GENERAL,
            status=ToolStatus.DEPRECATED
        )
        
        assert tool_filter.matches(available_tool)
        assert not tool_filter.matches(deprecated_tool)
    
    def test_filter_matches_provider(self):
        """Test filter matching by provider."""
        tool_filter = ToolFilter(provider="weather_service")
        
        weather_tool = ToolMetadata(
            name="weather_tool",
            description="Weather",
            category=ToolCategory.WEATHER,
            provider="weather_service"
        )
        
        time_tool = ToolMetadata(
            name="time_tool",
            description="Time",
            category=ToolCategory.TIME,
            provider="time_service"
        )
        
        assert tool_filter.matches(weather_tool)
        assert not tool_filter.matches(time_tool)
    
    def test_filter_matches_tags(self):
        """Test filter matching by tags."""
        tool_filter = ToolFilter(tags=["weather", "forecast"])
        
        weather_tool = ToolMetadata(
            name="weather_tool",
            description="Weather",
            category=ToolCategory.WEATHER,
            tags=["weather", "forecast", "temperature"]
        )
        
        time_tool = ToolMetadata(
            name="time_tool",
            description="Time",
            category=ToolCategory.TIME,
            tags=["time", "clock"]
        )
        
        assert tool_filter.matches(weather_tool)
        assert not tool_filter.matches(time_tool)
    
    def test_filter_matches_name_pattern(self):
        """Test filter matching by name pattern."""
        tool_filter = ToolFilter(name_pattern="weather_*")
        
        weather_tool = ToolMetadata(
            name="weather_forecast",
            description="Weather forecast",
            category=ToolCategory.WEATHER
        )
        
        time_tool = ToolMetadata(
            name="time_current",
            description="Current time",
            category=ToolCategory.TIME
        )
        
        assert tool_filter.matches(weather_tool)
        assert not tool_filter.matches(time_tool)
    
    def test_filter_matches_multiple_criteria(self):
        """Test filter matching with multiple criteria."""
        tool_filter = ToolFilter(
            categories=[ToolCategory.WEATHER],
            status=ToolStatus.AVAILABLE,
            provider="weather_service"
        )
        
        matching_tool = ToolMetadata(
            name="weather_tool",
            description="Weather",
            category=ToolCategory.WEATHER,
            status=ToolStatus.AVAILABLE,
            provider="weather_service"
        )
        
        non_matching_tool = ToolMetadata(
            name="weather_tool",
            description="Weather",
            category=ToolCategory.WEATHER,
            status=ToolStatus.DEPRECATED,  # Different status
            provider="weather_service"
        )
        
        assert tool_filter.matches(matching_tool)
        assert not tool_filter.matches(non_matching_tool)
    
    def test_empty_filter_matches_all(self):
        """Test that empty filter matches all tools."""
        tool_filter = ToolFilter()
        
        tools = [
            ToolMetadata("tool1", "desc1", ToolCategory.WEATHER),
            ToolMetadata("tool2", "desc2", ToolCategory.TIME),
            ToolMetadata("tool3", "desc3", ToolCategory.SEARCH)
        ]
        
        for tool in tools:
            assert tool_filter.matches(tool)



class TestMCPToolDiscoveryProvider:
    """Test cases for MCPToolDiscoveryProvider class."""
    
    def test_provider_creation(self):
        """Test MCP discovery provider creation."""
        mock_client = Mock()
        provider = MCPToolDiscoveryProvider(mock_client)
        
        assert provider.mcp_client == mock_client
        assert provider.name == "mcp_provider"
    
    @pytest.mark.asyncio
    async def test_discover_tools_success(self):
        """Test successful tool discovery from MCP client."""
        mock_client = Mock()
        mock_client.list_tools = AsyncMock(return_value=[
            {
                "name": "weather_tool",
                "description": "Get weather information",
                "category": "weather",
                "parameters": {"location": {"type": "string", "required": True}}
            },
            {
                "name": "time_tool",
                "description": "Get current time",
                "category": "time",
                "parameters": {}
            }
        ])
        
        provider = MCPToolDiscoveryProvider(mock_client)
        tools = await provider.discover_tools()
        
        assert len(tools) == 2
        assert tools[0].name == "weather_tool"
        assert tools[0].category == ToolCategory.WEATHER
        assert "location" in tools[0].parameters
        assert tools[1].name == "time_tool"
        assert tools[1].category == ToolCategory.TIME
    
    @pytest.mark.asyncio
    async def test_discover_tools_client_error(self):
        """Test tool discovery with MCP client error."""
        mock_client = Mock()
        mock_client.list_tools = AsyncMock(side_effect=Exception("Connection error"))
        
        provider = MCPToolDiscoveryProvider(mock_client)
        tools = await provider.discover_tools()
        
        assert tools == []
    
    @pytest.mark.asyncio
    async def test_discover_tools_with_filter(self):
        """Test MCP tool discovery with filter."""
        mock_client = Mock()
        mock_client.list_tools = AsyncMock(return_value=[
            {
                "name": "weather_tool",
                "description": "Weather",
                "category": "weather"
            },
            {
                "name": "search_tool",
                "description": "Search",
                "category": "search"
            }
        ])
        
        provider = MCPToolDiscoveryProvider(mock_client)
        tool_filter = ToolFilter(categories=[ToolCategory.WEATHER])
        tools = await provider.discover_tools(tool_filter)
        
        assert len(tools) == 1
        assert tools[0].name == "weather_tool"
        assert tools[0].category == ToolCategory.WEATHER


class TestARKToolRegistry:
    """Test cases for ARKToolRegistry class."""
    
    @pytest.fixture
    def registry(self):
        """Create a tool registry for testing."""
        return ARKToolRegistry()
    
    @pytest.fixture
    def sample_tools(self):
        """Create sample tools for testing."""
        return [
            ToolMetadata(
                name="weather_tool",
                description="Get weather information",
                category=ToolCategory.WEATHER,
                status=ToolStatus.AVAILABLE,
                provider="weather_service",
                tags=["weather", "forecast"]
            ),
            ToolMetadata(
                name="time_tool",
                description="Get current time",
                category=ToolCategory.TIME,
                status=ToolStatus.AVAILABLE,
                provider="time_service",
                tags=["time", "clock"]
            ),
            ToolMetadata(
                name="deprecated_tool",
                description="Old tool",
                category=ToolCategory.GENERAL,
                status=ToolStatus.DEPRECATED,
                provider="old_service"
            )
        ]
    
    def test_registry_creation(self, registry):
        """Test tool registry creation."""
        assert registry.tools == {}
        assert registry.providers == []
        assert registry.execution_stats == {}
    
    def test_register_tool(self, registry, sample_tools):
        """Test tool registration."""
        tool = sample_tools[0]
        registry.register_tool(tool)
        
        assert "weather_tool" in registry.tools
        assert registry.tools["weather_tool"] == tool
    
    def test_register_duplicate_tool(self, registry, sample_tools):
        """Test registering duplicate tool."""
        tool = sample_tools[0]
        registry.register_tool(tool)
        
        # Register same tool again
        registry.register_tool(tool)
        
        # Should still have only one entry
        assert len(registry.tools) == 1
        assert registry.tools["weather_tool"] == tool
    
    def test_unregister_tool(self, registry, sample_tools):
        """Test tool unregistration."""
        tool = sample_tools[0]
        registry.register_tool(tool)
        
        assert "weather_tool" in registry.tools
        
        registry.unregister_tool("weather_tool")
        
        assert "weather_tool" not in registry.tools
    
    def test_unregister_nonexistent_tool(self, registry):
        """Test unregistering nonexistent tool."""
        # Should not raise exception
        registry.unregister_tool("nonexistent_tool")
        assert len(registry.tools) == 0
    
    def test_get_tool(self, registry, sample_tools):
        """Test getting tool by name."""
        tool = sample_tools[0]
        registry.register_tool(tool)
        
        retrieved_tool = registry.get_tool("weather_tool")
        assert retrieved_tool == tool
        
        # Test nonexistent tool
        assert registry.get_tool("nonexistent_tool") is None
    
    def test_list_tools_no_filter(self, registry, sample_tools):
        """Test listing all tools without filter."""
        for tool in sample_tools:
            registry.register_tool(tool)
        
        tools = registry.list_tools()
        
        assert len(tools) == 3
        tool_names = [tool.name for tool in tools]
        assert "weather_tool" in tool_names
        assert "time_tool" in tool_names
        assert "deprecated_tool" in tool_names
    
    def test_list_tools_with_category_filter(self, registry, sample_tools):
        """Test listing tools with category filter."""
        for tool in sample_tools:
            registry.register_tool(tool)
        
        tool_filter = ToolFilter(categories=[ToolCategory.WEATHER])
        tools = registry.list_tools(tool_filter)
        
        assert len(tools) == 1
        assert tools[0].name == "weather_tool"
        assert tools[0].category == ToolCategory.WEATHER
    
    def test_list_tools_with_status_filter(self, registry, sample_tools):
        """Test listing tools with status filter."""
        for tool in sample_tools:
            registry.register_tool(tool)
        
        tool_filter = ToolFilter(status=ToolStatus.AVAILABLE)
        tools = registry.list_tools(tool_filter)
        
        assert len(tools) == 2
        tool_names = [tool.name for tool in tools]
        assert "weather_tool" in tool_names
        assert "time_tool" in tool_names
        assert "deprecated_tool" not in tool_names
    
    def test_list_tools_with_provider_filter(self, registry, sample_tools):
        """Test listing tools with provider filter."""
        for tool in sample_tools:
            registry.register_tool(tool)
        
        tool_filter = ToolFilter(provider="weather_service")
        tools = registry.list_tools(tool_filter)
        
        assert len(tools) == 1
        assert tools[0].name == "weather_tool"
        assert tools[0].provider == "weather_service"
    
    def test_list_tools_with_tags_filter(self, registry, sample_tools):
        """Test listing tools with tags filter."""
        for tool in sample_tools:
            registry.register_tool(tool)
        
        tool_filter = ToolFilter(tags=["weather"])
        tools = registry.list_tools(tool_filter)
        
        assert len(tools) == 1
        assert tools[0].name == "weather_tool"
        assert "weather" in tools[0].tags
    
    def test_add_discovery_provider(self, registry):
        """Test adding discovery provider."""
        mock_provider = Mock(spec=ToolDiscoveryProvider)
        mock_provider.name = "test_provider"
        
        registry.add_discovery_provider(mock_provider)
        
        assert mock_provider in registry.providers
        assert len(registry.providers) == 1
    
    def test_remove_discovery_provider(self, registry):
        """Test removing discovery provider."""
        mock_provider = Mock(spec=ToolDiscoveryProvider)
        mock_provider.name = "test_provider"
        
        registry.add_discovery_provider(mock_provider)
        assert mock_provider in registry.providers
        
        registry.remove_discovery_provider("test_provider")
        assert mock_provider not in registry.providers
    
    @pytest.mark.asyncio
    async def test_discover_tools(self, registry):
        """Test tool discovery from providers."""
        # Create mock provider
        mock_provider = Mock(spec=ToolDiscoveryProvider)
        mock_provider.name = "test_provider"
        mock_provider.discover_tools = AsyncMock(return_value=[
            ToolMetadata("discovered_tool", "Discovered", ToolCategory.GENERAL)
        ])
        
        registry.add_discovery_provider(mock_provider)
        
        discovered_tools = await registry.discover_tools()
        
        assert len(discovered_tools) == 1
        assert discovered_tools[0].name == "discovered_tool"
        
        # Tool should also be registered
        assert "discovered_tool" in registry.tools
    
    @pytest.mark.asyncio
    async def test_discover_tools_with_filter(self, registry):
        """Test tool discovery with filter."""
        mock_provider = Mock(spec=ToolDiscoveryProvider)
        mock_provider.name = "test_provider"
        mock_provider.discover_tools = AsyncMock(return_value=[
            ToolMetadata("weather_tool", "Weather", ToolCategory.WEATHER),
            ToolMetadata("time_tool", "Time", ToolCategory.TIME)
        ])
        
        registry.add_discovery_provider(mock_provider)
        
        tool_filter = ToolFilter(categories=[ToolCategory.WEATHER])
        discovered_tools = await registry.discover_tools(tool_filter)
        
        # Provider should be called with filter
        mock_provider.discover_tools.assert_called_once_with(tool_filter)
        
        assert len(discovered_tools) == 2  # Provider returns both, registry filters
        # But only weather tool should be registered due to filter
        filtered_tools = registry.list_tools(tool_filter)
        assert len(filtered_tools) == 1
        assert filtered_tools[0].name == "weather_tool"
    
    @pytest.mark.asyncio
    async def test_discover_tools_provider_error(self, registry):
        """Test tool discovery with provider error."""
        mock_provider = Mock(spec=ToolDiscoveryProvider)
        mock_provider.name = "error_provider"
        mock_provider.discover_tools = AsyncMock(side_effect=Exception("Discovery error"))
        
        registry.add_discovery_provider(mock_provider)
        
        # Should handle error gracefully
        discovered_tools = await registry.discover_tools()
        assert discovered_tools == []
    
    def test_validate_tool_valid(self, registry):
        """Test tool validation with valid tool."""
        tool = ToolMetadata(
            name="valid_tool",
            description="Valid tool",
            category=ToolCategory.GENERAL,
            status=ToolStatus.AVAILABLE
        )
        
        is_valid, errors = registry.validate_tool(tool)
        
        assert is_valid is True
        assert errors == []
    
    def test_validate_tool_invalid_name(self, registry):
        """Test tool validation with invalid name."""
        tool = ToolMetadata(
            name="",  # Empty name
            description="Tool with empty name",
            category=ToolCategory.GENERAL
        )
        
        is_valid, errors = registry.validate_tool(tool)
        
        assert is_valid is False
        assert len(errors) > 0
        assert any("name" in error.lower() for error in errors)
    
    def test_validate_tool_invalid_description(self, registry):
        """Test tool validation with invalid description."""
        tool = ToolMetadata(
            name="tool_no_desc",
            description="",  # Empty description
            category=ToolCategory.GENERAL
        )
        
        is_valid, errors = registry.validate_tool(tool)
        
        assert is_valid is False
        assert len(errors) > 0
        assert any("description" in error.lower() for error in errors)
    
    def test_record_execution(self, registry, sample_tools):
        """Test recording tool execution."""
        tool = sample_tools[0]
        registry.register_tool(tool)
        
        # Record successful execution
        registry.record_execution("weather_tool", True, 0.5)
        
        stats = registry.execution_stats["weather_tool"]
        assert stats["total_executions"] == 1
        assert stats["successful_executions"] == 1
        assert stats["failed_executions"] == 0
        assert stats["average_execution_time"] == 0.5
        assert stats["last_execution"] is not None
    
    def test_record_execution_failure(self, registry, sample_tools):
        """Test recording failed tool execution."""
        tool = sample_tools[0]
        registry.register_tool(tool)
        
        # Record failed execution
        registry.record_execution("weather_tool", False, 1.0)
        
        stats = registry.execution_stats["weather_tool"]
        assert stats["total_executions"] == 1
        assert stats["successful_executions"] == 0
        assert stats["failed_executions"] == 1
        assert stats["average_execution_time"] == 1.0
    
    def test_record_multiple_executions(self, registry, sample_tools):
        """Test recording multiple tool executions."""
        tool = sample_tools[0]
        registry.register_tool(tool)
        
        # Record multiple executions
        registry.record_execution("weather_tool", True, 0.5)
        registry.record_execution("weather_tool", True, 1.0)
        registry.record_execution("weather_tool", False, 2.0)
        
        stats = registry.execution_stats["weather_tool"]
        assert stats["total_executions"] == 3
        assert stats["successful_executions"] == 2
        assert stats["failed_executions"] == 1
        assert stats["average_execution_time"] == (0.5 + 1.0 + 2.0) / 3
    
    def test_get_execution_stats(self, registry, sample_tools):
        """Test getting execution statistics."""
        tool = sample_tools[0]
        registry.register_tool(tool)
        
        # Record some executions
        registry.record_execution("weather_tool", True, 0.5)
        registry.record_execution("weather_tool", False, 1.0)
        
        stats = registry.get_execution_stats("weather_tool")
        
        assert stats is not None
        assert stats["total_executions"] == 2
        assert stats["successful_executions"] == 1
        assert stats["failed_executions"] == 1
        
        # Test nonexistent tool
        assert registry.get_execution_stats("nonexistent_tool") is None
    
    def test_get_all_execution_stats(self, registry, sample_tools):
        """Test getting all execution statistics."""
        # Register tools and record executions
        for tool in sample_tools[:2]:  # Only first two tools
            registry.register_tool(tool)
            registry.record_execution(tool.name, True, 0.5)
        
        all_stats = registry.get_all_execution_stats()
        
        assert len(all_stats) == 2
        assert "weather_tool" in all_stats
        assert "time_tool" in all_stats
        assert "deprecated_tool" not in all_stats  # No executions recorded
    
    def test_clear_tools(self, registry, sample_tools):
        """Test clearing all tools."""
        for tool in sample_tools:
            registry.register_tool(tool)
        
        assert len(registry.tools) == 3
        
        registry.clear_tools()
        
        assert len(registry.tools) == 0
        assert registry.execution_stats == {}
    
    def test_get_tool_count(self, registry, sample_tools):
        """Test getting tool count."""
        assert registry.get_tool_count() == 0
        
        for tool in sample_tools:
            registry.register_tool(tool)
        
        assert registry.get_tool_count() == 3
        
        # Test with filter
        tool_filter = ToolFilter(status=ToolStatus.AVAILABLE)
        available_count = registry.get_tool_count(tool_filter)
        assert available_count == 2  # weather_tool and time_tool
    
    def test_get_categories(self, registry, sample_tools):
        """Test getting available categories."""
        for tool in sample_tools:
            registry.register_tool(tool)
        
        categories = registry.get_categories()
        
        assert ToolCategory.WEATHER in categories
        assert ToolCategory.TIME in categories
        assert ToolCategory.GENERAL in categories
        assert len(categories) == 3
    
    def test_get_providers(self, registry, sample_tools):
        """Test getting available providers."""
        for tool in sample_tools:
            registry.register_tool(tool)
        
        providers = registry.get_providers()
        
        assert "weather_service" in providers
        assert "time_service" in providers
        assert "old_service" in providers
        assert len(providers) == 3


class TestToolRegistryIntegration:
    """Integration tests for tool registry functionality."""
    
    @pytest.mark.asyncio
    async def test_full_discovery_and_registration_flow(self):
        """Test complete tool discovery and registration flow."""
        registry = ARKToolRegistry()
        
        # Create file provider
        tools_data = {
            "tools": [
                {
                    "name": "weather_tool",
                    "description": "Get weather",
                    "category": "weather",
                    "status": "available",
                    "provider": "weather_service"
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(tools_data, f)
            temp_file = f.name
        
        try:
            file_provider = FileToolDiscoveryProvider(temp_file)
            registry.add_discovery_provider(file_provider)
            
            # Create MCP provider
            mock_client = Mock()
            mock_client.list_tools = AsyncMock(return_value=[
                {
                    "name": "time_tool",
                    "description": "Get time",
                    "category": "time"
                }
            ])
            
            mcp_provider = MCPToolDiscoveryProvider(mock_client)
            registry.add_discovery_provider(mcp_provider)
            
            # Discover tools
            discovered_tools = await registry.discover_tools()
            
            # Should have tools from both providers
            assert len(discovered_tools) == 2
            tool_names = [tool.name for tool in discovered_tools]
            assert "weather_tool" in tool_names
            assert "time_tool" in tool_names
            
            # Tools should be registered
            assert len(registry.tools) == 2
            assert registry.get_tool("weather_tool") is not None
            assert registry.get_tool("time_tool") is not None
            
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_performance_with_many_tools(self):
        """Test registry performance with many tools."""
        registry = ARKToolRegistry()
        
        # Register many tools
        tools = []
        for i in range(1000):
            tool = ToolMetadata(
                name=f"tool_{i}",
                description=f"Tool number {i}",
                category=ToolCategory.GENERAL,
                status=ToolStatus.AVAILABLE,
                provider=f"provider_{i % 10}"  # 10 different providers
            )
            tools.append(tool)
            registry.register_tool(tool)
        
        # Test listing performance
        import time
        start_time = time.time()
        
        all_tools = registry.list_tools()
        assert len(all_tools) == 1000
        
        # Test filtering performance
        tool_filter = ToolFilter(provider="provider_0")
        filtered_tools = registry.list_tools(tool_filter)
        assert len(filtered_tools) == 100  # Every 10th tool
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should be fast even with many tools
        assert processing_time < 1.0  # Less than 1 second
    
    def test_concurrent_access(self):
        """Test concurrent access to registry."""
        import threading
        import time
        
        registry = ARKToolRegistry()
        results = []
        
        def register_tools(start_index):
            for i in range(start_index, start_index + 100):
                tool = ToolMetadata(
                    name=f"tool_{i}",
                    description=f"Tool {i}",
                    category=ToolCategory.GENERAL
                )
                registry.register_tool(tool)
                results.append(f"registered_{i}")
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=register_tools, args=(i * 100,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have registered all tools
        assert len(registry.tools) == 500
        assert len(results) == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])