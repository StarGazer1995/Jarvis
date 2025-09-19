"""
Tests for the Server Configuration module.

This module contains comprehensive tests for the MCP server configuration
system, including configuration types, providers, validation, and the
central configuration manager.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.core.server_config import (
    ServerStatus, ServerType, AuthType, ConfigType, MCPServerConfig, ServerConfigProvider,
    FileConfigProvider, EnvironmentConfigProvider, ARKServerConfigManager
)
from src.core.security_manager import SecurityLevel


class TestServerType:
    """Test cases for ServerType enum."""
    
    def test_server_type_values(self):
        """Test ServerType enum values."""
        assert ServerType.STDIO.value == "stdio"
        assert ServerType.HTTP.value == "http"
        assert ServerType.WEBSOCKET.value == "websocket"
        assert ServerType.TCP.value == "tcp"
        assert ServerType.CUSTOM.value == "custom"
        
        # Test enum membership
        assert ServerType.STDIO in ServerType
        assert ServerType.HTTP in ServerType
        assert ServerType.WEBSOCKET in ServerType
        assert ServerType.TCP in ServerType
        assert ServerType.CUSTOM in ServerType


class TestServerStatus:
    """Test cases for ServerStatus enum."""
    
    def test_server_status_values(self):
        """Test ServerStatus enum values."""
        assert ServerStatus.UNKNOWN.value == "unknown"
        assert ServerStatus.CONNECTING.value == "connecting"
        assert ServerStatus.CONNECTED.value == "connected"
        assert ServerStatus.DISCONNECTED.value == "disconnected"
        assert ServerStatus.ERROR.value == "error"
        assert ServerStatus.MAINTENANCE.value == "maintenance"
        assert ServerStatus.DISABLED.value == "disabled"
        
        # Test enum membership
        assert ServerStatus.UNKNOWN in ServerStatus
        assert ServerStatus.CONNECTING in ServerStatus
        assert ServerStatus.CONNECTED in ServerStatus
        assert ServerStatus.DISCONNECTED in ServerStatus
        assert ServerStatus.ERROR in ServerStatus
        assert ServerStatus.MAINTENANCE in ServerStatus
        assert ServerStatus.DISABLED in ServerStatus


class TestAuthType:
    """Test cases for AuthType enum."""
    
    def test_auth_type_values(self):
        """Test AuthType enum values."""
        assert AuthType.NONE.value == "none"
        assert AuthType.API_KEY.value == "api_key"
        assert AuthType.BEARER_TOKEN.value == "bearer_token"
        assert AuthType.BASIC_AUTH.value == "basic_auth"
        assert AuthType.OAUTH2.value == "oauth2"
        assert AuthType.CUSTOM.value == "custom"
        
        # Test enum membership
        assert AuthType.NONE in AuthType
        assert AuthType.API_KEY in AuthType
        assert AuthType.BEARER_TOKEN in AuthType
        assert AuthType.BASIC_AUTH in AuthType
        assert AuthType.OAUTH2 in AuthType
        assert AuthType.CUSTOM in AuthType


class TestMCPServerConfig:
    """Test cases for MCPServerConfig class."""
    
    def test_config_creation_minimal(self):
        """Test basic server configuration creation with minimal parameters."""
        config = MCPServerConfig(
            name="test_server",
            server_type=ServerType.HTTP,
            host="localhost",
            port=8080
        )
        
        assert config.name == "test_server"
        assert config.host == "localhost"
        assert config.port == 8080
        assert config.protocol == "http"
        assert config.enabled is True
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.security_level == SecurityLevel.MEDIUM
        assert config.config_type == ConfigType.DEVELOPMENT
        assert config.api_key is None
        assert config.headers == {}
        assert config.metadata == {}
        assert isinstance(config.created_at, datetime)
        assert config.last_updated is None
    
    def test_config_creation_full(self):
        """Test server configuration creation with all parameters."""
        headers = {"Authorization": "Bearer token", "Content-Type": "application/json"}
        metadata = {"version": "1.0.0", "description": "Test server"}
        
        config = MCPServerConfig(
            name="production_server",
            server_type=ServerType.HTTP,
            host="api.example.com",
            port=443,
            protocol="https",
            enabled=True,
            timeout=60.0,
            max_retries=5,
            security_level=SecurityLevel.HIGH,
            config_type=ConfigType.PRODUCTION,
            api_key="secret_key_123",
            headers=headers,
            metadata=metadata
        )
        
        assert config.name == "production_server"
        assert config.host == "api.example.com"
        assert config.port == 443
        assert config.protocol == "https"
        assert config.enabled is True
        assert config.timeout == 60.0
        assert config.max_retries == 5
        assert config.security_level == SecurityLevel.HIGH
        assert config.config_type == ConfigType.PRODUCTION
        assert config.api_key == "secret_key_123"
        assert config.headers == headers
        assert config.metadata == metadata
    
    def test_config_url_property(self):
        """Test server configuration URL property."""
        config = MCPServerConfig(
            name="test_server",
            server_type=ServerType.HTTP,
            host="example.com",
            port=8080,
            protocol="https"
        )
        
        assert config.url == "https://example.com:8080"
    
    def test_config_url_property_default_ports(self):
        """Test URL property with default ports."""
        # HTTP default port
        http_config = MCPServerConfig(
            name="http_server",
            server_type=ServerType.HTTP,
            host="example.com",
            port=80,
            protocol="http"
        )
        assert http_config.url == "http://example.com"
        
        # HTTPS default port
        https_config = MCPServerConfig(
            name="https_server",
            server_type=ServerType.HTTP,
            host="example.com",
            port=443,
            protocol="https"
        )
        assert https_config.url == "https://example.com"
    
    def test_config_is_secure_property(self):
        """Test server configuration is_secure property."""
        http_config = MCPServerConfig(
            name="http_server",
            server_type=ServerType.HTTP,
            host="example.com",
            port=8080,
            protocol="http"
        )
        assert http_config.is_secure is False
        
        https_config = MCPServerConfig(
            name="https_server",
            server_type=ServerType.HTTP,
            host="example.com",
            port=443,
            protocol="https"
        )
        assert https_config.is_secure is True
    
    def test_config_to_dict(self):
        """Test server configuration serialization."""
        config = MCPServerConfig(
            name="test_server",
            server_type=ServerType.HTTP,
            host="localhost",
            port=8080,
            protocol="http",
            enabled=True,
            timeout=30.0,
            max_retries=3,
            security_level=SecurityLevel.MEDIUM,
            config_type=ConfigType.DEVELOPMENT,
            api_key="test_key",
            headers={"Content-Type": "application/json"},
            metadata={"version": "1.0.0"}
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["name"] == "test_server"
        assert config_dict["host"] == "localhost"
        assert config_dict["port"] == 8080
        assert config_dict["protocol"] == "http"
        assert config_dict["enabled"] is True
        assert config_dict["timeout"] == 30.0
        assert config_dict["max_retries"] == 3
        assert config_dict["security_level"] == "medium"
        assert config_dict["config_type"] == "development"
        assert config_dict["api_key"] == "test_key"
        assert config_dict["headers"] == {"Content-Type": "application/json"}
        assert config_dict["metadata"] == {"version": "1.0.0"}
        assert "created_at" in config_dict
        assert "last_updated" in config_dict
    
    def test_config_from_dict(self):
        """Test server configuration deserialization."""
        config_data = {
            "name": "api_server",
            "server_type": "http",
            "host": "api.example.com",
            "port": 443,
            "protocol": "https",
            "enabled": True,
            "timeout": 60.0,
            "max_retries": 5,
            "security_level": "high",
            "config_type": "production",
            "api_key": "prod_key_456",
            "headers": {"Authorization": "Bearer token"},
            "metadata": {"environment": "production"},
            "created_at": "2024-01-01T12:00:00",
            "last_updated": "2024-01-02T12:00:00"
        }
        
        config = MCPServerConfig.from_dict(config_data)
        
        assert config.name == "api_server"
        assert config.host == "api.example.com"
        assert config.port == 443
        assert config.protocol == "https"
        assert config.enabled is True
        assert config.timeout == 60.0
        assert config.max_retries == 5
        assert config.security_level == SecurityLevel.HIGH
        assert config.config_type == ConfigType.PRODUCTION
        assert config.api_key == "prod_key_456"
        assert config.headers == {"Authorization": "Bearer token"}
        assert config.metadata == {"environment": "production"}
    
    def test_config_update(self):
        """Test server configuration update."""
        config = MCPServerConfig(
            name="test_server",
            server_type=ServerType.HTTP,
            host="localhost",
            port=8080
        )
        
        original_created_at = config.created_at
        
        # Update configuration
        config.update(
            host="newhost.com",
            port=9090,
            enabled=False,
            timeout=45.0
        )
        
        assert config.host == "newhost.com"
        assert config.port == 9090
        assert config.enabled is False
        assert config.timeout == 45.0
        assert config.created_at == original_created_at  # Should not change
        assert config.last_updated is not None
        assert config.last_updated > original_created_at
    
    def test_config_validate_success(self):
        """Test successful configuration validation."""
        config = MCPServerConfig(
            name="valid_server",
            server_type=ServerType.HTTP,
            host="example.com",
            port=8080,
            protocol="https",
            timeout=30.0,
            max_retries=3
        )
        
        is_valid, errors = config.validate()
        
        assert is_valid is True
        assert errors == []
    
    def test_config_validate_invalid_name(self):
        """Test configuration validation with invalid name."""
        config = MCPServerConfig(
            name="",  # Empty name
            server_type=ServerType.HTTP,
            host="example.com",
            port=8080
        )
        
        is_valid, errors = config.validate()
        
        assert is_valid is False
        assert len(errors) > 0
        assert any("name" in error.lower() for error in errors)
    
    def test_config_validate_invalid_host(self):
        """Test configuration validation with invalid host."""
        config = MCPServerConfig(
            name="test_server",
            server_type=ServerType.HTTP,
            host="",  # Empty host
            port=8080
        )
        
        is_valid, errors = config.validate()
        
        assert is_valid is False
        assert len(errors) > 0
        assert any("host" in error.lower() for error in errors)
    
    def test_config_validate_invalid_port(self):
        """Test configuration validation with invalid port."""
        config = MCPServerConfig(
            name="test_server",
            server_type=ServerType.HTTP,
            host="example.com",
            port=0  # Invalid port
        )
        
        is_valid, errors = config.validate()
        
        assert is_valid is False
        assert len(errors) > 0
        assert any("port" in error.lower() for error in errors)
    
    def test_config_validate_invalid_protocol(self):
        """Test configuration validation with invalid protocol."""
        config = MCPServerConfig(
            name="test_server",
            server_type=ServerType.HTTP,
            host="example.com",
            port=8080,
            protocol="ftp"  # Invalid protocol
        )
        
        is_valid, errors = config.validate()
        
        assert is_valid is False
        assert len(errors) > 0
        assert any("protocol" in error.lower() for error in errors)
    
    def test_config_validate_invalid_timeout(self):
        """Test configuration validation with invalid timeout."""
        config = MCPServerConfig(
            name="test_server",
            server_type=ServerType.HTTP,
            host="example.com",
            port=8080,
            timeout=-1.0  # Negative timeout
        )
        
        is_valid, errors = config.validate()
        
        assert is_valid is False
        assert len(errors) > 0
        assert any("timeout" in error.lower() for error in errors)
    
    def test_config_validate_invalid_max_retries(self):
        """Test configuration validation with invalid max_retries."""
        config = MCPServerConfig(
            name="test_server",
            server_type=ServerType.HTTP,
            host="example.com",
            port=8080,
            max_retries=-1  # Negative retries
        )
        
        is_valid, errors = config.validate()
        
        assert is_valid is False
        assert len(errors) > 0
        assert any("retries" in error.lower() for error in errors)
    
    def test_config_string_representation(self):
        """Test server configuration string representation."""
        config = MCPServerConfig(
            name="test_server",
            server_type=ServerType.HTTP,
            host="example.com",
            port=8080,
            protocol="https"
        )
        
        repr_str = repr(config)
        assert "MCPServerConfig" in repr_str
        assert "test_server" in repr_str
        assert "https://example.com:8080" in repr_str


class TestFileConfigProvider:
    """Test cases for FileConfigProvider class."""
    
    def test_provider_creation(self):
        """Test file config provider creation."""
        provider = FileConfigProvider("/path/to/config.json")
        
        assert str(provider.file_path) == "/path/to/config.json"
        assert provider.name == "file_provider"
    
    @pytest.mark.asyncio
    async def test_load_configs_success(self):
        """Test successful configuration loading from file."""
        configs_data = {
            "servers": [
                {
                    "name": "weather_server",
                    "server_type": "http",
                    "host": "weather.api.com",
                    "port": 443,
                    "protocol": "https",
                    "enabled": True,
                    "security_level": "high",
                    "config_type": "production"
                },
                {
                    "name": "time_server",
                    "server_type": "http",
                    "host": "time.api.com",
                    "port": 80,
                    "protocol": "http",
                    "enabled": True,
                    "security_level": "medium",
                    "config_type": "development"
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(configs_data, f)
            temp_file = f.name
        
        try:
            provider = FileConfigProvider(temp_file)
            configs = await provider.load_configs()
            
            assert len(configs) == 2
            assert configs[0].name == "weather_server"
            assert configs[0].host == "weather.api.com"
            assert configs[0].security_level == SecurityLevel.HIGH
            assert configs[1].name == "time_server"
            assert configs[1].host == "time.api.com"
            assert configs[1].security_level == SecurityLevel.MEDIUM
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_load_configs_file_not_found(self):
        """Test configuration loading with missing file."""
        provider = FileConfigProvider("/nonexistent/config.json")
        configs = await provider.load_configs()
        
        assert configs == []
    
    @pytest.mark.asyncio
    async def test_load_configs_invalid_json(self):
        """Test configuration loading with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_file = f.name
        
        try:
            provider = FileConfigProvider(temp_file)
            configs = await provider.load_configs()
            
            assert configs == []
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_save_configs_success(self):
        """Test successful configuration saving to file."""
        configs = [
            MCPServerConfig(
                name="test_server",
                server_type=ServerType.HTTP,
                host="test.com",
                port=8080,
                protocol="https"
            )
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            provider = FileConfigProvider(temp_file)
            success = await provider.save_configs(configs)
            
            assert success is True
            
            # Verify file content
            with open(temp_file, 'r') as f:
                saved_data = json.load(f)
            
            assert "servers" in saved_data
            assert len(saved_data["servers"]) == 1
            assert saved_data["servers"][0]["name"] == "test_server"
            assert saved_data["servers"][0]["host"] == "test.com"
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_save_configs_write_error(self):
        """Test configuration saving with write error."""
        configs = [
            MCPServerConfig(
                name="test_server",
                server_type=ServerType.HTTP,
                host="test.com",
                port=8080
            )
        ]
        
        # Use invalid path
        provider = FileConfigProvider("/invalid/path/config.json")
        success = await provider.save_configs(configs)
        
        assert success is False


class TestEnvironmentConfigProvider:
    """Test cases for EnvironmentConfigProvider class."""
    
    def test_provider_creation(self):
        """Test environment config provider creation."""
        provider = EnvironmentConfigProvider()
        
        assert provider.name == "environment_provider"
        assert provider.prefix == "ARK_MCP_"
    
    def test_provider_creation_with_prefix(self):
        """Test environment config provider creation with custom prefix."""
        provider = EnvironmentConfigProvider(prefix="CUSTOM_")
        
        assert provider.prefix == "CUSTOM_"
    
    @pytest.mark.asyncio
    async def test_load_configs_from_environment(self):
        """Test configuration loading from environment variables."""
        env_vars = {
            "ARK_MCP_SERVER1_HOST": "env1.example.com",
            "ARK_MCP_SERVER1_PORT": "8080",
            "ARK_MCP_SERVER1_PROTOCOL": "https",
            "ARK_MCP_SERVER1_ENABLED": "true",
            "ARK_MCP_SERVER2_HOST": "env2.example.com",
            "ARK_MCP_SERVER2_PORT": "9090",
            "ARK_MCP_SERVER2_PROTOCOL": "http",
            "ARK_MCP_SERVER2_ENABLED": "false"
        }
        
        with patch.dict(os.environ, env_vars):
            provider = EnvironmentConfigProvider(prefix="ARK_MCP_")
            configs = await provider.load_configs()
            
            assert len(configs) == 2
            
            # Check first server
            server1 = next(c for c in configs if c.name == "server1")
            assert server1.host == "env1.example.com"
            assert server1.port == 8080
            assert server1.protocol == "https"
            assert server1.enabled is True
            
            # Check second server
            server2 = next(c for c in configs if c.name == "server2")
            assert server2.host == "env2.example.com"
            assert server2.port == 9090
            assert server2.protocol == "http"
            assert server2.enabled is False
    
    @pytest.mark.asyncio
    async def test_load_configs_no_environment_vars(self):
        """Test configuration loading with no environment variables."""
        provider = EnvironmentConfigProvider()
        configs = await provider.load_configs()
        
        assert configs == []
    
    @pytest.mark.asyncio
    async def test_load_configs_invalid_values(self):
        """Test configuration loading with invalid environment values."""
        env_vars = {
            "ARK_MCP_SERVER1_NAME": "invalid_server",
            "ARK_MCP_SERVER1_HOST": "example.com",
            "ARK_MCP_SERVER1_PORT": "invalid_port",  # Invalid port
            "ARK_MCP_SERVER1_PROTOCOL": "https"
        }
        
        with patch.dict(os.environ, env_vars):
            provider = EnvironmentConfigProvider()
            configs = await provider.load_configs()
            
            # Should skip invalid configurations
            assert configs == []
    
    @pytest.mark.asyncio
    async def test_save_configs_not_supported(self):
        """Test that saving configs is not supported for environment provider."""
        provider = EnvironmentConfigProvider()
        configs = [MCPServerConfig(name="test", server_type=ServerType.HTTP, host="host", port=8080)]
        
        success = await provider.save_configs(configs)
        
        assert success is False


class TestARKServerConfigManager:
    """Test cases for ARKServerConfigManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create a config manager for testing."""
        return ARKServerConfigManager()
    
    @pytest.fixture
    def sample_configs(self):
        """Create sample configurations for testing."""
        return [
            MCPServerConfig(
                name="weather_server",
                server_type=ServerType.HTTP,
                host="weather.api.com",
                port=443,
                protocol="https",
                enabled=True,
                security_level=SecurityLevel.HIGH,
                config_type=ConfigType.PRODUCTION
            ),
            MCPServerConfig(
                name="time_server",
                server_type=ServerType.HTTP,
                host="time.api.com",
                port=80,
                protocol="http",
                enabled=True,
                security_level=SecurityLevel.MEDIUM,
                config_type=ConfigType.DEVELOPMENT
            ),
            MCPServerConfig(
                name="disabled_server",
                server_type=ServerType.HTTP,
                host="disabled.api.com",
                port=8080,
                protocol="http",
                enabled=False,
                security_level=SecurityLevel.LOW,
                config_type=ConfigType.TESTING
            )
        ]
    
    def test_manager_creation(self, manager):
        """Test config manager creation."""
        assert manager.configs == {}
        assert manager.providers == []
    
    def test_add_config(self, manager, sample_configs):
        """Test adding configuration."""
        config = sample_configs[0]
        manager.add_config(config)
        
        assert "weather_server" in manager.configs
        assert manager.configs["weather_server"] == config
    
    def test_add_duplicate_config(self, manager, sample_configs):
        """Test adding duplicate configuration."""
        config = sample_configs[0]
        manager.add_config(config)
        
        # Add same config again
        manager.add_config(config)
        
        # Should still have only one entry
        assert len(manager.configs) == 1
        assert manager.configs["weather_server"] == config
    
    def test_remove_config(self, manager, sample_configs):
        """Test removing configuration."""
        config = sample_configs[0]
        manager.add_config(config)
        
        assert "weather_server" in manager.configs
        
        manager.remove_config("weather_server")
        
        assert "weather_server" not in manager.configs
    
    def test_remove_nonexistent_config(self, manager):
        """Test removing nonexistent configuration."""
        # Should not raise exception
        manager.remove_config("nonexistent_server")
        assert len(manager.configs) == 0
    
    def test_get_config(self, manager, sample_configs):
        """Test getting configuration by name."""
        config = sample_configs[0]
        manager.add_config(config)
        
        retrieved_config = manager.get_config("weather_server")
        assert retrieved_config == config
        
        # Test nonexistent config
        assert manager.get_config("nonexistent_server") is None
    
    def test_list_configs_no_filter(self, manager, sample_configs):
        """Test listing all configurations without filter."""
        for config in sample_configs:
            manager.add_config(config)
        
        configs = manager.list_configs()
        
        assert len(configs) == 3
        config_names = [config.name for config in configs]
        assert "weather_server" in config_names
        assert "time_server" in config_names
        assert "disabled_server" in config_names
    
    def test_list_configs_enabled_only(self, manager, sample_configs):
        """Test listing only enabled configurations."""
        for config in sample_configs:
            manager.add_config(config)
        
        configs = manager.list_configs(enabled_only=True)
        
        assert len(configs) == 2
        config_names = [config.name for config in configs]
        assert "weather_server" in config_names
        assert "time_server" in config_names
        assert "disabled_server" not in config_names
    
    def test_list_configs_by_type(self, manager, sample_configs):
        """Test listing configurations by type."""
        for config in sample_configs:
            manager.add_config(config)
        
        configs = manager.list_configs(config_type=ConfigType.PRODUCTION)
        
        assert len(configs) == 1
        assert configs[0].name == "weather_server"
        assert configs[0].config_type == ConfigType.PRODUCTION
    
    def test_list_configs_by_security_level(self, manager, sample_configs):
        """Test listing configurations by security level."""
        for config in sample_configs:
            manager.add_config(config)
        
        configs = manager.list_configs(security_level=SecurityLevel.HIGH)
        
        assert len(configs) == 1
        assert configs[0].name == "weather_server"
        assert configs[0].security_level == SecurityLevel.HIGH
    
    def test_update_config(self, manager, sample_configs):
        """Test updating configuration."""
        config = sample_configs[0]
        manager.add_config(config)
        
        original_created_at = config.created_at
        
        # Update configuration
        success = manager.update_config("weather_server", host="new.weather.com", port=9443)
        
        assert success is True
        
        updated_config = manager.get_config("weather_server")
        assert updated_config.host == "new.weather.com"
        assert updated_config.port == 9443
        assert updated_config.created_at == original_created_at
        assert updated_config.last_updated is not None
    
    def test_update_nonexistent_config(self, manager):
        """Test updating nonexistent configuration."""
        success = manager.update_config("nonexistent_server", host="new.host.com")
        
        assert success is False
    
    def test_validate_config(self, manager, sample_configs):
        """Test configuration validation."""
        config = sample_configs[0]
        
        is_valid, errors = manager.validate_config(config)
        
        assert is_valid is True
        assert errors == []
    
    def test_validate_invalid_config(self, manager):
        """Test validation of invalid configuration."""
        invalid_config = MCPServerConfig(
            name="",  # Invalid name
            server_type=ServerType.HTTP,
            host="example.com",
            port=0  # Invalid port
        )
        
        is_valid, errors = manager.validate_config(invalid_config)
        
        assert is_valid is False
        assert len(errors) > 0
    
    def test_add_provider(self, manager):
        """Test adding configuration provider."""
        mock_provider = Mock(spec=ServerConfigProvider)
        mock_provider.name = "test_provider"
        
        manager.add_provider(mock_provider)
        
        assert mock_provider in manager.providers
        assert len(manager.providers) == 1
    
    def test_remove_provider(self, manager):
        """Test removing configuration provider."""
        mock_provider = Mock(spec=ServerConfigProvider)
        mock_provider.name = "test_provider"
        
        manager.add_provider(mock_provider)
        assert mock_provider in manager.providers
        
        manager.remove_provider("test_provider")
        assert mock_provider not in manager.providers
    
    @pytest.mark.asyncio
    async def test_load_from_providers(self, manager):
        """Test loading configurations from providers."""
        # Create mock provider
        mock_provider = Mock(spec=ServerConfigProvider)
        mock_provider.name = "test_provider"
        mock_provider.load_configs = AsyncMock(return_value=[
            MCPServerConfig(
                name="provider_server",
                server_type=ServerType.HTTP,
                host="provider.com",
                port=8080
            )
        ])
        
        manager.add_provider(mock_provider)
        
        loaded_configs = await manager.load_from_providers()
        
        assert len(loaded_configs) == 1
        assert loaded_configs[0].name == "provider_server"
        
        # Config should also be added to manager
        assert "provider_server" in manager.configs
    
    @pytest.mark.asyncio
    async def test_load_from_providers_error(self, manager):
        """Test loading configurations with provider error."""
        mock_provider = Mock(spec=ServerConfigProvider)
        mock_provider.name = "error_provider"
        mock_provider.load_configs = AsyncMock(side_effect=Exception("Load error"))
        
        manager.add_provider(mock_provider)
        
        # Should handle error gracefully
        loaded_configs = await manager.load_from_providers()
        assert loaded_configs == []
    
    @pytest.mark.asyncio
    async def test_save_to_providers(self, manager, sample_configs):
        """Test saving configurations to providers."""
        for config in sample_configs:
            manager.add_config(config)
        
        # Create mock provider
        mock_provider = Mock(spec=ServerConfigProvider)
        mock_provider.name = "test_provider"
        mock_provider.save_configs = AsyncMock(return_value=True)
        
        manager.add_provider(mock_provider)
        
        success = await manager.save_to_providers()
        
        assert success is True
        mock_provider.save_configs.assert_called_once()
        
        # Check that all configs were passed to provider
        saved_configs = mock_provider.save_configs.call_args[0][0]
        assert len(saved_configs) == 3
    
    @pytest.mark.asyncio
    async def test_save_to_providers_error(self, manager, sample_configs):
        """Test saving configurations with provider error."""
        for config in sample_configs:
            manager.add_config(config)
        
        mock_provider = Mock(spec=ServerConfigProvider)
        mock_provider.name = "error_provider"
        mock_provider.save_configs = AsyncMock(side_effect=Exception("Save error"))
        
        manager.add_provider(mock_provider)
        
        # Should handle error gracefully
        success = await manager.save_to_providers()
        assert success is False
    
    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self, manager, sample_configs):
        """Test health check with all healthy servers."""
        for config in sample_configs[:2]:  # Only enabled servers
            manager.add_config(config)
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Mock successful responses
            mock_response = Mock()
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response
            
            health_status = await manager.health_check()
            
            assert len(health_status) == 2
            assert all(status["healthy"] for status in health_status.values())
            assert "weather_server" in health_status
            assert "time_server" in health_status
    
    @pytest.mark.asyncio
    async def test_health_check_some_unhealthy(self, manager, sample_configs):
        """Test health check with some unhealthy servers."""
        for config in sample_configs[:2]:  # Only enabled servers
            manager.add_config(config)
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            # Create a mock session
            mock_session = Mock()
            
            # Mock the session context manager
            mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Mock the get method to return different responses based on URL
            def mock_get_side_effect(*args, **kwargs):
                url = args[0] if args else kwargs.get('url', '')
                
                mock_response = Mock()
                if 'weather' in str(url):
                    mock_response.status = 200  # Healthy
                else:
                    mock_response.status = 500  # Unhealthy
                
                # Create a mock context manager for the response
                mock_response_context = Mock()
                mock_response_context.__aenter__ = AsyncMock(return_value=mock_response)
                mock_response_context.__aexit__ = AsyncMock(return_value=None)
                
                return mock_response_context
            
            mock_session.get.side_effect = mock_get_side_effect
            
            health_status = await manager.health_check()
            
            assert len(health_status) == 2
            assert health_status["weather_server"]["healthy"] is True
            assert health_status["time_server"]["healthy"] is False
    
    @pytest.mark.asyncio
    async def test_health_check_connection_error(self, manager, sample_configs):
        """Test health check with connection errors."""
        config = sample_configs[0]
        manager.add_config(config)
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Mock connection error
            mock_get.side_effect = Exception("Connection failed")
            
            health_status = await manager.health_check()
            
            assert len(health_status) == 1
            assert health_status["weather_server"]["healthy"] is False
            assert "error" in health_status["weather_server"]
    
    def test_get_config_count(self, manager, sample_configs):
        """Test getting configuration count."""
        assert manager.get_config_count() == 0
        
        for config in sample_configs:
            manager.add_config(config)
        
        assert manager.get_config_count() == 3
        
        # Test with enabled only
        enabled_count = manager.get_config_count(enabled_only=True)
        assert enabled_count == 2
    
    def test_get_config_types(self, manager, sample_configs):
        """Test getting available configuration types."""
        for config in sample_configs:
            manager.add_config(config)
        
        config_types = manager.get_config_types()
        
        assert ConfigType.PRODUCTION in config_types
        assert ConfigType.DEVELOPMENT in config_types
        assert ConfigType.TESTING in config_types
        assert len(config_types) == 3
    
    def test_get_security_levels(self, manager, sample_configs):
        """Test getting available security levels."""
        for config in sample_configs:
            manager.add_config(config)
        
        security_levels = manager.get_security_levels()
        
        assert SecurityLevel.HIGH in security_levels
        assert SecurityLevel.MEDIUM in security_levels
        assert SecurityLevel.LOW in security_levels
        assert len(security_levels) == 3
    
    def test_clear_configs(self, manager, sample_configs):
        """Test clearing all configurations."""
        for config in sample_configs:
            manager.add_config(config)
        
        assert len(manager.configs) == 3
        
        manager.clear_configs()
        
        assert len(manager.configs) == 0


class TestServerConfigIntegration:
    """Integration tests for server configuration functionality."""
    
    @pytest.mark.asyncio
    async def test_full_config_management_flow(self):
        """Test complete configuration management flow."""
        manager = ARKServerConfigManager()
        
        # Create file provider
        configs_data = [
            {
                "name": "file_server",
                "server_type": "http",
                "host": "file.example.com",
                "port": 443,
                "protocol": "https",
                "enabled": True
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(configs_data, f)
            temp_file = f.name
        
        try:
            file_provider = FileConfigProvider(temp_file)
            manager.add_provider(file_provider)
            
            # Create environment provider
            env_vars = {
                "ARK_MCP_ENVSERVER_HOST": "env.example.com",
                "ARK_MCP_ENVSERVER_PORT": "8080",
                "ARK_MCP_ENVSERVER_PROTOCOL": "http",
                "ARK_MCP_ENVSERVER_ENABLED": "true"
            }
            
            with patch.dict(os.environ, env_vars):
                env_provider = EnvironmentConfigProvider(prefix="ARK_MCP_")
                manager.add_provider(env_provider)
                
                # Load configurations
                loaded_configs = await manager.load_from_providers()
                
                # Should have configs from both providers
                assert len(loaded_configs) == 2
                config_names = [config.name for config in loaded_configs]
                assert "file_server" in config_names
                assert "envserver" in config_names
                
                # Configs should be in manager
                assert len(manager.configs) == 2
                assert manager.get_config("file_server") is not None
                assert manager.get_config("envserver") is not None
                
                # Update a configuration
                success = manager.update_config("file_server", port=9443)
                assert success is True
                
                updated_config = manager.get_config("file_server")
                assert updated_config.port == 9443
                
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_performance_with_many_configs(self):
        """Test manager performance with many configurations."""
        manager = ARKServerConfigManager()
        
        # Add many configurations
        configs = []
        for i in range(1000):
            config = MCPServerConfig(
                name=f"server_{i}",
                server_type=ServerType.HTTP,
                host=f"server{i}.example.com",
                port=8000 + i,
                protocol="https",
                enabled=i % 2 == 0,  # Half enabled, half disabled
                security_level=SecurityLevel.MEDIUM,
                config_type=ConfigType.DEVELOPMENT
            )
            configs.append(config)
            manager.add_config(config)
        
        # Test listing performance
        import time
        start_time = time.time()
        
        all_configs = manager.list_configs()
        assert len(all_configs) == 1000
        
        # Test filtering performance
        enabled_configs = manager.list_configs(enabled_only=True)
        assert len(enabled_configs) == 500
        
        production_configs = manager.list_configs(config_type=ConfigType.PRODUCTION)
        assert len(production_configs) == 0  # None are production
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should be fast even with many configs
        assert processing_time < 1.0  # Less than 1 second


if __name__ == "__main__":
    pytest.main([__file__, "-v"])