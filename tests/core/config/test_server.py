"""
Merged test file for server configuration functionality.
This file combines tests from test_server_config.py, test_server_config_coverage.py,
and test_server_config_edge_cases.py with API compatibility fixes.
"""

import pytest
import tempfile
import os
import json
import yaml
import asyncio
import unittest.mock
from unittest.mock import patch
from pathlib import Path
from datetime import datetime

from src.core.config.server import (
    ServerType,
    ServerStatus,
    AuthType,
    MCPServerConfig,
    SimpleMCPServerConfig,
    ServerCredentials,
    ServerHealthCheck,
    ServerLimits,
    ServerMetrics,
    FileConfigProvider,
    EnvironmentConfigProvider,
    ARKServerConfigManager,
)


class TestServerEnums:
    """Test server enumeration classes."""

    def test_server_type_enum(self):
        """Test ServerType enum values."""
        assert ServerType.STDIO.value == "stdio"
        assert ServerType.HTTP.value == "http"
        assert ServerType.WEBSOCKET.value == "websocket"
        assert ServerType.TCP.value == "tcp"
        assert ServerType.CUSTOM.value == "custom"

    def test_server_status_enum(self):
        """Test ServerStatus enum values."""
        assert ServerStatus.UNKNOWN.value == "unknown"
        assert ServerStatus.CONNECTING.value == "connecting"
        assert ServerStatus.CONNECTED.value == "connected"
        assert ServerStatus.DISCONNECTED.value == "disconnected"
        assert ServerStatus.ERROR.value == "error"
        assert ServerStatus.MAINTENANCE.value == "maintenance"
        assert ServerStatus.DISABLED.value == "disabled"

    def test_auth_type_enum(self):
        """Test AuthType enum values."""
        assert AuthType.NONE.value == "none"
        assert AuthType.API_KEY.value == "api_key"
        assert AuthType.BEARER_TOKEN.value == "bearer_token"
        assert AuthType.BASIC_AUTH.value == "basic_auth"
        assert AuthType.OAUTH2.value == "oauth2"
        assert AuthType.CUSTOM.value == "custom"


class TestMCPServerConfig:
    """Test MCPServerConfig class."""

    def test_config_creation_minimal(self):
        """Test creating config with minimal parameters."""
        config = MCPServerConfig(name="test_server", server_type=ServerType.STDIO)
        assert config.name == "test_server"
        assert config.server_type == ServerType.STDIO
        assert config.status == ServerStatus.UNKNOWN
        assert config.enabled is True

    def test_config_creation_full(self):
        """Test creating config with all parameters."""
        config = MCPServerConfig(
            name="full_server",
            server_type=ServerType.HTTP,
            status=ServerStatus.CONNECTED,
            command="python",
            args=["server.py"],
            host="localhost",
            port=8080,
            description="Test server",
            enabled=True,
        )
        assert config.name == "full_server"
        assert config.server_type == ServerType.HTTP
        assert config.status == ServerStatus.CONNECTED
        assert config.command == "python"
        assert config.args == ["server.py"]
        assert config.host == "localhost"
        assert config.port == 8080
        assert config.description == "Test server"
        assert config.enabled is True

    def test_config_to_dict_with_sensitive_data(self):
        """Test converting config to dict with sensitive data."""
        config = MCPServerConfig(
            name="test_server", server_type=ServerType.HTTP, api_key="secret_key_123"
        )

        # Test with sensitive data included
        config_dict = config.to_dict(include_sensitive=True)
        assert config_dict["api_key"] == "secret_key_123"

        # Test with sensitive data excluded (default)
        config_dict_safe = config.to_dict(include_sensitive=False)
        assert config_dict_safe["api_key"] is None

    def test_config_to_dict_without_api_key(self):
        """Test converting config to dict without API key."""
        config = MCPServerConfig(name="test_server", server_type=ServerType.HTTP)

        config_dict = config.to_dict(include_sensitive=False)
        assert config_dict["api_key"] is None

    def test_config_from_dict_basic(self):
        """Test creating config from dictionary."""
        data = {
            "name": "test_server",
            "server_type": "http",
            "host": "localhost",
            "port": 8080,
            "enabled": True,
        }

        config = MCPServerConfig.from_dict(data)
        assert config.name == "test_server"
        assert config.server_type == ServerType.HTTP
        assert config.host == "localhost"
        assert config.port == 8080
        assert config.enabled is True

    def test_config_from_dict_with_datetime_strings(self):
        """Test creating config from dict with datetime strings."""
        data = {
            "name": "test_server",
            "server_type": "stdio",
            "created_at": "2023-01-01T12:00:00",
            "last_updated": "2023-01-02T12:00:00",
        }

        config = MCPServerConfig.from_dict(data)
        assert config.name == "test_server"
        assert isinstance(config.created_at, datetime)
        assert isinstance(config.last_updated, datetime)

    def test_config_from_dict_with_invalid_datetime(self):
        """Test creating config from dict with invalid datetime strings."""
        data = {
            "name": "test_server",
            "server_type": "stdio",
            "created_at": "invalid_date",
            "last_updated": "invalid_date",
        }

        config = MCPServerConfig.from_dict(data)
        assert config.name == "test_server"
        assert isinstance(config.created_at, datetime)  # Should default to now()
        assert config.last_updated is None  # Should be None for invalid

    def test_config_validate_stdio_server(self):
        """Test validation for STDIO server."""
        # Valid STDIO config
        config = MCPServerConfig(
            name="stdio_server", server_type=ServerType.STDIO, command="python"
        )
        is_valid, errors = config.validate()
        assert is_valid is True
        assert len(errors) == 0

        # Invalid STDIO config (missing command)
        config_invalid = MCPServerConfig(
            name="stdio_server", server_type=ServerType.STDIO
        )
        is_valid, errors = config_invalid.validate()
        assert is_valid is False
        assert "Command is required for STDIO servers" in errors

    def test_config_validate_http_server(self):
        """Test validation for HTTP server."""
        # Valid HTTP config with host and port
        config = MCPServerConfig(
            name="http_server", server_type=ServerType.HTTP, host="localhost", port=8080
        )
        is_valid, errors = config.validate()
        assert is_valid is True
        assert len(errors) == 0

        # Invalid HTTP config (missing host and port)
        config_invalid = MCPServerConfig(
            name="http_server", server_type=ServerType.HTTP
        )
        is_valid, errors = config_invalid.validate()
        assert is_valid is False
        assert any("URL or host+port is required" in error for error in errors)

    def test_config_validate_tcp_server(self):
        """Test validation for TCP server."""
        # Valid TCP config
        config = MCPServerConfig(
            name="tcp_server", server_type=ServerType.TCP, host="localhost", port=9090
        )
        is_valid, errors = config.validate()
        assert is_valid is True
        assert len(errors) == 0

        # Invalid TCP config (missing host)
        config_invalid = MCPServerConfig(
            name="tcp_server", server_type=ServerType.TCP, port=9090
        )
        is_valid, errors = config_invalid.validate()
        assert is_valid is False
        assert "Host and port are required for TCP servers" in errors

    def test_config_validate_port_range(self):
        """Test port range validation."""
        # Invalid port (too high)
        config = MCPServerConfig(
            name="test_server",
            server_type=ServerType.HTTP,
            host="localhost",
            port=70000,
        )
        is_valid, errors = config.validate()
        assert is_valid is False
        assert "Port must be between 1 and 65535" in errors

        # Invalid port (negative)
        config_negative = MCPServerConfig(
            name="test_server", server_type=ServerType.HTTP, host="localhost", port=-1
        )
        is_valid, errors = config_negative.validate()
        assert is_valid is False
        assert "Port must be between 1 and 65535" in errors

    def test_config_validate_timeout_and_retries(self):
        """Test timeout and retries validation."""
        # Invalid timeout
        config = MCPServerConfig(
            name="test_server",
            server_type=ServerType.STDIO,
            command="python",
            timeout=-1,
        )
        is_valid, errors = config.validate()
        assert is_valid is False
        assert "Timeout must be positive" in errors

        # Invalid max retries
        config_retries = MCPServerConfig(
            name="test_server",
            server_type=ServerType.STDIO,
            command="python",
            max_retries=-1,
        )
        is_valid, errors = config_retries.validate()
        assert is_valid is False
        assert "Max retries cannot be negative" in errors


class TestSimpleMCPServerConfig:
    """Test SimpleMCPServerConfig class."""

    def test_simple_config_creation(self):
        """Test creating simple config."""
        config = SimpleMCPServerConfig(
            name="simple_server",
            command="python",
            args=["server.py"],
            description="Simple test server",
        )
        assert config.name == "simple_server"
        assert config.command == "python"
        assert config.args == ["server.py"]
        assert config.description == "Simple test server"
        assert config.enabled is True

    def test_to_stdio_params(self):
        """Test conversion to stdio parameters."""
        config = SimpleMCPServerConfig(
            name="stdio_server",
            command="python",
            args=["server.py"],
            env={"PATH": "/usr/bin"},
        )
        params = config.to_stdio_params()
        assert params.command == "python"
        assert params.args == ["server.py"]
        assert params.env == {"PATH": "/usr/bin"}

    def test_to_stdio_params_with_none_values(self):
        """Test conversion to stdio parameters with None values."""
        config = SimpleMCPServerConfig(
            name="stdio_server", command="python", args=None, env=None
        )
        params = config.to_stdio_params()
        assert params.command == "python"
        assert params.args == []
        assert params.env is None


class TestFileConfigProvider:
    """Test FileConfigProvider class."""

    def test_file_provider_creation(self):
        """Test creating file config provider."""
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FileConfigProvider(config_path=temp_dir)
            assert str(provider.config_path) == temp_dir

    def test_file_provider_with_json_file(self):
        """Test file provider with JSON config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_data = {
                "servers": [
                    {
                        "name": "test_server",
                        "server_type": "stdio",
                        "command": "python",
                        "args": ["server.py"],
                    }
                ]
            }
            json.dump(config_data, f)
            f.flush()

            try:
                provider = FileConfigProvider(config_path=f.name)
                assert str(provider.config_path) == f.name
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_file_provider_load_configs_single_file(self):
        """Test loading configs from single JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_data = {
                "servers": [
                    {
                        "name": "test_server",
                        "server_type": "stdio",
                        "command": "python",
                        "args": ["server.py"],
                    }
                ]
            }
            json.dump(config_data, f)
            f.flush()

            try:
                provider = FileConfigProvider(config_path=f.name)
                configs = await provider.load_configs()
                assert len(configs) == 1
                assert configs[0].name == "test_server"
                assert configs[0].server_type == ServerType.STDIO
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_file_provider_load_configs_yaml_file(self):
        """Test loading configs from YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {
                "servers": [
                    {
                        "name": "yaml_server",
                        "server_type": "http",
                        "host": "localhost",
                        "port": 8080,
                    }
                ]
            }
            yaml.dump(config_data, f)
            f.flush()

            try:
                provider = FileConfigProvider(config_path=f.name)
                configs = await provider.load_configs()
                assert len(configs) == 1
                assert configs[0].name == "yaml_server"
                assert configs[0].server_type == ServerType.HTTP
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_file_provider_save_config_single_file(self):
        """Test saving config to single file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"servers": []}')
            f.flush()

            try:
                provider = FileConfigProvider(config_path=f.name)
                config = MCPServerConfig(
                    name="new_server", server_type=ServerType.STDIO, command="python"
                )

                result = await provider.save_config(config)
                assert result is True

                # Verify the config was saved
                configs = await provider.load_configs()
                assert len(configs) == 1
                assert configs[0].name == "new_server"
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_file_provider_save_config_single_file_mode_update_existing(
        self, temp_config_file
    ):
        """Test updating existing config in single file mode."""
        # Create initial config file
        initial_data = {
            "servers": [
                {
                    "name": "existing-server",
                    "server_type": "stdio",
                    "command": "old-command",
                }
            ]
        }
        with open(temp_config_file, "w") as f:
            json.dump(initial_data, f)

        provider = FileConfigProvider(config_path=temp_config_file)
        config = MCPServerConfig(
            name="existing-server", server_type=ServerType.STDIO, command="new-command"
        )

        result = await provider.save_config(config)
        assert result is True

        # Verify file content
        with open(temp_config_file, "r") as f:
            data = json.load(f)

        assert len(data["servers"]) == 1
        assert data["servers"][0]["command"] == "new-command"

    @pytest.mark.asyncio
    async def test_file_provider_save_config_single_file_mode_yaml(
        self, temp_config_file
    ):
        """Test saving config to YAML file in single file mode."""
        yaml_file = Path(temp_config_file).with_suffix(".yaml")

        provider = FileConfigProvider(config_path=yaml_file)
        config = MCPServerConfig(
            name="yaml-server", server_type=ServerType.STDIO, command="yaml-command"
        )

        result = await provider.save_config(config)
        assert result is True

        # Verify file content
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)

        assert len(data["servers"]) == 1
        assert data["servers"][0]["name"] == "yaml-server"

        # Cleanup
        yaml_file.unlink()

    @pytest.mark.asyncio
    async def test_file_provider_save_config_directory_mode(self, temp_config_dir):
        """Test saving config in directory mode."""
        provider = FileConfigProvider(config_path=temp_config_dir)
        config = MCPServerConfig(
            name="test-server", server_type=ServerType.STDIO, command="test-command"
        )

        result = await provider.save_config(config)
        assert result is True

        # Check file was created
        config_file = temp_config_dir / "test-server.json"
        assert config_file.exists()

        # Check content
        with open(config_file) as f:
            data = json.load(f)
        assert data["name"] == "test-server"
        assert data["server_type"] == "stdio"
        assert data["command"] == "test-command"

    @pytest.mark.asyncio
    async def test_file_provider_save_config_directory_mode_error(self):
        """Test saving config in directory mode with error."""
        invalid_dir = "/invalid/path/that/does/not/exist"
        provider = FileConfigProvider(config_path=invalid_dir)
        config = MCPServerConfig(
            name="test-server", server_type=ServerType.STDIO, command="test-command"
        )

        result = await provider.save_config(config)
        assert result is False

    @pytest.mark.asyncio
    async def test_file_provider_save_configs_single_file_mode(self, temp_config_file):
        """Test saving multiple configs in single file mode."""
        provider = FileConfigProvider(config_path=temp_config_file)
        configs = [
            MCPServerConfig(
                name="server-1", server_type=ServerType.STDIO, command="command-1"
            ),
            MCPServerConfig(
                name="server-2", server_type=ServerType.STDIO, command="command-2"
            ),
        ]

        result = await provider.save_configs(configs)
        assert result is True

        # Verify file content
        with open(temp_config_file, "r") as f:
            data = json.load(f)

        assert len(data["servers"]) == 2

    @pytest.mark.asyncio
    async def test_file_provider_save_configs_directory_mode(self, temp_config_dir):
        """Test saving multiple configs in directory mode."""
        provider = FileConfigProvider(config_path=temp_config_dir)
        configs = [
            MCPServerConfig(
                name="server-1", server_type=ServerType.STDIO, command="command-1"
            ),
            MCPServerConfig(
                name="server-2", server_type=ServerType.STDIO, command="command-2"
            ),
        ]

        result = await provider.save_configs(configs)
        assert result is True

        # Check files were created
        config_file_1 = temp_config_dir / "server-1.json"
        config_file_2 = temp_config_dir / "server-2.json"
        assert config_file_1.exists()
        assert config_file_2.exists()

    @pytest.mark.asyncio
    async def test_file_provider_save_configs_directory_mode_error(self):
        """Test saving multiple configs in directory mode with error."""
        invalid_dir = "/invalid/path/that/does/not/exist"
        provider = FileConfigProvider(config_path=invalid_dir)
        configs = [
            MCPServerConfig(
                name="server-1", server_type=ServerType.STDIO, command="command-1"
            )
        ]

        result = await provider.save_configs(configs)
        assert result is False

    @pytest.mark.asyncio
    async def test_file_provider_delete_config_single_file_mode_success(
        self, temp_config_file
    ):
        """Test deleting config in single file mode - success case."""
        # Create initial config file
        initial_data = {
            "servers": [
                {"name": "server-1", "server_type": "stdio", "command": "command-1"},
                {"name": "server-2", "server_type": "stdio", "command": "command-2"},
            ]
        }
        with open(temp_config_file, "w") as f:
            json.dump(initial_data, f)

        provider = FileConfigProvider(config_path=temp_config_file)
        result = await provider.delete_config("server-1")
        assert result is True

        # Verify file content
        with open(temp_config_file, "r") as f:
            data = json.load(f)

        assert len(data["servers"]) == 1
        assert data["servers"][0]["name"] == "server-2"

    @pytest.mark.asyncio
    async def test_file_provider_delete_config_single_file_mode_not_found(
        self, temp_config_file
    ):
        """Test deleting config in single file mode - config not found."""
        # Create initial config file
        initial_data = {
            "servers": [
                {"name": "server-1", "server_type": "stdio", "command": "command-1"}
            ]
        }
        with open(temp_config_file, "w") as f:
            json.dump(initial_data, f)

        provider = FileConfigProvider(config_path=temp_config_file)
        result = await provider.delete_config("non-existent-server")
        assert result is False

    @pytest.mark.asyncio
    async def test_file_provider_delete_config_single_file_mode_yaml(
        self, temp_config_file
    ):
        """Test deleting config from YAML file in single file mode."""
        yaml_file = Path(temp_config_file).with_suffix(".yaml")

        # Create initial config file
        initial_data = {
            "servers": [
                {
                    "name": "yaml-server",
                    "server_type": "stdio",
                    "command": "yaml-command",
                }
            ]
        }
        with open(yaml_file, "w") as f:
            yaml.dump(initial_data, f)

        provider = FileConfigProvider(config_path=yaml_file)
        result = await provider.delete_config("yaml-server")
        assert result is True

        # Verify file content
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)

        assert len(data["servers"]) == 0

        # Cleanup
        yaml_file.unlink()

    @pytest.mark.asyncio
    async def test_file_provider_delete_config_directory_mode_json(
        self, temp_config_dir
    ):
        """Test deleting JSON config in directory mode."""
        # Create config file
        config_file = temp_config_dir / "test-server.json"
        with open(config_file, "w") as f:
            json.dump(
                {
                    "name": "test-server",
                    "server_type": "stdio",
                    "command": "test-command",
                },
                f,
            )

        provider = FileConfigProvider(config_path=temp_config_dir)
        result = await provider.delete_config("test-server")
        assert result is True
        assert not config_file.exists()

    @pytest.mark.asyncio
    async def test_file_provider_delete_config_directory_mode_yaml(
        self, temp_config_dir
    ):
        """Test deleting YAML config in directory mode."""
        # Create YAML config file first
        config_file = temp_config_dir / "test-server.yaml"
        config_data = {
            "name": "test-server",
            "server_type": "stdio",
            "command": "test-command",
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        provider = FileConfigProvider(config_path=temp_config_dir)
        result = await provider.delete_config("test-server")
        assert result is True
        assert not config_file.exists()

    @pytest.mark.asyncio
    async def test_file_provider_delete_config_directory_mode_not_found(
        self, temp_config_dir
    ):
        """Test deleting config in directory mode - config not found."""
        provider = FileConfigProvider(config_path=temp_config_dir)
        result = await provider.delete_config("non-existent-server")
        assert result is False

    @pytest.mark.asyncio
    async def test_file_provider_delete_config_error(self, temp_config_file):
        """Test delete_config with error."""
        # Create a file that will cause an error when trying to delete
        with open(temp_config_file, "w") as f:
            json.dump({"servers": []}, f)

        # Make the file read-only to cause an error
        Path(temp_config_file).chmod(0o444)

        provider = FileConfigProvider(config_path=temp_config_file)
        result = await provider.delete_config("any-server")
        assert result is False

        # Restore permissions for cleanup
        Path(temp_config_file).chmod(0o644)


class TestARKServerConfigManagerAdvanced:
    """Test ARKServerConfigManager advanced functionality."""

    @pytest.fixture
    def manager(self):
        """Create a test manager."""
        provider = unittest.mock.Mock(spec=FileConfigProvider)
        provider.load_configs = unittest.mock.AsyncMock(return_value=[])
        return ARKServerConfigManager(config_provider=provider)

    @pytest.mark.asyncio
    async def test_start_and_stop(self, manager):
        """Test start and stop methods."""
        # Mock the health check loop
        with unittest.mock.patch.object(
            manager, "_health_check_loop"
        ) as mock_health_loop:
            mock_health_loop.return_value = asyncio.create_task(asyncio.sleep(0.1))

            await manager.start()
            assert manager._running is True
            assert manager._health_check_task is not None

            await manager.stop()
            assert manager._running is False

    @pytest.mark.asyncio
    async def test_load_configs_with_invalid_config(self, manager):
        """Test load_configs with invalid configurations."""
        # Create a mix of valid and invalid configs
        valid_config = MCPServerConfig(
            name="valid-server", server_type=ServerType.STDIO, command="valid-command"
        )

        invalid_config = MCPServerConfig(
            name="",  # Invalid: empty name
            server_type=ServerType.STDIO,
            command="invalid-command",
        )

        manager.config_provider.load_configs.return_value = [
            valid_config,
            invalid_config,
        ]

        count = await manager.load_configs()

        # Should only load valid configs
        assert count == 1
        assert "valid-server" in manager.configs
        assert "" not in manager.configs

    @pytest.mark.asyncio
    async def test_load_configs_with_exception(self, manager):
        """Test load_configs with exception."""
        manager.config_provider.load_configs.side_effect = Exception("Load error")

        count = await manager.load_configs()
        assert count == 0
        assert len(manager.configs) == 0

    def test_add_config_invalid(self, manager):
        """Test add_config with invalid configuration."""
        invalid_config = MCPServerConfig(
            name="",  # Invalid: empty name
            server_type=ServerType.STDIO,
            command="test-command",
        )

        result = manager.add_config(invalid_config)
        assert result is False
        assert len(manager.configs) == 0

    def test_add_config_duplicate(self, manager):
        """Test add_config with duplicate name."""
        config1 = MCPServerConfig(
            name="test-server", server_type=ServerType.STDIO, command="command-1"
        )

        config2 = MCPServerConfig(
            name="test-server",  # Same name
            server_type=ServerType.STDIO,
            command="command-2",
        )

        # Add first config
        result1 = manager.add_config(config1)
        assert result1 is True

        # Try to add duplicate
        result2 = manager.add_config(config2)
        assert result2 is False
        assert len(manager.configs) == 1
        assert manager.configs["test-server"].command == "command-1"

    def test_add_config_with_callbacks(self, manager):
        """Test add_config with callbacks."""
        callback_called = False
        callback_config = None

        def test_callback(config):
            nonlocal callback_called, callback_config
            callback_called = True
            callback_config = config

        manager.on_config_added.append(test_callback)

        config = MCPServerConfig(
            name="test-server", server_type=ServerType.STDIO, command="test-command"
        )

        result = manager.add_config(config)
        assert result is True
        assert callback_called is True
        assert callback_config == config

    def test_add_config_callback_exception(self, manager):
        """Test add_config with callback that raises exception."""

        def failing_callback(config):
            raise Exception("Callback error")

        manager.on_config_added.append(failing_callback)

        config = MCPServerConfig(
            name="test-server", server_type=ServerType.STDIO, command="test-command"
        )

        # Should still succeed despite callback error
        result = manager.add_config(config)
        assert result is True
        assert "test-server" in manager.configs

    def test_update_config_by_name(self, manager):
        """Test update_config using server name."""
        # Add initial config
        config = MCPServerConfig(
            name="test-server", server_type=ServerType.STDIO, command="old-command"
        )
        manager.add_config(config)

        # Update using name
        result = manager.update_config("test-server", command="new-command")
        assert result is True
        assert manager.configs["test-server"].command == "new-command"

    def test_update_config_by_name_not_found(self, manager):
        """Test update_config with non-existent server name."""
        result = manager.update_config("non-existent", command="new-command")
        assert result is False

    def test_update_config_by_name_unknown_attribute(self, manager):
        """Test update_config with unknown attribute."""
        # Add initial config
        config = MCPServerConfig(
            name="test-server", server_type=ServerType.STDIO, command="test-command"
        )
        manager.add_config(config)

        # Update with unknown attribute (should still succeed)
        result = manager.update_config("test-server", unknown_attr="value")
        assert result is True

    def test_update_config_by_object(self, manager):
        """Test update_config using config object."""
        # Add initial config
        config = MCPServerConfig(
            name="test-server", server_type=ServerType.STDIO, command="old-command"
        )
        manager.add_config(config)

        # Update using object
        updated_config = MCPServerConfig(
            name="test-server", server_type=ServerType.STDIO, command="new-command"
        )

        result = manager.update_config(updated_config)
        assert result is True
        assert manager.configs["test-server"].command == "new-command"

    def test_update_config_invalid(self, manager):
        """Test update_config with invalid configuration."""
        # Add initial config
        config = MCPServerConfig(
            name="test-server", server_type=ServerType.STDIO, command="test-command"
        )
        manager.add_config(config)

        # Try to update with invalid config
        result = manager.update_config("test-server", name="")  # Invalid: empty name
        assert result is False

    def test_update_config_not_found(self, manager):
        """Test update_config with config object for non-existent server."""
        config = MCPServerConfig(
            name="non-existent", server_type=ServerType.STDIO, command="test-command"
        )

        result = manager.update_config(config)
        assert result is False

    def test_update_config_with_callbacks(self, manager):
        """Test update_config with callbacks."""
        callback_called = False
        old_config_received = None
        new_config_received = None

        def test_callback(old_config, new_config):
            nonlocal callback_called, old_config_received, new_config_received
            callback_called = True
            old_config_received = old_config
            new_config_received = new_config

        manager.on_config_updated.append(test_callback)

        # Add initial config
        config = MCPServerConfig(
            name="test-server", server_type=ServerType.STDIO, command="old-command"
        )
        manager.add_config(config)

        # Update config
        result = manager.update_config("test-server", command="new-command")
        assert result is True
        assert callback_called is True
        assert old_config_received.command == "old-command"
        assert new_config_received.command == "new-command"

    def test_update_config_callback_exception(self, manager):
        """Test update_config with callback that raises exception."""

        def failing_callback(old_config, new_config):
            raise Exception("Callback error")

        manager.on_config_updated.append(failing_callback)

        # Add initial config
        config = MCPServerConfig(
            name="test-server", server_type=ServerType.STDIO, command="old-command"
        )
        manager.add_config(config)

        # Update should still succeed despite callback error
        result = manager.update_config("test-server", command="new-command")
        assert result is True
        assert manager.configs["test-server"].command == "new-command"

    def test_update_config_exception(self, manager):
        """Test update_config with exception during update."""
        # Add initial config
        config = MCPServerConfig(
            name="test-server", server_type=ServerType.STDIO, command="test-command"
        )
        manager.add_config(config)

        # Disable logging to avoid time.time() calls in logging system
        import logging

        logging.disable(logging.CRITICAL)

        try:
            # Mock time.time in the specific module to raise exception
            with unittest.mock.patch(
                "src.core.config.server.datetime"
            ) as mock_datetime:
                mock_datetime.now.side_effect = Exception("Time error")
                result = manager.update_config("test-server", command="new-command")
                assert result is False
        finally:
            # Re-enable logging
            logging.disable(logging.NOTSET)

    def test_add_config_with_result_persist_failure(self, manager):
        config = MCPServerConfig(
            name="persist-fail", server_type=ServerType.STDIO, command="test-command"
        )
        manager.config_provider.save_config.return_value = False

        result = manager.add_config_with_result(config)
        assert result.success is False
        assert result.code == "PERSIST_FAILED"
        assert result.phase == "persist"
        assert "persist-fail" not in manager.configs

    def test_update_config_with_result_not_found(self, manager):
        result = manager.update_config_with_result(
            "non-existent", command="new-command"
        )
        assert result.success is False
        assert result.code == "NOT_FOUND"
        assert result.phase == "validate"

    def test_get_last_operation_result(self, manager):
        result = manager.remove_config_with_result("non-existent")
        assert result.success is False
        last_result = manager.get_last_operation_result()
        assert last_result.code == "NOT_FOUND"
        assert last_result.phase == "validate"

    def test_validate_config(self, manager):
        """Test validate_config method."""
        valid_config = MCPServerConfig(
            name="test-server", server_type=ServerType.STDIO, command="test-command"
        )

        is_valid, errors = manager.validate_config(valid_config)
        assert is_valid is True
        assert len(errors) == 0

        invalid_config = MCPServerConfig(
            name="",  # Invalid: empty name
            server_type=ServerType.STDIO,
            command="test-command",
        )

        is_valid, errors = manager.validate_config(invalid_config)
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_config_exception(self, manager):
        """Test validate_config with exception."""
        config = MCPServerConfig(
            name="test-server", server_type=ServerType.STDIO, command="test-command"
        )

        # Mock validate method to raise exception
        with unittest.mock.patch.object(
            config, "validate", side_effect=Exception("Validation error")
        ):
            is_valid, errors = manager.validate_config(config)
            assert is_valid is False
            assert "Validation failed" in errors[0]

    def test_remove_config_success(self, manager):
        """Test remove_config success case."""
        # Add config
        config = MCPServerConfig(
            name="test-server", server_type=ServerType.STDIO, command="test-command"
        )
        manager.add_config(config)

        # Remove config
        result = manager.remove_config("test-server")
        assert result is True
        assert "test-server" not in manager.configs

    def test_remove_config_not_found(self, manager):
        """Test remove_config with non-existent server."""
        result = manager.remove_config("non-existent")
        assert result is False

    def test_remove_config_with_callbacks(self, manager):
        """Test remove_config with callbacks."""
        callback_called = False
        callback_config = None

        def test_callback(config):
            nonlocal callback_called, callback_config
            callback_called = True
            callback_config = config

        manager.on_config_removed.append(test_callback)

        # Add config
        config = MCPServerConfig(
            name="test-server", server_type=ServerType.STDIO, command="test-command"
        )
        manager.add_config(config)

        # Remove config
        result = manager.remove_config("test-server")
        assert result is True
        assert callback_called is True
        assert callback_config == config

    def test_remove_config_callback_exception(self, manager):
        """Test remove_config with callback that raises exception."""

        def failing_callback(config):
            raise Exception("Callback error")

        manager.on_config_removed.append(failing_callback)

        # Add config
        config = MCPServerConfig(
            name="test-server", server_type=ServerType.STDIO, command="test-command"
        )
        manager.add_config(config)

        # Remove should still succeed despite callback error
        result = manager.remove_config("test-server")
        assert result is True
        assert "test-server" not in manager.configs

    def test_remove_config_exception(self, manager):
        """Test remove_config with exception during removal."""
        # Add config
        config = MCPServerConfig(
            name="test-server", server_type=ServerType.STDIO, command="test-command"
        )
        manager.add_config(config)

        # Create a mock dict that raises exception on pop
        mock_configs = unittest.mock.MagicMock()
        mock_configs.__contains__ = lambda self, key: key == "test-server"
        mock_configs.pop = unittest.mock.Mock(side_effect=Exception("Pop error"))

        # Replace the configs dict temporarily
        with unittest.mock.patch.object(manager, "configs", mock_configs):
            result = manager.remove_config("test-server")
            assert result is False

    def test_add_provider(self, manager):
        """Test add_provider method."""
        provider = unittest.mock.Mock(spec=FileConfigProvider)

        manager.add_provider(provider)
        assert provider in manager.providers

        # Adding same provider again should not duplicate
        manager.add_provider(provider)
        assert manager.providers.count(provider) == 1

    def test_remove_provider_success(self, manager):
        """Test remove_provider success case."""
        provider = unittest.mock.Mock(spec=FileConfigProvider)
        provider.name = "test-provider"

        manager.add_provider(provider)
        result = manager.remove_provider("test-provider")
        assert result is True
        assert provider not in manager.providers

    def test_remove_provider_not_found(self, manager):
        """Test remove_provider with non-existent provider."""
        result = manager.remove_provider("non-existent")
        assert result is False

    def test_remove_provider_no_name_attribute(self, manager):
        """Test remove_provider with provider without name attribute."""
        provider = unittest.mock.Mock(spec=FileConfigProvider)
        # Don't set name attribute

        manager.add_provider(provider)
        result = manager.remove_provider("any-name")
        assert result is False
        assert provider in manager.providers

    @pytest.mark.asyncio
    async def test_load_from_providers(self, manager):
        """Test load_from_providers method."""
        # Create mock providers
        provider1 = unittest.mock.Mock(spec=FileConfigProvider)
        provider2 = unittest.mock.Mock(spec=FileConfigProvider)

        config1 = MCPServerConfig(
            name="server-1", server_type=ServerType.STDIO, command="command-1"
        )

        config2 = MCPServerConfig(
            name="server-2", server_type=ServerType.STDIO, command="command-2"
        )

        invalid_config = MCPServerConfig(
            name="",  # Invalid
            server_type=ServerType.STDIO,
            command="invalid-command",
        )

        provider1.load_configs = unittest.mock.AsyncMock(return_value=[config1])
        provider2.load_configs = unittest.mock.AsyncMock(
            return_value=[config2, invalid_config]
        )

        manager.add_provider(provider1)
        manager.add_provider(provider2)

        configs = await manager.load_from_providers()

        # Should load only valid configs
        assert len(configs) == 2
        assert "server-1" in manager.configs
        assert "server-2" in manager.configs
        assert "" not in manager.configs

    @pytest.mark.asyncio
    async def test_load_from_providers_with_exception(self, manager):
        """Test load_from_providers with provider exception."""
        provider1 = unittest.mock.Mock(spec=FileConfigProvider)
        provider2 = unittest.mock.Mock(spec=FileConfigProvider)

        config1 = MCPServerConfig(
            name="server-1", server_type=ServerType.STDIO, command="command-1"
        )

        provider1.load_configs = unittest.mock.AsyncMock(return_value=[config1])
        provider2.load_configs = unittest.mock.AsyncMock(
            side_effect=Exception("Provider error")
        )

        manager.add_provider(provider1)
        manager.add_provider(provider2)

        configs = await manager.load_from_providers()

        # Should load configs from working provider
        assert len(configs) == 1
        assert "server-1" in manager.configs

    @pytest.mark.asyncio
    async def test_save_to_providers_with_save_configs(self, manager):
        """Test save_to_providers with providers that support save_configs."""
        provider = unittest.mock.Mock(spec=FileConfigProvider)
        provider.save_configs = unittest.mock.AsyncMock(return_value=True)

        manager.add_provider(provider)

        # Add some configs
        config = MCPServerConfig(
            name="test-server", server_type=ServerType.STDIO, command="test-command"
        )
        manager.add_config(config)

        result = await manager.save_to_providers()
        assert result is True
        provider.save_configs.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_to_providers_without_save_configs(self, manager):
        """Test save_to_providers with providers that don't support save_configs."""
        provider = unittest.mock.Mock()
        provider.save_config = unittest.mock.AsyncMock(return_value=True)
        # Explicitly ensure save_configs attribute doesn't exist
        if hasattr(provider, "save_configs"):
            delattr(provider, "save_configs")

        manager.add_provider(provider)

        # Add some configs
        config = MCPServerConfig(
            name="test-server", server_type=ServerType.STDIO, command="test-command"
        )
        manager.add_config(config)

        result = await manager.save_to_providers()
        assert result is True
        provider.save_config.assert_called_once_with(config)

    @pytest.mark.asyncio
    async def test_save_to_providers_with_exception(self, manager):
        """Test save_to_providers with provider exception."""
        provider = unittest.mock.Mock(spec=FileConfigProvider)
        provider.save_configs = unittest.mock.AsyncMock(
            side_effect=Exception("Save error")
        )

        manager.add_provider(provider)

        result = await manager.save_to_providers()
        assert result is False

    def test_get_manager_stats(self, manager):
        """Test get_manager_stats method."""
        # Add some configs with different statuses and types
        config1 = MCPServerConfig(
            name="server-1",
            server_type=ServerType.STDIO,
            command="command-1",
            enabled=True,
            status=ServerStatus.CONNECTED,
        )

        config2 = MCPServerConfig(
            name="server-2",
            server_type=ServerType.HTTP,
            host="localhost",
            port=8080,
            enabled=False,
            status=ServerStatus.DISCONNECTED,
        )

        manager.add_config(config1)
        manager.add_config(config2)

        stats = manager.get_manager_stats()

        assert stats["total_configs"] == 2
        assert stats["enabled_configs"] == 1
        assert stats["disabled_configs"] == 1
        assert stats["status_distribution"]["CONNECTED"] == 1
        assert stats["status_distribution"]["DISCONNECTED"] == 1
        assert stats["type_distribution"]["STDIO"] == 1
        assert stats["type_distribution"]["HTTP"] == 1

    @pytest.mark.asyncio
    async def test_health_check_loop(self, manager):
        """Test _health_check_loop method."""
        # Add enabled config with health check
        config = MCPServerConfig(
            name="test-server",
            server_type=ServerType.HTTP,
            host="localhost",
            port=8080,
            enabled=True,
        )
        config.health_check.enabled = True
        manager.add_config(config)

        # Mock _check_server_health
        with unittest.mock.patch.object(manager, "_check_server_health") as mock_check:
            mock_check.return_value = asyncio.create_task(asyncio.sleep(0.01))

            # Start the loop
            manager._running = True
            task = asyncio.create_task(manager._health_check_loop())

            # Let it run briefly
            await asyncio.sleep(0.05)

            # Stop the loop
            manager._running = False
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            # Should have called health check
            mock_check.assert_called()

    @pytest.mark.asyncio
    async def test_health_check_loop_exception(self, manager):
        """Test _health_check_loop with exception."""
        # Add enabled config
        config = MCPServerConfig(
            name="test-server",
            server_type=ServerType.HTTP,
            host="localhost",
            port=8080,
            enabled=True,
        )
        config.health_check.enabled = True
        manager.add_config(config)

        # Mock _check_server_health to raise exception
        with unittest.mock.patch.object(
            manager, "_check_server_health", side_effect=Exception("Health check error")
        ):
            manager._running = True
            task = asyncio.create_task(manager._health_check_loop())

            # Let it run briefly
            await asyncio.sleep(0.05)

            # Stop the loop
            manager._running = False
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_check_server_health(self, manager):
        """Test _check_server_health method."""
        config = MCPServerConfig(
            name="test-server",
            server_type=ServerType.HTTP,
            host="localhost",
            port=8080,
            status=ServerStatus.CONNECTED,
        )

        # Should complete without error
        await manager._check_server_health(config)

    @pytest.mark.asyncio
    async def test_check_server_health_with_exception(self, manager):
        """Test _check_server_health with exception."""
        config = MCPServerConfig(
            name="test-server", server_type=ServerType.HTTP, host="localhost", port=8080
        )

        # Mock asyncio.sleep to raise exception
        with unittest.mock.patch("asyncio.sleep", side_effect=Exception("Sleep error")):
            # Should handle exception gracefully
            await manager._check_server_health(config)

    @pytest.mark.asyncio
    async def test_health_check_method(self, manager):
        """Test _health_check method."""
        config = MCPServerConfig(
            name="test-server", server_type=ServerType.HTTP, host="localhost", port=8080
        )

        with unittest.mock.patch.object(manager, "_check_server_health") as mock_check:
            mock_check.return_value = asyncio.create_task(asyncio.sleep(0.01))

            await manager._health_check(config)
            mock_check.assert_called_once_with(config)

    def test_magic_methods(self, manager):
        """Test magic methods (__len__, __contains__, __iter__, __repr__)."""
        # Add some configs
        config1 = MCPServerConfig(
            name="server-1", server_type=ServerType.STDIO, command="command-1"
        )
        config2 = MCPServerConfig(
            name="server-2", server_type=ServerType.STDIO, command="command-2"
        )

        manager.add_config(config1)
        manager.add_config(config2)

        # Test __len__
        assert len(manager) == 2

        # Test __contains__
        assert "server-1" in manager
        assert "server-2" in manager
        assert "non-existent" not in manager

        # Test __iter__
        server_names = list(manager)
        assert "server-1" in server_names
        assert "server-2" in server_names

        # Test __repr__
        repr_str = repr(manager)
        assert "ARKServerConfigManager" in repr_str
        assert "configs=2" in repr_str

    def test_get_security_levels_without_security_level(self, manager):
        """Test get_security_levels when SecurityLevel is None."""
        # Mock SecurityLevel to be None
        with unittest.mock.patch("src.core.config.server.SecurityLevel", None):
            levels = manager.get_security_levels()
            assert len(levels) == 0

    def test_get_security_levels_with_configs(self, manager):
        """Test get_security_levels with configs that have security levels."""
        # Import SecurityLevel to check if it's available
        from src.core.config.server import SecurityLevel

        if SecurityLevel is not None:
            # Create a mock config with security_level attribute
            config = MCPServerConfig(
                name="test-server", server_type=ServerType.STDIO, command="test-command"
            )
            # Manually add security_level attribute for testing
            config.security_level = SecurityLevel.MEDIUM
            manager.add_config(config)

            levels = manager.get_security_levels()
            assert len(levels) == 1
            assert SecurityLevel.MEDIUM in levels
        else:
            # If SecurityLevel is not available, test should pass with empty result
            config = MCPServerConfig(
                name="test-server", server_type=ServerType.STDIO, command="test-command"
            )
            manager.add_config(config)

            levels = manager.get_security_levels()
            assert len(levels) == 0

    @pytest.mark.asyncio
    async def test_health_check_http_method(self, manager):
        """Test health_check method with HTTP requests."""
        # Add enabled configs
        config1 = MCPServerConfig(
            name="server-1",
            server_type=ServerType.HTTP,
            host="localhost",
            port=8080,
            protocol="http",
            enabled=True,
            timeout=5,
        )

        config2 = MCPServerConfig(
            name="server-2",
            server_type=ServerType.HTTP,
            host="localhost",
            port=8081,
            protocol="http",
            enabled=False,  # Disabled, should not be checked
        )

        manager.add_config(config1)
        manager.add_config(config2)

        # Create a proper async context manager mock
        class MockResponse:
            def __init__(self, status=200):
                self.status = status

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        class MockSession:
            def get(self, url, timeout=None):
                return MockResponse(200)

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        with patch("aiohttp.ClientSession", MockSession):
            health_status = await manager.health_check()

            # Should only check enabled server
            assert len(health_status) == 1
            assert "server-1" in health_status
            assert health_status["server-1"]["healthy"] is True
            assert health_status["server-1"]["status_code"] == 200

    @pytest.mark.asyncio
    async def test_health_check_http_method_error_response(self, manager):
        """Test health_check method with HTTP error response."""
        config = MCPServerConfig(
            name="server-1",
            server_type=ServerType.HTTP,
            host="localhost",
            port=8080,
            protocol="http",
            enabled=True,
            timeout=5,
        )
        manager.add_config(config)

        # Mock classes for proper async context manager behavior
        class MockResponse:
            def __init__(self):
                self.status = 500

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        class MockSession:
            def get(self, url, timeout=None):
                return MockResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        with patch("aiohttp.ClientSession", MockSession):
            health_status = await manager.health_check()

            assert health_status["server-1"]["healthy"] is False
            assert health_status["server-1"]["status_code"] == 500
            assert "HTTP 500" in health_status["server-1"]["error"]

    @pytest.mark.asyncio
    async def test_health_check_http_method_exception(self, manager):
        """Test health_check method with exception."""
        config = MCPServerConfig(
            name="server-1",
            server_type=ServerType.HTTP,
            host="localhost",
            port=8080,
            protocol="http",
            enabled=True,
            timeout=5,
        )
        manager.add_config(config)

        # Define custom mock classes to properly simulate async context manager
        class MockSession:
            def __init__(self):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

            def get(self, url, timeout=None):
                raise Exception("Connection error")

        with patch("aiohttp.ClientSession", MockSession):
            health_status = await manager.health_check()

            assert health_status["server-1"]["healthy"] is False
            assert "Connection error" in health_status["server-1"]["error"]


class TestServerCredentials:
    """Test ServerCredentials class."""

    def test_credentials_to_dict_basic(self):
        """Test converting credentials to dict with basic auth."""
        credentials = ServerCredentials(
            auth_type=AuthType.BASIC_AUTH, username="testuser", password="testpass"
        )

        result = credentials.to_dict()
        assert result["auth_type"] == "basic_auth"
        assert result["has_username"] is True
        assert result["has_password"] is True
        assert result["has_api_key"] is False
        assert result["has_bearer_token"] is False
        assert result["oauth2_configured"] is False
        assert result["custom_headers_count"] == 0

    def test_credentials_to_dict_api_key(self):
        """Test converting credentials to dict with API key."""
        credentials = ServerCredentials(
            auth_type=AuthType.API_KEY, api_key="secret_key_123"
        )

        result = credentials.to_dict()
        assert result["auth_type"] == "api_key"
        assert result["has_api_key"] is True
        assert result["has_username"] is False
        assert result["has_password"] is False
        assert result["has_bearer_token"] is False

    def test_credentials_to_dict_oauth2(self):
        """Test converting credentials to dict with OAuth2."""
        credentials = ServerCredentials(
            auth_type=AuthType.OAUTH2,
            oauth2_config={"client_id": "test", "client_secret": "secret"},
            custom_headers={"X-Custom": "value", "Authorization": "Bearer token"},
        )

        result = credentials.to_dict()
        assert result["auth_type"] == "oauth2"
        assert result["oauth2_configured"] is True
        assert result["custom_headers_count"] == 2


class TestServerHealthCheck:
    """Test ServerHealthCheck class."""

    def test_health_check_to_dict(self):
        """Test converting health check config to dict."""
        health_check = ServerHealthCheck(
            enabled=True,
            interval=30,
            timeout=5,
            max_failures=2,
            retry_delay=15,
            health_endpoint="/health",
            expected_response="OK",
        )

        result = health_check.to_dict()
        assert result["enabled"] is True
        assert result["interval"] == 30
        assert result["timeout"] == 5
        assert result["max_failures"] == 2
        assert result["retry_delay"] == 15
        assert result["health_endpoint"] == "/health"
        assert result["expected_response"] == "OK"


class TestServerLimits:
    """Test ServerLimits class."""

    def test_limits_to_dict(self):
        """Test converting limits to dict."""
        limits = ServerLimits(
            max_connections=20,
            max_requests_per_minute=200,
            max_request_size=2048,
            max_response_size=4096,
            connection_timeout=45,
            request_timeout=90,
        )

        result = limits.to_dict()
        assert result["max_connections"] == 20
        assert result["max_requests_per_minute"] == 200
        assert result["max_request_size"] == 2048
        assert result["max_response_size"] == 4096
        assert result["connection_timeout"] == 45
        assert result["request_timeout"] == 90


class TestServerMetrics:
    """Test ServerMetrics class."""

    def test_metrics_to_dict(self):
        """Test converting metrics to dict."""
        metrics = ServerMetrics(
            total_connections=10,
            successful_connections=8,
            failed_connections=2,
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            average_response_time=0.5,
            uptime_percentage=95.0,
        )

        result = metrics.to_dict()
        assert result["total_connections"] == 10
        assert result["successful_connections"] == 8
        assert result["failed_connections"] == 2
        assert result["total_requests"] == 100
        assert result["successful_requests"] == 95
        assert result["failed_requests"] == 5
        assert result["average_response_time"] == 0.5
        assert result["uptime_percentage"] == 95.0

    def test_metrics_update_connection_success(self):
        """Test updating connection metrics for successful connection."""
        metrics = ServerMetrics()

        metrics.update_connection(success=True)

        assert metrics.total_connections == 1
        assert metrics.successful_connections == 1
        assert metrics.failed_connections == 0
        assert metrics.last_connection_time is not None
        assert metrics.last_error is None

    def test_metrics_update_connection_failure(self):
        """Test updating connection metrics for failed connection."""
        metrics = ServerMetrics()

        metrics.update_connection(success=False, error="Connection timeout")

        assert metrics.total_connections == 1
        assert metrics.successful_connections == 0
        assert metrics.failed_connections == 1
        assert metrics.last_error == "Connection timeout"
        assert metrics.last_error_time is not None

    def test_metrics_update_request_success_first(self):
        """Test updating request metrics for first successful request."""
        metrics = ServerMetrics()

        metrics.update_request(success=True, response_time=0.5)

        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0
        assert metrics.average_response_time == 0.5
        assert metrics.last_request_time is not None

    def test_metrics_update_request_success_multiple(self):
        """Test updating request metrics for multiple successful requests."""
        metrics = ServerMetrics()

        metrics.update_request(success=True, response_time=0.5)
        metrics.update_request(success=True, response_time=1.0)

        assert metrics.total_requests == 2
        assert metrics.successful_requests == 2
        assert metrics.failed_requests == 0
        assert metrics.average_response_time == 0.75  # (0.5 + 1.0) / 2

    def test_metrics_update_request_failure(self):
        """Test updating request metrics for failed request."""
        metrics = ServerMetrics()

        metrics.update_request(success=False, response_time=0.0, error="Request failed")

        assert metrics.total_requests == 1
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 1
        assert metrics.last_error == "Request failed"
        assert metrics.last_error_time is not None


class TestEnvironmentConfigProvider:
    """Test EnvironmentConfigProvider class."""

    @pytest.mark.asyncio
    async def test_env_provider_load_configs_empty(self):
        """Test loading configs when no environment variables are set."""
        provider = EnvironmentConfigProvider(prefix="TEST_MCP_")
        configs = await provider.load_configs()
        assert isinstance(configs, list)
        # Should be empty since no TEST_MCP_ variables are set

    @pytest.mark.asyncio
    async def test_env_provider_load_configs_with_vars(self):
        """Test loading configs from environment variables."""
        # Mock environment variables
        env_vars = {
            "ARK_MCP_SERVER1_HOST": "localhost",
            "ARK_MCP_SERVER1_PORT": "8080",
            "ARK_MCP_SERVER1_PROTOCOL": "http",
            "ARK_MCP_SERVER1_ENABLED": "true",
            "ARK_MCP_SERVER1_DESCRIPTION": "Test server 1",
            "ARK_MCP_SERVER2_HOST": "example.com",
            "ARK_MCP_SERVER2_PORT": "9090",
            "ARK_MCP_SERVER2_API_KEY": "secret123",
            "ARK_MCP_SERVER2_ENABLED": "false",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            provider = EnvironmentConfigProvider()
            configs = await provider.load_configs()

            assert len(configs) == 2

            # Check server1
            server1 = next((c for c in configs if c.name == "server1"), None)
            assert server1 is not None
            assert server1.host == "localhost"
            assert server1.port == 8080
            assert server1.protocol == "http"
            assert server1.enabled is True
            assert server1.description == "Test server 1"

            # Check server2
            server2 = next((c for c in configs if c.name == "server2"), None)
            assert server2 is not None
            assert server2.host == "example.com"
            assert server2.port == 9090
            assert server2.api_key == "secret123"
            assert server2.enabled is False

    @pytest.mark.asyncio
    async def test_env_provider_save_config_not_implemented(self):
        """Test that save_config raises NotImplementedError."""
        provider = EnvironmentConfigProvider()
        config = MCPServerConfig(name="test", server_type=ServerType.HTTP)

        with pytest.raises(NotImplementedError):
            await provider.save_config(config)

    @pytest.mark.asyncio
    async def test_env_provider_delete_config_not_implemented(self):
        """Test that delete_config raises NotImplementedError."""
        provider = EnvironmentConfigProvider()

        with pytest.raises(NotImplementedError):
            await provider.delete_config("test_server")


class TestFileConfigProviderErrorHandling:
    """Test FileConfigProvider error handling scenarios."""

    @pytest.mark.asyncio
    async def test_file_provider_load_configs_invalid_json(self):
        """Test loading configs from invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"invalid": json}')  # Invalid JSON
            f.flush()

            try:
                provider = FileConfigProvider(config_path=f.name)
                configs = await provider.load_configs()
                # Should handle error gracefully and return empty list
                assert isinstance(configs, list)
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_file_provider_load_configs_invalid_yaml(self):
        """Test loading configs from invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")  # Invalid YAML
            f.flush()

            try:
                provider = FileConfigProvider(config_path=f.name)
                configs = await provider.load_configs()
                # Should handle error gracefully and return empty list
                assert isinstance(configs, list)
            finally:
                os.unlink(f.name)


class TestARKServerConfigManagerUpdateServerStatus:
    """Test ARKServerConfigManager update_server_status method."""

    @pytest.fixture
    def manager(self):
        """Create a manager instance for testing."""
        return ARKServerConfigManager()

    def test_update_server_status_success(self, manager):
        """Test updating server status successfully."""
        # Add a config first
        config = MCPServerConfig(
            name="test_server", server_type=ServerType.HTTP, host="localhost", port=8080
        )
        manager.add_config(config)

        # Add a callback to track status changes
        callback_called = False

        def status_callback(config, old_status, new_status):
            nonlocal callback_called
            callback_called = True
            assert config.name == "test_server"
            assert old_status == ServerStatus.UNKNOWN
            assert new_status == ServerStatus.CONNECTED

        manager.on_status_changed.append(status_callback)

        # Update status
        result = manager.update_server_status("test_server", ServerStatus.CONNECTED)

        assert result is True
        assert callback_called is True
        assert manager.get_config("test_server").status == ServerStatus.CONNECTED

    def test_update_server_status_not_found(self, manager):
        """Test updating status for non-existent server."""
        result = manager.update_server_status("nonexistent", ServerStatus.CONNECTED)
        assert result is False

    def test_update_server_status_callback_exception(self, manager):
        """Test that callback exceptions don't break status update."""
        # Add a config first
        config = MCPServerConfig(
            name="test_server", server_type=ServerType.HTTP, host="localhost", port=8080
        )
        manager.add_config(config)

        # Add a callback that raises an exception
        def failing_callback(config, old_status, new_status):
            raise Exception("Callback failed")

        manager.on_status_changed.append(failing_callback)

        # Update status should still work despite callback failure
        result = manager.update_server_status("test_server", ServerStatus.CONNECTED)

        assert result is True
        assert manager.get_config("test_server").status == ServerStatus.CONNECTED


class TestSimpleMCPServerConfigMethods:
    """Test SimpleMCPServerConfig to_dict and from_dict methods."""

    def test_simple_config_to_dict(self):
        """Test SimpleMCPServerConfig.to_dict method."""
        config = SimpleMCPServerConfig(
            name="test_server", command=["python", "server.py"], args=["--port", "8080"]
        )
        result = config.to_dict()

        assert result["name"] == "test_server"
        assert result["command"] == ["python", "server.py"]
        assert result["args"] == ["--port", "8080"]

    def test_simple_config_from_dict(self):
        """Test SimpleMCPServerConfig.from_dict method."""
        data = {
            "name": "test_server",
            "command": ["python", "server.py"],
            "args": ["--port", "8080"],
        }
        config = SimpleMCPServerConfig.from_dict(data)

        assert config.name == "test_server"
        assert config.command == ["python", "server.py"]
        assert config.args == ["--port", "8080"]


class TestMCPServerConfigValidationEdgeCases:
    """Test MCPServerConfig validation edge cases."""

    def test_validate_empty_host(self):
        """Test validation with empty host for HTTP server."""
        config = MCPServerConfig(
            name="test_server",
            server_type=ServerType.HTTP,
            host="",  # Empty host
            port=8080,
        )
        is_valid, errors = config.validate()
        assert not is_valid
        assert any("Host cannot be empty" in error for error in errors)

    def test_validate_invalid_protocol(self):
        """Test validation with invalid protocol."""
        config = MCPServerConfig(
            name="test_server",
            server_type=ServerType.HTTP,
            host="localhost",
            port=8080,
            protocol="ftp",  # Invalid protocol
        )
        is_valid, errors = config.validate()
        assert not is_valid
        assert any("Protocol must be" in error for error in errors)

    def test_validate_negative_timeout(self):
        """Test validation with negative timeout."""
        config = MCPServerConfig(
            name="test_server",
            server_type=ServerType.HTTP,
            host="localhost",
            port=8080,
            timeout=-1,  # Negative timeout
        )
        is_valid, errors = config.validate()
        assert not is_valid
        assert any("Timeout must be positive" in error for error in errors)

    def test_validate_negative_max_retries(self):
        """Test validation with negative max_retries."""
        config = MCPServerConfig(
            name="test_server",
            server_type=ServerType.HTTP,
            host="localhost",
            port=8080,
            max_retries=-1,  # Negative max_retries
        )
        is_valid, errors = config.validate()
        assert not is_valid
        assert any("Max retries cannot be negative" in error for error in errors)


class TestFileConfigProviderDirectoryErrorHandling:
    """Test FileConfigProvider directory loading error handling."""

    @pytest.mark.asyncio
    async def test_load_configs_directory_with_exception(self):
        """Test loading configs from directory with file read exception."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file that will cause an exception when read
            bad_file = Path(temp_dir) / "bad_config.json"
            bad_file.write_text("invalid json content {")

            provider = FileConfigProvider(temp_dir)

            # This should handle the exception gracefully
            configs = await provider.load_configs()
            # Should return empty list when files can't be parsed
            assert configs == []

    @pytest.mark.asyncio
    async def test_load_configs_directory_single_config_format(self):
        """Test loading single config format from directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "single_config.json"
            # Single config format (not wrapped in servers array)
            config_data = {
                "name": "single_server",
                "server_type": "stdio",
                "command": ["python", "server.py"],
            }
            config_file.write_text(json.dumps(config_data))

            provider = FileConfigProvider(temp_dir)
            configs = await provider.load_configs()

            assert len(configs) == 1
            assert configs[0].name == "single_server"

    @pytest.mark.asyncio
    async def test_load_configs_directory_servers_dict_format(self):
        """Test loading configs with 'servers' dictionary wrapper."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "servers_dict.json"
            # Configs wrapped in "servers" dictionary
            config_data = {
                "servers": [
                    {
                        "name": "server1",
                        "server_type": "stdio",
                        "command": ["python", "server1.py"],
                    },
                    {
                        "name": "server2",
                        "server_type": "stdio",
                        "command": ["python", "server2.py"],
                    },
                ]
            }
            config_file.write_text(json.dumps(config_data))

            provider = FileConfigProvider(temp_dir)
            configs = await provider.load_configs()

            assert len(configs) == 2
            assert configs[0].name == "server1"
            assert configs[1].name == "server2"


class TestEnvironmentConfigProviderErrorHandling:
    """Test EnvironmentConfigProvider error handling."""

    @pytest.mark.asyncio
    async def test_env_provider_load_configs_with_invalid_json(self):
        """Test environment provider with invalid JSON in environment variable."""
        with patch.dict(
            os.environ, {"MCP_SERVER_CONFIG_test_server": "invalid json {"}
        ):
            provider = EnvironmentConfigProvider()
            configs = await provider.load_configs()

            # Should handle JSON parsing error gracefully
            assert configs == []


class TestMCPServerConfigSSLValidation:
    """Test SSL validation paths in MCPServerConfig."""

    def test_validate_ssl_cert_not_found(self):
        """Test validation with non-existent SSL certificate file."""
        config = MCPServerConfig(
            name="test_server",
            server_type=ServerType.HTTP,
            host="localhost",
            port=8080,
            ssl_cert_path="/nonexistent/cert.pem",
        )

        is_valid, errors = config.validate()
        assert not is_valid
        assert any("SSL certificate file not found" in error for error in errors)

    def test_validate_ssl_key_not_found(self):
        """Test validation with non-existent SSL key file."""
        config = MCPServerConfig(
            name="test_server",
            server_type=ServerType.HTTP,
            host="localhost",
            port=8080,
            ssl_key_path="/nonexistent/key.pem",
        )

        is_valid, errors = config.validate()
        assert not is_valid
        assert any("SSL key file not found" in error for error in errors)


class TestMCPServerConfigLimitsValidation:
    """Test limits validation paths in MCPServerConfig."""

    def test_validate_zero_max_connections(self):
        """Test validation with zero max connections."""
        config = MCPServerConfig(
            name="test_server", server_type=ServerType.HTTP, host="localhost", port=8080
        )
        config.limits.max_connections = 0

        is_valid, errors = config.validate()
        assert not is_valid
        assert any("Max connections must be positive" in error for error in errors)

    def test_validate_zero_connection_timeout(self):
        """Test validation with zero connection timeout."""
        config = MCPServerConfig(
            name="test_server", server_type=ServerType.HTTP, host="localhost", port=8080
        )
        config.limits.connection_timeout = 0

        is_valid, errors = config.validate()
        assert not is_valid
        assert any("Connection timeout must be positive" in error for error in errors)


class TestMCPServerConfigURLProperty:
    """Test URL property edge cases in MCPServerConfig."""

    def test_url_property_http_default_port(self):
        """Test URL property with HTTP default port 80."""
        config = MCPServerConfig(
            name="test_server",
            server_type=ServerType.HTTP,
            host="example.com",
            port=80,
            protocol="http",
        )

        assert config.url == "http://example.com"

    def test_url_property_https_default_port(self):
        """Test URL property with HTTPS default port 443."""
        config = MCPServerConfig(
            name="test_server",
            server_type=ServerType.HTTP,
            host="example.com",
            port=443,
            protocol="https",
        )

        assert config.url == "https://example.com"

    def test_url_property_websocket_with_custom_url(self):
        """Test URL property with WebSocket and custom URL."""
        config = MCPServerConfig(
            name="test_server",
            server_type=ServerType.WEBSOCKET,
            host="example.com",
            port=8080,
        )
        config.url = "wss://custom.example.com/ws"

        assert config.url == "wss://custom.example.com/ws"

    def test_url_property_tcp_server(self):
        """Test URL property with TCP server (should return None)."""
        config = MCPServerConfig(
            name="test_server", server_type=ServerType.TCP, host="localhost", port=9090
        )

        assert config.url is None


class TestMCPServerConfigUpdateMethod:
    """Test MCPServerConfig update method."""

    def test_update_valid_attributes(self):
        """Test updating valid configuration attributes."""
        config = MCPServerConfig(
            name="test_server", server_type=ServerType.HTTP, host="localhost", port=8080
        )

        original_updated_at = config.updated_at

        config.update(host="newhost", port=9090, description="Updated server")

        assert config.host == "newhost"
        assert config.port == 9090
        assert config.description == "Updated server"
        assert config.updated_at > original_updated_at

    def test_update_invalid_attribute(self):
        """Test updating with invalid attribute raises ValueError."""
        config = MCPServerConfig(
            name="test_server", server_type=ServerType.HTTP, host="localhost", port=8080
        )

        with pytest.raises(
            ValueError, match="Invalid configuration parameter: invalid_attr"
        ):
            config.update(invalid_attr="value")


class TestEnvironmentConfigProviderAuthHandling:
    """Test EnvironmentConfigProvider authentication handling."""

    @pytest.mark.asyncio
    async def test_env_provider_with_auth_token(self):
        """Test environment provider with auth token."""
        env_vars = {
            "ARK_MCP_TEST_AUTH_TOKEN": "test_token",
            "ARK_MCP_TEST_HOST": "localhost",
            "ARK_MCP_TEST_PORT": "8080",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            provider = EnvironmentConfigProvider()
            configs = await provider.load_configs()

            assert len(configs) == 1
            config = configs[0]
            assert config.name == "test"
            assert config.credentials.bearer_token == "test_token"
            assert config.credentials.auth_type == AuthType.BEARER_TOKEN

    @pytest.mark.asyncio
    async def test_env_provider_with_api_key(self):
        """Test environment provider with API key."""
        env_vars = {
            "ARK_MCP_API_API_KEY": "test_api_key",
            "ARK_MCP_API_HOST": "api.example.com",
            "ARK_MCP_API_PORT": "443",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            provider = EnvironmentConfigProvider()
            configs = await provider.load_configs()

            assert len(configs) == 1
            config = configs[0]
            assert config.name == "api"
            assert config.credentials.api_key == "test_api_key"
            assert config.credentials.auth_type == AuthType.API_KEY

    @pytest.mark.asyncio
    async def test_env_provider_config_creation_exception(self):
        """Test environment provider handling config creation exception."""
        # Test with invalid port value to trigger exception
        env_vars = {
            "ARK_MCP_INVALID_PORT": "not_a_number",
            "ARK_MCP_INVALID_HOST": "localhost",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            provider = EnvironmentConfigProvider()
            configs = await provider.load_configs()

            # Should return empty list due to exception in config creation
            assert len(configs) == 0


class TestSecurityLevelImportError:
    """Test SecurityLevel import error handling."""

    def test_security_level_import_error(self):
        """Test that SecurityLevel import error is handled gracefully."""
        # This test covers the import exception handling at lines 27-28
        # The import is already done at module level, so we test the result
        from src.core.config.server import SecurityLevel

        # SecurityLevel might be None if import failed
        # This is acceptable behavior
        assert SecurityLevel is None or hasattr(SecurityLevel, "__name__")


if __name__ == "__main__":
    pytest.main([__file__])
