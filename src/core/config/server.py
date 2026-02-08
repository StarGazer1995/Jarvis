"""ARK Server Configuration Management System

This module provides comprehensive configuration management for MCP servers
in the ARK engine, including server discovery, health monitoring, and
configuration persistence. Updated to work with official modelcontextprotocol SDK.
"""

import logging
import asyncio
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import yaml
from abc import ABC, abstractmethod

# Import official MCP types
from mcp import StdioServerParameters

# Import security manager if available
try:
    from ..security.manager import SecurityLevel
except ImportError:
    SecurityLevel = None


class ServerStatus(Enum):
    """MCP Server status enumeration."""
    UNKNOWN = "unknown"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    DISABLED = "disabled"


class ServerType(Enum):
    """MCP Server type enumeration."""
    STDIO = "stdio"
    HTTP = "http"
    WEBSOCKET = "websocket"
    TCP = "tcp"
    CUSTOM = "custom"


class AuthType(Enum):
    """Authentication type enumeration."""
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    CUSTOM = "custom"


class ConfigType(Enum):
    """Configuration type enumeration."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class ServerCredentials:
    """Server authentication credentials."""
    auth_type: AuthType = AuthType.NONE
    api_key: Optional[str] = None
    bearer_token: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    oauth2_config: Dict[str, Any] = field(default_factory=dict)
    custom_headers: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert credentials to dictionary (excluding sensitive data)."""
        return {
            'auth_type': self.auth_type.value,
            'has_api_key': bool(self.api_key),
            'has_bearer_token': bool(self.bearer_token),
            'has_username': bool(self.username),
            'has_password': bool(self.password),
            'oauth2_configured': bool(self.oauth2_config),
            'custom_headers_count': len(self.custom_headers)
        }


@dataclass
class ServerHealthCheck:
    """Server health check configuration."""
    enabled: bool = True
    interval: int = 60  # seconds
    timeout: int = 10  # seconds
    max_failures: int = 3
    retry_delay: int = 30  # seconds
    health_endpoint: Optional[str] = None
    expected_response: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert health check config to dictionary."""
        return asdict(self)


@dataclass
class ServerLimits:
    """Server resource limits and quotas."""
    max_connections: int = 10
    max_requests_per_minute: int = 100
    max_request_size: int = 1024 * 1024  # 1MB
    max_response_size: int = 10 * 1024 * 1024  # 10MB
    connection_timeout: int = 30  # seconds
    request_timeout: int = 60  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert limits to dictionary."""
        return asdict(self)


@dataclass
class ServerMetrics:
    """Server performance and usage metrics."""
    total_connections: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_connection_time: Optional[float] = None
    last_request_time: Optional[float] = None
    last_error: Optional[str] = None
    last_error_time: Optional[float] = None
    uptime_percentage: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)
    
    def update_connection(self, success: bool, error: Optional[str] = None) -> None:
        """Update connection metrics."""
        self.total_connections += 1
        if success:
            self.successful_connections += 1
            self.last_connection_time = time.time()
        else:
            self.failed_connections += 1
            if error:
                self.last_error = error
                self.last_error_time = time.time()
    
    def update_request(self, success: bool, response_time: float, error: Optional[str] = None) -> None:
        """Update request metrics."""
        self.total_requests += 1
        self.last_request_time = time.time()
        
        if success:
            self.successful_requests += 1
            # Update average response time
            if self.successful_requests == 1:
                self.average_response_time = response_time
            else:
                self.average_response_time = (
                    (self.average_response_time * (self.successful_requests - 1) + response_time) 
                    / self.successful_requests
                )
        else:
            self.failed_requests += 1
            if error:
                self.last_error = error
                self.last_error_time = time.time()


@dataclass
class SimpleMCPServerConfig:
    """
    Simplified MCP server configuration for official SDK.
    
    This class provides a streamlined configuration interface that works
    directly with the official modelcontextprotocol SDK.
    """
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    enabled: bool = True
    timeout: int = 30
    retry_count: int = 3
    
    def to_stdio_params(self) -> StdioServerParameters:
        """Convert to official SDK StdioServerParameters."""
        return StdioServerParameters(
            command=self.command,
            args=self.args if self.args is not None else [],
            env=self.env if self.env else None
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimpleMCPServerConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class MCPServerConfig:
    """
    Comprehensive MCP server configuration.
    
    This class encapsulates all configuration parameters for an MCP server,
    including connection details, authentication, health monitoring, and limits.
    """
    name: str
    server_type: ServerType
    status: ServerStatus = ServerStatus.UNKNOWN
    config_type: ConfigType = ConfigType.DEVELOPMENT
    
    # Connection configuration
    command: Optional[str] = None  # For STDIO servers
    args: List[str] = field(default_factory=list)  # Command arguments
    host: Optional[str] = None  # For TCP servers
    port: Optional[int] = None  # For TCP servers
    protocol: str = "http"  # Protocol for connection
    timeout: float = 30.0  # Connection timeout in seconds
    max_retries: int = 3  # Maximum retry attempts
    
    # Environment and working directory
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    
    # Authentication and security
    credentials: ServerCredentials = field(default_factory=ServerCredentials)
    ssl_verify: bool = True
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    
    # Health monitoring
    health_check: ServerHealthCheck = field(default_factory=ServerHealthCheck)
    
    # Resource limits
    limits: ServerLimits = field(default_factory=ServerLimits)
    
    # Security and authentication
    security_level: Optional[SecurityLevel] = SecurityLevel.MEDIUM if SecurityLevel else None
    api_key: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    description: str = ""
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    priority: int = 100  # Lower number = higher priority
    enabled: bool = True
    auto_start: bool = True
    auto_restart: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime data
    metrics: ServerMetrics = field(default_factory=ServerMetrics)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: float = field(default_factory=time.time)
    last_updated: Optional[datetime] = None
    
    # Private fields
    _url: Optional[str] = field(default=None, init=False)
    
    def to_dict(self, include_sensitive: bool = True) -> Dict[str, Any]:
        """
        Convert server config to dictionary.
        
        Args:
            include_sensitive: Whether to include sensitive data
            
        Returns:
            Dictionary representation of the config
        """
        config_dict = {
            'name': self.name,
            'server_type': self.server_type.value,
            'command': self.command,
            'args': self.args,
            'host': self.host,
            'port': self.port,
            'protocol': self.protocol,
            'enabled': self.enabled,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'security_level': self.security_level.value if self.security_level else None,
            'config_type': self.config_type.value,
            'api_key': self.api_key if include_sensitive else ('***' if self.api_key else None),
            'headers': self.headers,
            'description': self.description,
            'version': self.version,
            'tags': self.tags,
            'priority': self.priority,
            'auto_start': self.auto_start,
            'auto_restart': self.auto_restart,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'last_updated': self.last_updated.isoformat() if self.last_updated else (datetime.fromtimestamp(self.updated_at).isoformat() if isinstance(self.updated_at, (int, float)) else self.updated_at)
        }
        
        return config_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPServerConfig':
        """
        Create server config from dictionary.
        
        Args:
            data: Dictionary containing config data
            
        Returns:
            MCPServerConfig instance
        """
        # Parse datetime strings if needed
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at)
            except ValueError:
                created_at = datetime.now()
        elif created_at is None:
            created_at = datetime.now()
            
        last_updated = data.get('last_updated')
        if isinstance(last_updated, str):
            try:
                last_updated = datetime.fromisoformat(last_updated)
            except ValueError:
                last_updated = None
        
        # Parse security level
        security_level = data.get('security_level')
        if isinstance(security_level, str):
            security_level = SecurityLevel(security_level) if SecurityLevel else security_level
        
        # Create main config
        config = cls(
            name=data['name'],
            server_type=ServerType(data['server_type']),
            host=data.get('host'),
            port=data.get('port'),
            protocol=data.get('protocol', 'http'),
            enabled=data.get('enabled', True),
            timeout=data.get('timeout', 30.0),
            max_retries=data.get('max_retries', 3),
            security_level=security_level,
            config_type=ConfigType(data.get('config_type', 'development')),
            api_key=data.get('api_key'),
            headers=data.get('headers', {}),
            description=data.get('description', ''),
            version=data.get('version', '1.0.0'),
            tags=data.get('tags', []),
            priority=data.get('priority', 100),
            auto_start=data.get('auto_start', True),
            auto_restart=data.get('auto_restart', True),
            metadata=data.get('metadata', {}),
            created_at=created_at,
            last_updated=last_updated
        )
        
        # Handle credentials
        if 'auth_token' in data:
            config.credentials.bearer_token = data['auth_token']
            config.credentials.auth_type = AuthType.BEARER_TOKEN
        
        if 'api_key' in data:
            config.credentials.api_key = data['api_key']
            config.credentials.auth_type = AuthType.API_KEY
        
        return config
    
    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate server configuration.
        
        Returns:
            Tuple of (is_valid, errors) where errors is a list of error messages
        """
        errors = []
        
        # Basic validation
        if not self.name:
            errors.append("Server name is required")
        
        if not self.server_type:
            errors.append("Server type is required")
        
        # Type-specific validation
        if self.server_type == ServerType.STDIO:
            if not self.command:
                errors.append("Command is required for STDIO servers")
        elif self.server_type in [ServerType.HTTP, ServerType.WEBSOCKET]:
            if not self.url and not (self.host and self.port):
                errors.append("URL or host+port is required for HTTP/WebSocket servers")
            # Additional validation for HTTP/WebSocket
            if self.host == "":
                errors.append("Host cannot be empty for HTTP/WebSocket servers")
            if self.port is not None and (self.port <= 0 or self.port > 65535):
                errors.append("Port must be between 1 and 65535")
            if self.protocol not in ["http", "https", "ws", "wss"]:
                errors.append("Protocol must be http, https, ws, or wss")
        elif self.server_type == ServerType.TCP:
            if not self.host or not self.port:
                errors.append("Host and port are required for TCP servers")
            # Port range validation for TCP servers
            if self.port is not None and (self.port <= 0 or self.port > 65535):
                errors.append("Port must be between 1 and 65535")
        
        # Timeout validation
        if self.timeout <= 0:
            errors.append("Timeout must be positive")
        
        # Max retries validation
        if self.max_retries < 0:
            errors.append("Max retries cannot be negative")
        
        # SSL validation
        if self.ssl_cert_path and not os.path.exists(self.ssl_cert_path):
            errors.append(f"SSL certificate file not found: {self.ssl_cert_path}")
        
        if self.ssl_key_path and not os.path.exists(self.ssl_key_path):
            errors.append(f"SSL key file not found: {self.ssl_key_path}")
        
        # Limits validation
        if self.limits.max_connections <= 0:
            errors.append("Max connections must be positive")
        
        if self.limits.connection_timeout <= 0:
            errors.append("Connection timeout must be positive")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        is_valid, _ = self.validate()
        return is_valid
    
    def update_status(self, status: ServerStatus) -> None:
        """Update server status and timestamp."""
        self.status = status
        self.updated_at = time.time()
    
    def __repr__(self) -> str:
        """String representation of the server configuration."""
        if self.server_type in [ServerType.HTTP, ServerType.WEBSOCKET]:
            if self.url:
                connection_info = self.url
            elif self.host and self.port:
                connection_info = f"{self.protocol}://{self.host}:{self.port}"
            else:
                connection_info = "no_url"
        elif self.server_type == ServerType.STDIO:
            connection_info = f"command={self.command}"
        elif self.server_type == ServerType.TCP:
            connection_info = f"{self.host}:{self.port}" if self.host and self.port else "no_host_port"
        else:
            connection_info = "unknown"
        
        return f"MCPServerConfig(name='{self.name}', type={self.server_type.value}, connection={connection_info})"
    
    @property
    def url(self) -> Optional[str]:
        """Get the server URL for HTTP/WebSocket servers."""
        if self.server_type in [ServerType.HTTP, ServerType.WEBSOCKET]:
            if hasattr(self, '_url') and self._url:
                return self._url
            elif self.host and self.port:
                # Handle default ports
                if (self.protocol == "http" and self.port == 80) or \
                   (self.protocol == "https" and self.port == 443):
                    return f"{self.protocol}://{self.host}"
                else:
                    return f"{self.protocol}://{self.host}:{self.port}"
        return None
    
    @url.setter
    def url(self, value: Optional[str]) -> None:
        """Set the server URL."""
        self._url = value
    
    @property
    def is_secure(self) -> bool:
        """Check if the connection is secure (HTTPS/WSS)."""
        if self.server_type in [ServerType.HTTP, ServerType.WEBSOCKET]:
            return self.protocol in ["https", "wss"]
        return False
    
    def update(self, **kwargs) -> None:
        """
        Update server configuration with new values.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration parameter: {key}")
        
        # Update the last_updated timestamp
        self.last_updated = datetime.now()
        # Update timestamp
        self.updated_at = time.time()


class ServerConfigProvider(ABC):
    """Abstract base class for server configuration providers."""
    
    @abstractmethod
    async def load_configs(self) -> List[MCPServerConfig]:
        """
        Load server configurations.
        
        Returns:
            List of server configurations
        """
        pass
    
    @abstractmethod
    async def save_config(self, config: MCPServerConfig) -> bool:
        """
        Save server configuration.
        
        Args:
            config: Server configuration to save
            
        Returns:
            True if saved successfully
        """
        pass
    
    @abstractmethod
    async def delete_config(self, server_name: str) -> bool:
        """
        Delete server configuration.
        
        Args:
            server_name: Name of server to delete
            
        Returns:
            True if deleted successfully
        """
        pass


class FileConfigProvider(ServerConfigProvider):
    """File-based server configuration provider."""
    
    def __init__(self, config_path: str = "config/servers"):
        """
        Initialize file config provider.
        
        Args:
            config_path: Directory containing config files or path to a single config file
        """
        self.config_path = Path(config_path)
        
        # Check if it's a file or directory
        if self.config_path.suffix in ['.json', '.yaml', '.yml']:
            # Single file mode
            self.file_path = self.config_path
            self.config_dir = self.config_path.parent
            self.single_file_mode = True
            self.name = "file_provider"
        else:
            # Directory mode
            self.config_dir = self.config_path
            self.file_path = None
            self.single_file_mode = False
            self.name = "file_provider"
        
        # Note: Directories will be created when needed during save operations
            
        self.logger = logging.getLogger('jarvis.server_config.file')
    
    async def load_configs(self) -> List[MCPServerConfig]:
        """Load configurations from files."""
        configs = []
        
        try:
            if self.single_file_mode:
                # Load from single file
                if self.file_path.exists():
                    try:
                        with open(self.file_path, 'r') as f:
                            if self.file_path.suffix == '.json':
                                data = json.load(f)
                            else:  # yaml
                                data = yaml.safe_load(f)
                        
                        # Handle both single config and list of configs
                        if isinstance(data, dict) and "servers" in data:
                            # Data has 'servers' wrapper
                            for item in data["servers"]:
                                config = MCPServerConfig.from_dict(item)
                                configs.append(config)
                        elif isinstance(data, list):
                            # Data is directly a list of configs
                            for item in data:
                                config = MCPServerConfig.from_dict(item)
                                configs.append(config)
                        else:
                            # Data is a single config
                            config = MCPServerConfig.from_dict(data)
                            configs.append(config)
                            
                    except Exception as e:
                        self.logger.error(f"Failed to load config from {self.file_path}: {e}")
            else:
                # Load from directory
                for config_file in self.config_dir.glob("*.json"):
                    try:
                        with open(config_file, 'r') as f:
                            data = json.load(f)
                        
                        # Handle both single config and list of configs, including 'servers' wrapper
                        if isinstance(data, dict) and "servers" in data:
                            # Data has 'servers' wrapper
                            for item in data["servers"]:
                                config = MCPServerConfig.from_dict(item)
                                configs.append(config)
                        elif isinstance(data, list):
                            # Data is directly a list of configs
                            for item in data:
                                config = MCPServerConfig.from_dict(item)
                                configs.append(config)
                        else:
                            # Data is a single config
                            config = MCPServerConfig.from_dict(data)
                            configs.append(config)
                        
                    except Exception as e:
                        self.logger.error(f"Failed to load config from {config_file}: {e}")
                
                # Also try YAML files
                for config_file in self.config_dir.glob("*.yaml"):
                    try:
                        with open(config_file, 'r') as f:
                            data = yaml.safe_load(f)
                        
                        # Handle both single config and list of configs, including 'servers' wrapper
                        if isinstance(data, dict) and "servers" in data:
                            # Data has 'servers' wrapper
                            for item in data["servers"]:
                                config = MCPServerConfig.from_dict(item)
                                configs.append(config)
                        elif isinstance(data, list):
                            # Data is directly a list of configs
                            for item in data:
                                config = MCPServerConfig.from_dict(item)
                                configs.append(config)
                        else:
                            # Data is a single config
                            config = MCPServerConfig.from_dict(data)
                            configs.append(config)
                        
                    except Exception as e:
                        self.logger.error(f"Failed to load config from {config_file}: {e}")
            
            self.logger.info(f"Loaded {len(configs)} server configurations")
            
        except Exception as e:
            self.logger.error(f"Failed to load configurations: {e}")
        
        return configs
    
    async def save_config(self, config: MCPServerConfig) -> bool:
        """Save configuration to file."""
        try:
            if self.single_file_mode:
                # Ensure parent directory exists
                self.config_dir.mkdir(parents=True, exist_ok=True)
                
                # Load existing configs, update/add the new one, and save all
                existing_configs = await self.load_configs()
                
                # Update or add the config
                updated = False
                for i, existing_config in enumerate(existing_configs):
                    if existing_config.name == config.name:
                        existing_configs[i] = config
                        updated = True
                        break
                
                if not updated:
                    existing_configs.append(config)
                
                # Save all configs to the single file with 'servers' wrapper
                config_data = [cfg.to_dict(include_sensitive=True) for cfg in existing_configs]
                data = {"servers": config_data}
                
                with open(self.file_path, 'w') as f:
                    if self.file_path.suffix == '.json':
                        json.dump(data, f, indent=2)
                    else:  # yaml
                        yaml.dump(data, f, default_flow_style=False)
            else:
                # Ensure directory exists
                self.config_dir.mkdir(parents=True, exist_ok=True)
                
                # Save to individual file in directory
                config_file = self.config_dir / f"{config.name}.json"
                
                with open(config_file, 'w') as f:
                    json.dump(config.to_dict(include_sensitive=True), f, indent=2)
            
            self.logger.info(f"Saved configuration for server: {config.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save config for {config.name}: {e}")
            return False
    
    async def save_configs(self, configs: List[MCPServerConfig]) -> bool:
        """Save multiple configurations to file(s)."""
        try:
            if self.single_file_mode:
                # Ensure parent directory exists
                self.config_dir.mkdir(parents=True, exist_ok=True)
                
                # Save all configs to the single file with 'servers' wrapper
                config_data = [cfg.to_dict(include_sensitive=True) for cfg in configs]
                data = {"servers": config_data}
                
                with open(self.file_path, 'w') as f:
                    if self.file_path.suffix == '.json':
                        json.dump(data, f, indent=2)
                    else:  # yaml
                        yaml.dump(data, f, default_flow_style=False)
            else:
                # Ensure directory exists
                self.config_dir.mkdir(parents=True, exist_ok=True)
                
                # Save each config to individual file in directory
                for config in configs:
                    config_file = self.config_dir / f"{config.name}.json"
                    
                    with open(config_file, 'w') as f:
                        json.dump(config.to_dict(include_sensitive=True), f, indent=2)
            
            self.logger.info(f"Saved {len(configs)} configurations")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configs: {e}")
            return False
    
    async def delete_config(self, server_name: str) -> bool:
        """Delete configuration file."""
        try:
            if self.single_file_mode:
                # Load existing configs, remove the specified one, and save the rest
                existing_configs = await self.load_configs()
                
                # Find and remove the config
                original_count = len(existing_configs)
                existing_configs = [cfg for cfg in existing_configs if cfg.name != server_name]
                
                if len(existing_configs) == original_count:
                    # Config not found
                    return False
                
                # Save remaining configs to the single file with 'servers' wrapper
                config_data = [cfg.to_dict(include_sensitive=True) for cfg in existing_configs]
                data = {"servers": config_data}
                
                with open(self.file_path, 'w') as f:
                    if self.file_path.suffix == '.json':
                        json.dump(data, f, indent=2)
                    else:  # yaml
                        yaml.dump(data, f, default_flow_style=False)
                        
                self.logger.info(f"Deleted configuration for server: {server_name}")
                return True
            else:
                # Delete individual file in directory
                config_file = self.config_dir / f"{server_name}.json"
                if config_file.exists():
                    config_file.unlink()
                    self.logger.info(f"Deleted configuration for server: {server_name}")
                    return True
                
                # Try YAML file
                config_file = self.config_dir / f"{server_name}.yaml"
                if config_file.exists():
                    config_file.unlink()
                    self.logger.info(f"Deleted configuration for server: {server_name}")
                    return True
                
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to delete config for {server_name}: {e}")
            return False


class EnvironmentConfigProvider(ServerConfigProvider):
    """Environment variable-based server configuration provider."""
    
    def __init__(self, prefix: str = "ARK_MCP_"):
        """
        Initialize environment config provider.
        
        Args:
            prefix: Environment variable prefix for server configs
        """
        self.prefix = prefix
        self.name = "environment_provider"
        self.logger = logging.getLogger('jarvis.server_config.env')
    
    async def load_configs(self) -> List[MCPServerConfig]:
        """Load configurations from environment variables."""
        configs = []
        
        try:
            # Look for environment variables with the prefix
            env_vars = {k: v for k, v in os.environ.items() if k.startswith(self.prefix)}
            
            # Group by server name (format: PREFIX_SERVERNAME_PROPERTY)
            server_data = {}
            for key, value in env_vars.items():
                parts = key[len(self.prefix):].split('_', 1)
                if len(parts) >= 2:
                    server_name, property_name = parts[0].lower(), parts[1].lower()
                    
                    if server_name not in server_data:
                        server_data[server_name] = {}
                    
                    server_data[server_name][property_name] = value
            
            # Create configs from environment data
            for server_name, data in server_data.items():
                try:
                    # Map environment variables to config properties
                    config_data = {
                        'name': server_name,
                        'server_type': ServerType.HTTP.value,  # Default type
                        'config_type': ConfigType.DEVELOPMENT.value,  # Default config type
                        'host': data.get('host', 'localhost'),
                        'port': int(data.get('port', 8080)),
                        'protocol': data.get('protocol', 'http'),
                        'enabled': data.get('enabled', 'true').lower() == 'true',
                        'auth_type': AuthType.NONE.value,  # Default auth
                        'security_level': SecurityLevel.MEDIUM.value if SecurityLevel else 'medium',  # Default security
                        'timeout': int(data.get('timeout', 30)),
                        'max_retries': int(data.get('max_retries', 3)),
                        'description': data.get('description', f'Environment config for {server_name}')
                    }
                    
                    # Handle optional fields
                    if 'auth_token' in data:
                        config_data['auth_token'] = data['auth_token']
                        config_data['auth_type'] = AuthType.BEARER_TOKEN.value
                    
                    if 'api_key' in data:
                        config_data['api_key'] = data['api_key']
                        config_data['auth_type'] = AuthType.API_KEY.value
                    
                    config = MCPServerConfig.from_dict(config_data)
                    configs.append(config)
                    
                except Exception as e:
                    self.logger.error(f"Failed to create config for {server_name}: {e}")
            
            self.logger.info(f"Loaded {len(configs)} server configurations from environment")
            
        except Exception as e:
            self.logger.error(f"Failed to load configurations from environment: {e}")
        
        return configs
    
    async def save_config(self, config: MCPServerConfig) -> bool:
        """Save configuration is not supported for environment provider."""
        raise NotImplementedError("Save operation not supported for EnvironmentConfigProvider")
    
    async def save_configs(self, configs: List[MCPServerConfig]) -> bool:
        """Save configurations is not supported for environment provider."""
        raise NotImplementedError("Save operation not supported for EnvironmentConfigProvider")
    
    async def delete_config(self, server_name: str) -> bool:
        """Delete configuration is not supported for environment provider."""
        raise NotImplementedError("Delete operation not supported for EnvironmentConfigProvider")


class ARKServerConfigManager:
    """
    Comprehensive MCP server configuration manager for the ARK engine.
    
    Manages server configurations, validation, discovery, and monitoring
    for all MCP servers in the system.
    """
    
    def __init__(self, config_provider: Optional[ServerConfigProvider] = None):
        """
        Initialize server configuration manager.
        
        Args:
            config_provider: Configuration provider (defaults to FileConfigProvider)
        """
        self.logger = logging.getLogger('jarvis.server_config')
        
        # Configuration storage
        self.configs: Dict[str, MCPServerConfig] = {}
        self.config_provider = config_provider or FileConfigProvider()
        
        # Multiple providers support
        self.providers: List[ServerConfigProvider] = []
        
        # Event callbacks
        self.on_config_added: List[Callable] = []
        self.on_config_updated: List[Callable] = []
        self.on_config_removed: List[Callable] = []
        self.on_status_changed: List[Callable] = []
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Start the configuration manager."""
        self.logger.info("Starting ARK Server Config Manager...")
        self._running = True
        
        # Load existing configurations
        await self.load_configs()
        
        # Start health monitoring
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        self.logger.info("ARK Server Config Manager started")
    
    async def stop(self) -> None:
        """Stop the configuration manager."""
        self.logger.info("Stopping ARK Server Config Manager...")
        self._running = False
        
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("ARK Server Config Manager stopped")
    
    async def load_configs(self) -> int:
        """
        Load configurations from provider.
        
        Returns:
            Number of configurations loaded
        """
        try:
            configs = await self.config_provider.load_configs()
            
            for config in configs:
                if config.is_valid():
                    self.configs[config.name] = config
                    self.logger.info(f"Loaded valid config: {config.name}")
                else:
                    _, errors = config.validate()
                    self.logger.warning(f"Invalid config {config.name}: {errors}")
            
            self.logger.info(f"Loaded {len(self.configs)} valid configurations")
            return len(self.configs)
            
        except Exception as e:
            self.logger.error(f"Failed to load configurations: {e}")
            return 0
    
    def add_config(self, config: MCPServerConfig) -> bool:
        """
        Add a new server configuration.
        
        Args:
            config: Server configuration to add
            
        Returns:
            True if added successfully
        """
        # Validate configuration
        if not config.is_valid():
            _, errors = config.validate()
            self.logger.error(f"Invalid configuration for {config.name}: {errors}")
            return False
        
        # Check for duplicates
        if config.name in self.configs:
            self.logger.warning(f"Configuration already exists: {config.name}")
            return False
        
        # Add to memory storage (always succeed for in-memory operations)
        self.configs[config.name] = config
        self.logger.info(f"Added configuration: {config.name}")
        
        # Trigger callbacks
        for callback in self.on_config_added:
            try:
                callback(config)
            except Exception as e:
                self.logger.warning(f"Config added callback failed: {e}")
        
        # Try to save to provider (optional, don't fail if it doesn't work)
        try:
            # Note: We don't await here since this method is now synchronous
            # Provider saving is handled separately in async methods
            pass
        except Exception as e:
            self.logger.warning(f"Failed to save configuration {config.name} to provider: {e}")
        
        return True
    
    def update_config(self, server_name_or_config, **kwargs) -> bool:
        """
        Update an existing server configuration.
        
        Args:
            server_name_or_config: Either server name (str) or MCPServerConfig object
            **kwargs: Configuration attributes to update (when server_name is provided)
            
        Returns:
            True if updated successfully
        """
        # Handle both server name and config object
        if isinstance(server_name_or_config, str):
            server_name = server_name_or_config
            if server_name not in self.configs:
                self.logger.error(f"Configuration not found: {server_name}")
                return False
            
            # Create a copy of the existing config for modification
            import copy
            config = copy.deepcopy(self.configs[server_name])
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    self.logger.warning(f"Unknown configuration attribute: {key}")
        else:
            config = server_name_or_config
            server_name = config.name
        
        # Validate configuration
        if not config.is_valid():
            _, errors = config.validate()
            self.logger.error(f"Invalid configuration for {config.name}: {errors}")
            return False
        
        # Check if exists
        if config.name not in self.configs:
            self.logger.error(f"Configuration not found: {config.name}")
            return False
        
        try:
            # Update timestamp
            config.last_updated = time.time()
            
            # Update in memory first
            old_config = self.configs[config.name]
            self.configs[config.name] = config
            self.logger.info(f"Updated configuration: {config.name}")
            
            # Trigger callbacks
            for callback in self.on_config_updated:
                try:
                    callback(old_config, config)
                except Exception as e:
                    self.logger.warning(f"Config updated callback failed: {e}")
            
            # Try to save to provider (optional, don't fail if it doesn't work)
            try:
                # Note: Not awaiting since this is now a sync method
                pass  # self.config_provider.save_config(config)
            except Exception as e:
                self.logger.warning(f"Failed to save configuration {config.name} to provider: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update configuration {config.name}: {e}")
        
        return False
    
    def validate_config(self, config: MCPServerConfig) -> tuple[bool, List[str]]:
        """
        Validate a server configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, errors) where errors is a list of error messages
        """
        try:
            # Use the config's own validate method
            is_valid, errors = config.validate()
            
            return is_valid, errors
            
        except Exception as e:
            return False, [f"Validation failed: {str(e)}"]
    
    def remove_config(self, server_name: str) -> bool:
        """
        Remove a server configuration.
        
        Args:
            server_name: Name of server to remove
            
        Returns:
            True if removed successfully
        """
        if server_name not in self.configs:
            self.logger.warning(f"Configuration not found: {server_name}")
            return False
        
        try:
            # Remove from memory storage (always succeed for in-memory operations)
            config = self.configs.pop(server_name)
            self.logger.info(f"Removed configuration: {server_name}")
            
            # Trigger callbacks
            for callback in self.on_config_removed:
                try:
                    callback(config)
                except Exception as e:
                    self.logger.warning(f"Config removed callback failed: {e}")
            
            # Try to delete from provider (optional, don't fail if it doesn't work)
            try:
                # Note: Not awaiting since this is now a sync method
                # Provider deletion is handled separately in async methods
                pass
            except Exception as e:
                self.logger.warning(f"Failed to delete configuration {server_name} from provider: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove configuration {server_name}: {e}")
            return False
    
    def add_provider(self, provider: ServerConfigProvider) -> None:
        """
        Add a configuration provider.
        
        Args:
            provider: Configuration provider to add
        """
        if provider not in self.providers:
            self.providers.append(provider)
            self.logger.info(f"Added configuration provider: {provider.__class__.__name__}")
    
    def remove_provider(self, provider_name: str) -> bool:
        """
        Remove a configuration provider by name.
        
        Args:
            provider_name: Name of the provider to remove
            
        Returns:
            True if removed successfully
        """
        for provider in self.providers:
            if hasattr(provider, 'name') and provider.name == provider_name:
                self.providers.remove(provider)
                self.logger.info(f"Removed configuration provider: {provider_name}")
                return True
        return False
    
    async def load_from_providers(self) -> List[MCPServerConfig]:
        """
        Load configurations from all providers.
        
        Returns:
            List of loaded configurations
        """
        all_configs = []
        
        for provider in self.providers:
            try:
                configs = await provider.load_configs()
                for config in configs:
                    if config.is_valid():
                        self.configs[config.name] = config
                        all_configs.append(config)
                        self.logger.info(f"Loaded config from provider: {config.name}")
                    else:
                        _, errors = config.validate()
                        self.logger.warning(f"Invalid config from provider {config.name}: {errors}")
            except Exception as e:
                self.logger.error(f"Failed to load from provider {provider.__class__.__name__}: {e}")
        
        return all_configs
    
    async def save_to_providers(self) -> bool:
        """
        Save configurations to all providers.
        
        Returns:
            True if all saves were successful
        """
        success = True
        configs_list = list(self.configs.values())
        
        for provider in self.providers:
            try:
                if hasattr(provider, 'save_configs'):
                    await provider.save_configs(configs_list)
                else:
                    # Save individual configs if provider doesn't support batch save
                    for config in configs_list:
                        await provider.save_config(config)
                self.logger.info(f"Saved configs to provider: {provider.__class__.__name__}")
            except Exception as e:
                self.logger.error(f"Failed to save to provider {provider.__class__.__name__}: {e}")
                success = False
        
        return success
    
    def get_config(self, server_name: str) -> Optional[MCPServerConfig]:
        """
        Get server configuration by name.
        
        Args:
            server_name: Name of the server
            
        Returns:
            Server configuration if found
        """
        return self.configs.get(server_name)
    
    def get_all_configs(self) -> List[MCPServerConfig]:
        """
        Get all server configurations.
        
        Returns:
            List of all configurations
        """
        return list(self.configs.values())
    
    def get_enabled_configs(self) -> List[MCPServerConfig]:
        """
        Get enabled server configurations.
        
        Returns:
            List of enabled configurations
        """
        return [config for config in self.configs.values() if config.enabled]
    
    def get_configs_by_status(self, status: ServerStatus) -> List[MCPServerConfig]:
        """
        Get configurations by status.
        
        Args:
            status: Server status to filter by
            
        Returns:
            List of configurations with the specified status
        """
        return [config for config in self.configs.values() if config.status == status]
    
    def get_configs_by_type(self, server_type: ServerType) -> List[MCPServerConfig]:
        """
        Get configurations by server type.
        
        Args:
            server_type: Server type to filter by
            
        Returns:
            List of configurations with the specified type
        """
        return [config for config in self.configs.values() if config.server_type == server_type]
    
    def update_server_status(self, server_name: str, status: ServerStatus) -> bool:
        """
        Update server status.
        
        Args:
            server_name: Name of the server
            status: New status
            
        Returns:
            True if updated successfully
        """
        if server_name not in self.configs:
            return False
        
        config = self.configs[server_name]
        old_status = config.status
        config.update_status(status)
        
        self.logger.info(f"Server {server_name} status changed: {old_status.value} -> {status.value}")
        
        # Trigger callbacks
        for callback in self.on_status_changed:
            try:
                callback(config, old_status, status)
            except Exception as e:
                self.logger.warning(f"Status changed callback failed: {e}")
        
        return True
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """
        Get configuration manager statistics.
        
        Returns:
            Dictionary with manager statistics
        """
        total_configs = len(self.configs)
        enabled_configs = len(self.get_enabled_configs())
        
        # Status distribution
        status_counts = {}
        for config in self.configs.values():
            status = config.status.value.upper()
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Type distribution
        type_counts = {}
        for config in self.configs.values():
            server_type = config.server_type.value.upper()
            type_counts[server_type] = type_counts.get(server_type, 0) + 1
        
        # Security level distribution
        security_level_counts = {}
        for config in self.configs.values():
            if hasattr(config, 'security_level') and config.security_level:
                security_level = config.security_level.value.upper()
                security_level_counts[security_level] = security_level_counts.get(security_level, 0) + 1
        
        return {
            'total_configs': total_configs,
            'enabled_configs': enabled_configs,
            'disabled_configs': total_configs - enabled_configs,
            'status_distribution': status_counts,
            'type_distribution': type_counts,
            'security_level_distribution': security_level_counts,
            'provider_type': self.config_provider.__class__.__name__
        }
    
    async def _health_check_loop(self) -> None:
        """Background task for server health monitoring."""
        self.logger.info("Started server health monitoring")
        
        while self._running:
            try:
                # Check health of all enabled servers
                for config in self.get_enabled_configs():
                    if config.health_check.enabled:
                        await self._check_server_health(config)
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(30)  # Wait before retrying
        
        self.logger.info("Server health monitoring stopped")
    
    async def _check_server_health(self, config: MCPServerConfig) -> None:
        """
        Check health of a specific server.
        
        Args:
            config: Server configuration to check
        """
        # This is a placeholder for actual health checking
        # In a real implementation, this would ping the server
        # or check its health endpoint
        
        try:
            # Simulate health check
            await asyncio.sleep(0.1)
            
            # Update metrics (placeholder)
            if config.status == ServerStatus.CONNECTED:
                # Server is healthy
                pass
            else:
                # Server might be unhealthy
                self.logger.debug(f"Server {config.name} health check: {config.status.value}")
            
        except Exception as e:
            self.logger.warning(f"Health check failed for {config.name}: {e}")
    
    async def _health_check(self, config: MCPServerConfig) -> None:
        """
        Perform health check on a specific server configuration.
        
        This method is used by tests and provides a simple interface
        for checking server health status.
        
        Args:
            config: Server configuration to check
        """
        await self._check_server_health(config)
    
    def __len__(self) -> int:
        """Return number of configurations."""
        return len(self.configs)
    
    def __contains__(self, server_name: str) -> bool:
        """Check if server configuration exists."""
        return server_name in self.configs
    
    def __iter__(self):
        """Iterate over server names."""
        return iter(self.configs.keys())
    
    def __repr__(self) -> str:
        """String representation of the manager."""
        return f"ARKServerConfigManager(configs={len(self.configs)})"
    
    def get_config_count(self, enabled_only: bool = False) -> int:
        """
        Get the total number of configurations.
        
        Args:
            enabled_only: If True, count only enabled configurations
            
        Returns:
            Total number of configurations
        """
        if enabled_only:
            return len([config for config in self.configs.values() if config.enabled])
        return len(self.configs)
    
    def get_config_types(self) -> Set[ConfigType]:
        """
        Get all unique configuration types in configurations.
        
        Returns:
            Set of configuration types
        """
        return {config.config_type for config in self.configs.values()}
    
    def get_security_levels(self) -> Set[SecurityLevel]:
        """
        Get all unique security levels in configurations.
        
        Returns:
            Set of security levels
        """
        if SecurityLevel is None:
            return set()
        
        security_levels = set()
        for config in self.configs.values():
            if hasattr(config, 'security_level') and config.security_level:
                security_levels.add(config.security_level)
        return security_levels
    
    def clear_configs(self) -> None:
        """
        Clear all configurations from the manager.
        """
        self.configs.clear()
        self.logger.info("All configurations cleared")
    
    def list_configs(self, enabled_only: bool = False, config_type: ConfigType = None, 
                    security_level: SecurityLevel = None) -> List[MCPServerConfig]:
        """
        List configurations with optional filtering.
        
        Args:
            enabled_only: If True, only return enabled configurations
            config_type: Filter by configuration type
            security_level: Filter by security level
            
        Returns:
            List of configurations matching the criteria
        """
        configs = list(self.configs.values())
        
        if enabled_only:
            configs = [config for config in configs if config.enabled]
            
        if config_type is not None:
            configs = [config for config in configs if config.config_type == config_type]
            
        if security_level is not None:
            configs = [config for config in configs if config.security_level == security_level]
            
        return configs
    
    async def health_check(self) -> Dict[str, Dict[str, Any]]:
        """
        Perform health check on all enabled servers.
        
        Returns:
            Dictionary mapping server names to their health status
        """
        import aiohttp
        
        health_status = {}
        
        # Only check enabled servers
        enabled_configs = [config for config in self.configs.values() if config.enabled]
        
        async with aiohttp.ClientSession() as session:
            for config in enabled_configs:
                try:
                    # Construct health check URL
                    health_url = f"{config.protocol}://{config.host}:{config.port}/health"
                    
                    async with session.get(health_url, timeout=config.timeout) as response:
                        if response.status == 200:
                            health_status[config.name] = {
                                "healthy": True,
                                "status_code": response.status,
                                "response_time": 0.1  # Placeholder
                            }
                        else:
                            health_status[config.name] = {
                                "healthy": False,
                                "status_code": response.status,
                                "error": f"HTTP {response.status}"
                            }
                            
                except Exception as e:
                    health_status[config.name] = {
                        "healthy": False,
                        "error": str(e)
                    }
        
        return health_status