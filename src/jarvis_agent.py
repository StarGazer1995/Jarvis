"""
Jarvis AI Agent - Main Agent Class

This module implements the main JarvisAgent class that integrates
the ARK (Autonomous Reasoning Kernel) engine with a conversational
AI interface.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .core.ark.engine import ARKEngine, ARKState
from .core.config.server import SimpleMCPServerConfig


@dataclass
class JarvisConfig:
    """Configuration for Jarvis Agent."""

    name: str = "Jarvis"
    version: str = "1.0.0"
    log_level: str = "INFO"
    max_conversation_history: int = 100
    enable_tool_chaining: bool = True
    confidence_threshold: float = 0.7
    mcp_servers: list[SimpleMCPServerConfig] = None

    def __post_init__(self):
        if self.mcp_servers is None:
            self.mcp_servers = []


class JarvisAgent:
    """
    Jarvis AI Agent - ARK-Powered Conversational Assistant

    This is the main agent class that provides a conversational interface
    to the ARK (Autonomous Reasoning Kernel) engine. It handles user
    interactions, manages conversation flow, and coordinates with
    various AI capabilities.
    """

    def __init__(self, config: JarvisConfig | None = None):
        """
        Initialize Jarvis Agent.

        Args:
            config: Configuration for the agent
        """
        self.config = config or JarvisConfig()

        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger("jarvis.agent")

        # Initialize ARK engine
        ark_config = {
            "max_conversation_history": self.config.max_conversation_history,
            "enable_tool_chaining": self.config.enable_tool_chaining,
            "confidence_threshold": self.config.confidence_threshold,
        }
        self.ark_engine = ARKEngine(ark_config)

        # Agent state
        self.is_running = False
        self.is_initialized = False
        self.conversation_active = False

        # Callbacks for extensibility
        self.on_startup_callbacks: list[Callable] = []
        self.on_shutdown_callbacks: list[Callable] = []
        self.on_message_callbacks: list[Callable] = []

        self.logger.info(f"Jarvis Agent v{self.config.version} initialized")

    def _setup_logging(self) -> None:
        """Setup logging configuration for Jarvis with structured format."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)

        # Define a structured formatter that includes extra fields
        class StructuredFormatter(logging.Formatter):
            """Formatter that appends structured extra fields as JSON."""

            def format(self, record: logging.LogRecord) -> str:
                base = super().format(record)
                # Append extra fields as JSON suffix if present
                extra_fields = {
                    k: v
                    for k, v in vars(record).items()
                    if k
                    not in (
                        "args",
                        "asctime",
                        "created",
                        "exc_info",
                        "exc_text",
                        "filename",
                        "funcName",
                        "levelname",
                        "levelno",
                        "lineno",
                        "message",
                        "module",
                        "msecs",
                        "msg",
                        "name",
                        "pathname",
                        "process",
                        "processName",
                        "relativeCreated",
                        "stack_info",
                        "thread",
                        "threadName",
                    )
                    and k.startswith(("component", "event", "status", "error"))
                }
                if extra_fields:
                    import json

                    base += " " + json.dumps(extra_fields)
                return base

        # Configure root logger
        handler = logging.StreamHandler()
        handler.setFormatter(
            StructuredFormatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        # Remove existing handlers and add our structured one
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)
        root_logger.addHandler(handler)

        # Set specific logger levels
        logging.getLogger("jarvis").setLevel(log_level)
        logging.getLogger("ark").setLevel(log_level)

    async def initialize(self) -> bool:
        """
        Initialize Jarvis Agent and ARK engine.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Jarvis Agent...")

            # Initialize ARK engine
            success = await self.ark_engine.initialize(self.config.mcp_servers)
            if not success:
                self.logger.error("Failed to initialize ARK engine")
                return False

            # Run startup callbacks
            for callback in self.on_startup_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(self)
                    else:
                        callback(self)
                except Exception as e:
                    self.logger.warning(f"Startup callback failed: {e}")

            self.is_initialized = True
            self.logger.info("Jarvis Agent initialization complete")
            return True

        except Exception as e:
            self.logger.error(f"Jarvis Agent initialization failed: {e}")
            return False

    def register_capabilities(self, capabilities: list[Any]) -> None:
        """
        Register capabilities with the ARK engine.

        Args:
            capabilities: List of capability objects that implement get_tools()
        """
        if not self.is_initialized:
            self.logger.warning(
                "Agent not initialized. Capabilities might be overwritten during initialization."
            )

        for capability in capabilities:
            if not hasattr(capability, "get_tools"):
                self.logger.warning(
                    f"Capability {capability} does not implement get_tools()"
                )
                continue

            tools = capability.get_tools()
            for tool_key, tool_def in tools.items():
                func = tool_def.get("func")
                schema = tool_def.get("schema")

                if func and schema:
                    tool_name = schema["name"]
                    # Register function logic
                    if hasattr(self.ark_engine, "tools_node"):
                        self.ark_engine.tools_node.register_tool(tool_name, func)

                    # Register schema
                    self.ark_engine.available_tools[tool_name] = schema
                    self.logger.info(f"Registered tool: {tool_name}")

    async def start(self) -> None:
        """Start Jarvis Agent."""
        if not self.is_initialized:
            success = await self.initialize()
            if not success:
                raise RuntimeError("Failed to initialize Jarvis Agent")

        self.is_running = True
        self.logger.info("Jarvis Agent started and ready for interaction")

    async def stop(self) -> None:
        """Stop Jarvis Agent."""
        self.logger.info("Stopping Jarvis Agent...")

        self.is_running = False
        self.conversation_active = False

        # Run shutdown callbacks
        for callback in self.on_shutdown_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self)
                else:
                    callback(self)
            except Exception as e:
                self.logger.warning(f"Shutdown callback failed: {e}")

        # Shutdown ARK engine
        await self.ark_engine.shutdown()

        self.logger.info("Jarvis Agent stopped")

    async def process_message(
        self,
        message: str,
        user_id: str | None = None,
        callbacks: dict[str, Callable] | None = None,
    ) -> str:
        """
        Process a user message and return a response.

        Args:
            message: User's message
            user_id: Optional user identifier
            callbacks: Optional callbacks for streaming/events

        Returns:
            Agent's response
        """
        if not self.is_running:
            return "I'm not currently running. Please start me first."

        if not message or not message.strip():
            return "I didn't receive any message. Could you please say something?"

        try:
            self.logger.debug(
                f"Processing message from user {user_id}: '{message[:50]}...'"
            )

            # Run message callbacks
            for callback in self.on_message_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(self, message, user_id)
                    else:
                        callback(self, message, user_id)
                except Exception as e:
                    self.logger.warning(f"Message callback failed: {e}")

            # Process through ARK engine
            response = await self.ark_engine.process_input(message, callbacks=callbacks)

            self.logger.debug(f"Generated response: '{response[:50]}...'")
            return response

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return f"I encountered an error while processing your message: {str(e)}"

    async def start_conversation(self, user_id: str | None = None) -> str:
        """
        Start a new conversation session.

        Args:
            user_id: Optional user identifier

        Returns:
            Welcome message
        """
        if not self.is_running:
            await self.start()

        self.conversation_active = True

        # Reset conversation context for new session
        self.ark_engine.context_manager.reset_session()

        welcome_message = (
            f"Hello! I'm {self.config.name}, your AI assistant powered by the ARK engine. "
            "I'm here to help you with questions, tasks, and various requests. "
            "What can I do for you today?"
        )

        self.logger.info(f"Started conversation for user {user_id}")
        return welcome_message

    async def end_conversation(self, user_id: str | None = None) -> str:
        """
        End the current conversation session.

        Args:
            user_id: Optional user identifier

        Returns:
            Goodbye message
        """
        self.conversation_active = False

        # Get conversation statistics
        stats = self.ark_engine.context_manager.get_session_stats()

        goodbye_message = (
            "Thank you for the conversation! "
            f"We exchanged {stats['turn_count']} messages over "
            f"{stats['duration_minutes']:.1f} minutes. "
            "Feel free to start a new conversation anytime!"
        )

        self.logger.info(f"Ended conversation for user {user_id}")
        return goodbye_message

    def add_mcp_server(self, server_config: SimpleMCPServerConfig) -> None:
        """
        Add an MCP server configuration.

        Args:
            server_config: MCP server configuration
        """
        self.config.mcp_servers.append(server_config)
        self.logger.info(f"Added MCP server configuration: {server_config.name}")

    def remove_mcp_server(self, server_name: str) -> bool:
        """
        Remove an MCP server configuration.

        Args:
            server_name: Name of the server to remove

        Returns:
            True if server was removed, False if not found
        """
        for i, server in enumerate(self.config.mcp_servers):
            if server.name == server_name:
                del self.config.mcp_servers[i]
                self.logger.info(f"Removed MCP server configuration: {server_name}")
                return True

        self.logger.warning(f"MCP server not found: {server_name}")
        return False

    def add_startup_callback(self, callback: Callable) -> None:
        """
        Add a callback to be executed on startup.

        Args:
            callback: Function to call on startup
        """
        self.on_startup_callbacks.append(callback)
        self.logger.debug("Added startup callback")

    def add_shutdown_callback(self, callback: Callable) -> None:
        """
        Add a callback to be executed on shutdown.

        Args:
            callback: Function to call on shutdown
        """
        self.on_shutdown_callbacks.append(callback)
        self.logger.debug("Added shutdown callback")

    def add_message_callback(self, callback: Callable) -> None:
        """
        Add a callback to be executed on each message.

        Args:
            callback: Function to call on each message
        """
        self.on_message_callbacks.append(callback)
        self.logger.debug("Added message callback")

    def get_status(self) -> dict[str, Any]:
        """
        Get current agent status.

        Returns:
            Dictionary with agent status information
        """
        ark_status = self.ark_engine.get_engine_status()

        return {
            "agent": {
                "name": self.config.name,
                "version": self.config.version,
                "is_running": self.is_running,
                "is_initialized": self.is_initialized,
                "conversation_active": self.conversation_active,
            },
            "configuration": {
                "log_level": self.config.log_level,
                "max_conversation_history": self.config.max_conversation_history,
                "enable_tool_chaining": self.config.enable_tool_chaining,
                "confidence_threshold": self.config.confidence_threshold,
                "mcp_servers_count": len(self.config.mcp_servers),
            },
            "ark_engine": ark_status,
        }

    def get_conversation_export(self, format: str = "json") -> str:
        """
        Export current conversation.

        Args:
            format: Export format ("json" or "text")

        Returns:
            Exported conversation data
        """
        return self.ark_engine.context_manager.export_conversation(format)

    def get_available_tools(self) -> list[str]:
        """
        Get list of available tools.

        Returns:
            List of available tool names
        """
        return list(self.ark_engine.available_tools.keys())

    def get_tool_usage_stats(self) -> dict[str, int]:
        """
        Get tool usage statistics.

        Returns:
            Dictionary of tool names to usage counts
        """
        return self.ark_engine.tool_usage_stats.copy()

    def set_user_preference(self, key: str, value: Any) -> None:
        """
        Set a user preference.

        Args:
            key: Preference key
            value: Preference value
        """
        self.ark_engine.context_manager.update_user_preference(key, value)
        self.logger.debug(f"Set user preference: {key} = {value}")

    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """
        Get a user preference.

        Args:
            key: Preference key
            default: Default value if not found

        Returns:
            Preference value or default
        """
        return self.ark_engine.context_manager.get_user_preference(key, default)

    async def health_check(self) -> dict[str, Any]:
        """
        Perform a health check of the agent and its components.

        Returns:
            Health check results
        """
        health_status = {
            "overall": "healthy",
            "components": {},
            "timestamp": self._get_timestamp(),
        }

        # Check agent status
        health_status["components"]["agent"] = {
            "status": "healthy"
            if self.is_running and self.is_initialized
            else "unhealthy",
            "is_running": self.is_running,
            "is_initialized": self.is_initialized,
        }

        # Check ARK engine status
        ark_state = self.ark_engine.state
        health_status["components"]["ark_engine"] = {
            "status": "healthy" if ark_state == ARKState.READY else "unhealthy",
            "state": ark_state.value,
            "available_tools": len(self.ark_engine.available_tools),
        }

        # Check MCP client status
        try:
            mcp_tools = await self.ark_engine.mcp_client.list_tools()
            health_status["components"]["mcp_client"] = {
                "status": "healthy",
                "connected_servers": len(self.ark_engine.mcp_client.sessions),
                "available_tools": len(mcp_tools),
            }
        except Exception as e:
            health_status["components"]["mcp_client"] = {
                "status": "unhealthy",
                "error": str(e),
            }

        # Determine overall health
        component_statuses = [
            comp["status"] for comp in health_status["components"].values()
        ]
        if all(status == "healthy" for status in component_statuses):
            health_status["overall"] = "healthy"
        elif any(status == "healthy" for status in component_statuses):
            health_status["overall"] = "degraded"
        else:
            health_status["overall"] = "unhealthy"

        return health_status

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime

        return datetime.now().isoformat()

    def __repr__(self) -> str:
        """String representation of Jarvis Agent."""
        return (
            f"JarvisAgent(name='{self.config.name}', version='{self.config.version}', "
            f"running={self.is_running}, initialized={self.is_initialized})"
        )

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
