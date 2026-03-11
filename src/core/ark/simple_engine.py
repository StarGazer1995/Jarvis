"""
Simple ARK Engine (Single ReAct Agent Mode)

This module implements a standalone ReAct agent that bypasses the
multi-agent supervisor graph for simpler, single-threaded execution.
"""

import logging
from typing import Any

from ..agent.react import ReActAgent
from ..agent.types import AgentState
from ..config.server import SimpleMCPServerConfig
from ..context.manager import ConversationContext
from ..mcp.client import ARKMCPClient
from ..prompt.manager import PromptManager
from .mcp_lifecycle import close_mcp_client, connect_servers, discover_tools
from .tasks import (
    Task,
    execute_manage_tasks,
    tasks_from_dicts,
    tasks_to_dicts,
)


class SimpleARKEngine(ReActAgent):
    """
    A simplified version of ARK Engine that uses a single ReAct loop
    instead of the LangGraph supervisor system.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.ark_logger = logging.getLogger("ark.simple_engine")

        # Initialize core components
        self.mcp_client = ARKMCPClient()
        self.context_manager = ConversationContext(
            max_history=self.config.get("max_conversation_history", 100)
        )

        self.prompt_manager = PromptManager()

        # ARK-specific attributes
        self.tool_usage_stats: dict[str, int] = {}
        self.todo_list: list[Task] = []

        self.ark_logger.info("Simple ARK Engine initialized")

    async def initialize(
        self, mcp_servers: list[SimpleMCPServerConfig] | None = None
    ) -> bool:
        """Initialize the engine with MCP servers."""
        if not await super().initialize():
            return False

        try:
            self.ark_logger.info("SimpleARK: Starting initialization")

            await connect_servers(
                self.mcp_client,
                self.ark_logger,
                mcp_servers,
                "SimpleARK: Connected to '{name}'",
                "SimpleARK: Failed to connect to '{name}'",
            )

            # Discover tools
            await self._discover_tools()

            self.state = AgentState.READY
            return True

        except Exception as e:
            self.state = AgentState.ERROR
            self.ark_logger.error(f"SimpleARK initialization failed: {e}")
            return False

    async def _discover_tools(self) -> None:
        """Discover tools from MCP servers."""
        try:
            self.available_tools = await discover_tools(
                self.mcp_client,
                self.ark_logger,
                "unknown_{index}",
                "Error discovering tools from {server_name}: {error}",
            )

            # Add built-in task management tool
            self.available_tools["manage_tasks"] = {
                "name": "manage_tasks",
                "description": (
                    "Manage the todo list (add, update, complete tasks). "
                    "Params: action (add/update/complete), description, id, status"
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["add", "update", "complete"],
                        },
                        "description": {"type": "string"},
                        "id": {"type": "string"},
                        "status": {"type": "string"},
                    },
                },
            }

            self.ark_logger.info(
                f"SimpleARK: Discovered {len(self.available_tools)} tools"
            )

        except Exception as e:
            self.ark_logger.error(f"Tool discovery failed: {e}")
            self.available_tools = {}

    async def execute_tool(self, name: str, params: Any) -> Any:
        """Execute a tool using the MCP client."""
        # Check if it's a built-in task management tool
        if name == "manage_tasks":
            return self._handle_manage_tasks(params)

        return await self.mcp_client.execute_tool(name, params)

    def _handle_manage_tasks(self, params: Any) -> str:
        """Handle internal task management."""
        todo_dicts = tasks_to_dicts(self.todo_list)
        result = execute_manage_tasks(params, todo_dicts)
        self.todo_list = tasks_from_dicts(todo_dicts)
        return result

    async def shutdown(self) -> None:
        """Shutdown resources."""
        await super().shutdown()
        await close_mcp_client(self.mcp_client)
