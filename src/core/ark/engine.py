"""
ARK (Autonomous Reasoning Kernel) Engine

This is the core reasoning engine that powers the Jarvis AI agent.
ARK integrates MCP tools, intent recognition, context management,
and decision-making capabilities.
"""

import logging
import time
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

# Import generic agent framework
from ..agent.react import ReActAgent
from ..agent.types import AgentState, AgentStep
from ..config.server import SimpleMCPServerConfig
from ..context.manager import ConversationContext
from ..llm.client import LLMManager
from ..llm.config import load_llm_config
from ..mcp.client import ARKMCPClient

# Observability
from ..observability import MetricsRegistry
from ..prompt.manager import PromptManager

# Security
from ..security.manager import ARKSecurityManager

# LangGraph imports
from .graph import create_ark_graph
from .mcp_lifecycle import close_mcp_client, connect_servers, discover_tools
from .nodes.tools import ToolsNode
from .tasks import Task, TaskStatus, tasks_from_dicts, tasks_to_dicts

# Aliases for backward compatibility
ARKState = AgentState
ReActStep = AgentStep
__all__ = [
    "ARKEngine",
    "Task",
    "TaskStatus",
    "ARKState",
    "ReActStep",
    "LLMManager",
    "load_llm_config",
    "PromptManager",
]


class ARKEngine(ReActAgent):
    """
    Autonomous Reasoning Kernel (ARK) Engine

    The core reasoning engine that orchestrates all AI agent capabilities
    including intent recognition, tool selection, context management,
    and response generation.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the ARK engine.

        Args:
            config: Configuration dictionary for ARK engine
        """
        super().__init__(config)
        self.ark_logger = logging.getLogger("ark.engine")

        # Initialize core components
        self.mcp_client = ARKMCPClient()
        self.context_manager = ConversationContext(
            max_history=self.config.get("max_conversation_history", 100)
        )

        # Security manager
        self.security_manager = ARKSecurityManager()

        # ARK-specific attributes
        self.tool_usage_stats: dict[str, int] = {}
        self.todo_list: list[Task] = []
        self._engine_start_time: float = time.time()

        # Configuration
        self.max_tool_chain_length = self.config.get("max_tool_chain_length", 5)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.enable_tool_chaining = self.config.get("enable_tool_chaining", True)

        # LangGraph
        self.graph = None

        # Initialize performance tracking
        self._initialize_performance_metrics()

        # Initialize Prometheus observability
        try:
            from ..observability import MetricsRegistry, observability_server

            MetricsRegistry.init_engine_info()
            self._metrics = MetricsRegistry
            self._observability = observability_server
            self._prometheus_available = True
        except ImportError:
            self._prometheus_available = False
            self._metrics = None
            self._observability = None

        self.ark_logger.info(
            "ARK engine initialized - Autonomous Reasoning Kernel ready"
        )

    async def initialize(
        self, mcp_servers: list[SimpleMCPServerConfig] | None = None
    ) -> bool:
        """
        Initialize ARK engine with MCP servers and capabilities.
        """
        # Call base initialize first (handles LLM)
        if not await super().initialize():
            return False

        try:
            self.ark_logger.info("ARK: Starting initialization sequence")

            await connect_servers(
                self.mcp_client,
                self.ark_logger,
                mcp_servers,
                "ARK: Connected to MCP server '{name}'",
                "ARK: Failed to connect to MCP server '{name}'",
            )

            # Discover available tools
            await self._discover_tools()

            # Initialize LangGraph
            self.ark_logger.info("ARK: Initializing LangGraph workflow")

            # Create ToolsNode with security manager
            self.tools_node = ToolsNode(
                self.mcp_client,
                security_manager=self.security_manager,
            )
            self.graph = create_ark_graph(
                self.llm_manager, self.mcp_client, tools_node_instance=self.tools_node
            )

            # Initialize performance tracking
            self._initialize_performance_metrics()

            # Mark engine as ready in observability
            try:
                MetricsRegistry.mark_engine_ready()
            except Exception:
                pass

            self.state = ARKState.READY
            available_tool_count = len(self.available_tools)
            self.ark_logger.info(
                f"ARK: Initialization complete - {available_tool_count} tools available",
                extra={
                    "component": "ark.engine",
                    "event": "initialization_complete",
                    "status": "success",
                    "tool_count": available_tool_count,
                },
            )
            return True

        except Exception as e:
            self.state = ARKState.ERROR
            self.ark_logger.error(
                f"ARK: Initialization failed: {e}",
                extra={
                    "component": "ark.engine",
                    "event": "initialization_failed",
                    "status": "error",
                    "error": str(e),
                },
            )
            # Log stack trace for debugging
            import traceback

            self.ark_logger.error(traceback.format_exc())
            return False

    async def process_input(
        self, user_input: str, callbacks: dict[str, Callable] | None = None, **kwargs
    ) -> str:
        """Process user input using LangGraph."""
        if self.state != ARKState.READY:
            return "ARK Engine is not ready."

        self.ark_logger.info(
            f"ARK: Processing input via LangGraph: '{user_input[:50]}...'"
        )

        # 1. Prepare Initial State
        history_messages = self.context_manager.get_cleaned_history(max_messages=20)
        session_id = self.context_manager.session_id
        self.ark_logger.info(
            f"ARK: Retrieved {len(history_messages)} history messages "
            f"for session {session_id}"
        )

        initial_state = {
            "messages": history_messages + [HumanMessage(content=user_input)],
            "user_input": user_input,
            "todo_list": tasks_to_dicts(self.todo_list),
            "available_tools": self.available_tools,
            "scratchpad": {},
            "sender": "user",
            "iteration_count": 0,
            "termination_reason": None,
        }

        try:
            # 2. Execute Graph
            if not self.graph:
                return "Error: LangGraph not initialized."

            self.ark_logger.info(
                "ARK: Starting graph execution",
                extra={
                    "component": "ark.engine",
                    "event": "graph_execution_start",
                    "status": "started",
                    "session_id": session_id,
                },
            )
            try:
                MetricsRegistry.graph_iterations_total.labels(node="master").inc()
            except Exception:
                pass

            run_config = (
                RunnableConfig(configurable={"callbacks": callbacks})
                if callbacks
                else {}
            )
            final_state = await self.graph.ainvoke(initial_state, config=run_config)

            self.ark_logger.info(
                "ARK: Graph execution completed",
                extra={
                    "component": "ark.engine",
                    "event": "graph_execution_complete",
                    "status": "success",
                    "session_id": session_id,
                },
            )

            # 3. Update internal state (todo list)
            new_todo_list_dicts = final_state.get("todo_list", [])
            self.todo_list = tasks_from_dicts(new_todo_list_dicts)

            # 4. Extract Final Response
            messages = final_state.get("messages", [])
            logging.info(f"ARK: Final messages after graph execution: {messages}")
            response = "No response generated."

            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, AIMessage):
                    response = last_msg.content
                elif hasattr(last_msg, "content"):
                    response = str(last_msg.content)
                else:
                    response = str(last_msg)

            # 5. Update Context (Legacy)
            if self.state != ARKState.ERROR:
                try:
                    # Try to get raw JSON response from additional_kwargs
                    raw_response = None
                    if messages:
                        last_msg = messages[-1]
                        if isinstance(last_msg, AIMessage):
                            raw_response = last_msg.additional_kwargs.get("raw_json")

                    self.ark_logger.info(
                        f"ARK: Updating context for session {session_id} with "
                        f"response len {len(response)}"
                    )
                    self.context_manager.add_exchange(
                        user_input=user_input,
                        agent_response=response,
                        intent="langgraph_execution",
                        tools_used=[
                            t.description for t in self.todo_list
                        ],  # simplified
                        metadata={"todo_count": len(self.todo_list)},
                        raw_response=raw_response,
                    )
                except Exception as e:
                    self.ark_logger.warning(f"Failed to update context: {e}")
            else:
                self.ark_logger.warning(
                    f"ARK: State is ERROR ({self.state}), skipping context update."
                )

            return response

        except Exception as e:
            self.ark_logger.error(
                f"ARK: Graph execution failed: {e}",
                extra={
                    "component": "ark.engine",
                    "event": "graph_execution_failed",
                    "status": "error",
                    "error": str(e),
                    "session_id": session_id,
                },
            )
            try:
                MetricsRegistry.record_error(
                    component="ark.engine", error_type=type(e).__name__
                )
            except Exception:
                pass
            import traceback

            self.ark_logger.error(traceback.format_exc())
            return f"Error executing request: {str(e)}"

    def _get_system_prompt(self) -> str:
        """Legacy method, kept for compatibility if needed."""
        # This logic is now moved to MasterNode, but we keep it here just in case
        return super()._get_system_prompt()

    async def execute_tool(self, name: str, params: Any) -> Any:
        """Legacy method. Tools are now executed by ToolsNode."""
        # We might still need this if something calls execute_tool directly
        return await super().execute_tool(name, params)

    def _manage_tasks(self, action: str, **kwargs) -> str:
        """Legacy method. Task management is now in ToolsNode."""
        return "Legacy _manage_tasks called. Please use LangGraph flow."

    async def _discover_tools(self) -> None:
        """Discover and catalog available tools from MCP servers."""
        try:
            self.available_tools = await discover_tools(
                self.mcp_client,
                self.ark_logger,
                "unknown_tool_{index}",
                "ARK: Failed to discover tools from server '{server_name}': {error}",
            )

            self.ark_logger.info(f"ARK: Discovered {len(self.available_tools)} tools")

            # Initialize usage stats for all tools
            for tool_name in self.available_tools.keys():
                if tool_name not in self.tool_usage_stats:
                    self.tool_usage_stats[tool_name] = 0

        except Exception as e:
            self.ark_logger.error(f"ARK: Tool discovery failed: {e}")
            self.available_tools = {}

    def _initialize_performance_metrics(self) -> None:
        """Initialize performance tracking metrics."""
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "tool_executions": 0,
            "average_response_time": 0.0,
            "intent_accuracy": 0.0,
            "tool_success_rate": 0.0,
            "start_time": self._get_timestamp(),
        }

    async def _start_observability(self) -> None:
        """Start the Prometheus metrics server if not already running."""
        if (
            self._prometheus_available
            and self._observability
            and not self._observability.is_running
        ):
            try:
                await self._observability.start()
            except Exception as e:
                self.ark_logger.warning(f"Failed to start observability server: {e}")

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime

        return datetime.now().isoformat()

    def get_status(self) -> dict[str, Any]:
        """Get basic engine status."""
        return {
            "state": self.state.value,
            "available_tools": len(self.available_tools),
            "todo_count": len(self.todo_list),
            "security_policies": list(self.security_manager.policies.keys()),
            "engine_uptime_seconds": time.time() - self._engine_start_time,
        }

    def get_engine_status(self) -> dict[str, Any]:
        """Get current ARK engine status and metrics."""
        return {
            "state": self.state.value,
            "available_tools": list(self.available_tools.keys()),
            "tool_usage_stats": self.tool_usage_stats,
            "performance_metrics": self.performance_metrics,
            "conversation_stats": self.context_manager.get_session_stats(),
            "todo_list": tasks_to_dicts(self.todo_list),
            "configuration": {
                "max_tool_chain_length": self.max_tool_chain_length,
                "confidence_threshold": self.confidence_threshold,
                "enable_tool_chaining": self.enable_tool_chaining,
                "graph_iteration_limit": self.graph.get_graph(xray=True).nodes.keys()
                if self.graph
                else 0,
            },
        }

    async def close(self) -> None:
        """Alias for shutdown."""
        await self.shutdown()

    async def shutdown(self) -> None:
        """Shutdown the ARK engine gracefully."""
        # Call base shutdown
        await super().shutdown()

        self.ark_logger.info("ARK: Initiating shutdown sequence")

        # Close MCP client connections
        try:
            await close_mcp_client(self.mcp_client)
        except Exception as e:
            self.ark_logger.warning(f"ARK: Error during MCP client shutdown: {e}")

        # Export conversation if needed
        try:
            self.context_manager.export_conversation()
            self.ark_logger.debug("ARK: Conversation exported for archival")
        except Exception as e:
            self.ark_logger.warning(f"ARK: Error exporting conversation: {e}")

        self.ark_logger.info("ARK: Shutdown complete")
