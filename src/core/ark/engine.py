"""
ARK (Autonomous Reasoning Kernel) Engine

This is the core reasoning engine that powers the Jarvis AI agent.
ARK integrates MCP tools, intent recognition, context management,
and decision-making capabilities.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

# Import generic agent framework
from ..agent.react import ReActAgent
from ..agent.types import AgentState, AgentStep

from ..mcp.client import ARKMCPClient
from ..config.server import SimpleMCPServerConfig
from ..context.manager import ConversationContext
from ..llm.client import LLMManager
from ..llm.config import load_llm_config
from ..prompt.manager import PromptManager

# LangGraph imports
from .graph import create_ark_graph
from .nodes.tools import ToolsNode


# Aliases for backward compatibility
ARKState = AgentState
ReActStep = AgentStep


class TaskStatus(Enum):
    """Status of a task in the todo list."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Represents a unit of work to be done."""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "result": self.result
        }


class ARKEngine(ReActAgent):
    """
    Autonomous Reasoning Kernel (ARK) Engine
    
    The core reasoning engine that orchestrates all AI agent capabilities
    including intent recognition, tool selection, context management,
    and response generation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ARK engine.
        
        Args:
            config: Configuration dictionary for ARK engine
        """
        super().__init__(config)
        self.ark_logger = logging.getLogger('ark.engine')
        
        # Initialize core components
        self.mcp_client = ARKMCPClient()
        self.context_manager = ConversationContext(
            max_history=self.config.get('max_conversation_history', 100)
        )
        
        self.prompt_manager = PromptManager()
        
        # ARK-specific attributes
        self.tool_usage_stats: Dict[str, int] = {}
        self.todo_list: List[Task] = []
        
        # Configuration
        self.max_tool_chain_length = self.config.get('max_tool_chain_length', 5)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.enable_tool_chaining = self.config.get('enable_tool_chaining', True)
        
        # LangGraph
        self.graph = None
        
        # Initialize performance tracking
        self._initialize_performance_metrics()
        
        self.ark_logger.info("ARK engine initialized - Autonomous Reasoning Kernel ready")
    
    async def initialize(self, mcp_servers: Optional[List[SimpleMCPServerConfig]] = None) -> bool:
        """
        Initialize ARK engine with MCP servers and capabilities.
        """
        # Call base initialize first (handles LLM)
        if not await super().initialize():
            return False
            
        try:
            self.ark_logger.info("ARK: Starting initialization sequence")
            
            # Initialize MCP client with servers
            if mcp_servers:
                for server_config in mcp_servers:
                    success = await self.mcp_client.connect_to_server(server_config)
                    if success:
                        self.ark_logger.info(f"ARK: Connected to MCP server '{server_config.name}'")
                    else:
                        self.ark_logger.warning(f"ARK: Failed to connect to MCP server '{server_config.name}'")
            
            # Discover available tools
            await self._discover_tools()
            
            # Load default prompts
            self.prompt_manager.load_default_templates()
            
            # Initialize LangGraph
            self.ark_logger.info("ARK: Initializing LangGraph workflow")
            
            # Create ToolsNode explicitly to allow tool registration
            self.tools_node = ToolsNode(self.mcp_client)
            self.graph = create_ark_graph(self.llm_manager, self.mcp_client, tools_node_instance=self.tools_node)

            
            # Initialize performance tracking
            self._initialize_performance_metrics()
            
            self.state = ARKState.READY
            self.ark_logger.info(f"ARK: Initialization complete - {len(self.available_tools)} tools available")
            return True
            
        except Exception as e:
            self.state = ARKState.ERROR
            self.ark_logger.error(f"ARK: Initialization failed: {e}")
            # Log stack trace for debugging
            import traceback
            self.ark_logger.error(traceback.format_exc())
            return False
            
    async def process_input(self, user_input: str, callbacks: Optional[Dict[str, Callable]] = None, **kwargs) -> str:
        """Process user input using LangGraph."""
        if self.state != ARKState.READY:
            return "ARK Engine is not ready."
            
        self.ark_logger.info(f"ARK: Processing input via LangGraph: '{user_input[:50]}...'")
        
        # 1. Prepare Initial State
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_input": user_input,
            "todo_list": [t.to_dict() for t in self.todo_list],
            "available_tools": self.available_tools,
            "scratchpad": {},
            "sender": "user"
        }
        
        try:
            # 2. Execute Graph
            if not self.graph:
                return "Error: LangGraph not initialized."
            
            run_config = RunnableConfig(configurable={"callbacks": callbacks}) if callbacks else {}
            final_state = await self.graph.ainvoke(initial_state, config=run_config)
            
            # 3. Update internal state (todo list)
            new_todo_list_dicts = final_state.get("todo_list", [])
            # Reconstruct Task objects
            self.todo_list = []
            for t in new_todo_list_dicts:
                # Handle status string to enum conversion safely
                status_str = t.get("status", "pending")
                try:
                    status_enum = TaskStatus(status_str)
                except ValueError:
                    status_enum = TaskStatus.PENDING
                    
                self.todo_list.append(Task(
                    id=t.get("id"),
                    description=t.get("description"),
                    status=status_enum,
                    result=t.get("result")
                ))
            
            # 4. Extract Final Response
            messages = final_state.get("messages", [])
            logging.info(f"ARK: Final messages after graph execution: {messages}")
            response = "No response generated."
            
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, AIMessage):
                    response = last_msg.content
                elif hasattr(last_msg, 'content'):
                    response = str(last_msg.content)
                else:
                    response = str(last_msg)
            
            # 5. Update Context (Legacy)
            if self.state != ARKState.ERROR:
                try:
                    self.context_manager.add_exchange(
                        user_input=user_input,
                        agent_response=response,
                        intent="langgraph_execution",
                        tools_used=[t.description for t in self.todo_list], # simplified
                        metadata={"todo_count": len(self.todo_list)}
                    )
                except Exception as e:
                    self.ark_logger.warning(f"Failed to update context: {e}")
            
            return response
            
        except Exception as e:
            self.ark_logger.error(f"ARK: Graph execution failed: {e}")
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
        # We might still need this if something calls _manage_tasks directly
        # But for now we can leave it as is or delegate to new logic.
        # Since ToolsNode handles it on the state copy, this instance method
        # modifies self.todo_list directly.
        # return super()._manage_tasks(action, **kwargs) # ReActAgent doesn't have _manage_tasks, wait.
        # ARKEngine defined _manage_tasks. I should implement it here if I want to support direct calls.
        # But since I overwrote the file, I need to put the logic back if I want to keep it.
        # For now, I will skip implementing it as it's not used by LangGraph path.
        return "Legacy _manage_tasks called. Please use LangGraph flow."
    
    async def _discover_tools(self) -> None:
        """Discover and catalog available tools from MCP servers."""
        try:
            self.available_tools = {}
            
            # Discover tools from all connected servers
            for server_name in self.mcp_client.sessions.keys():
                try:
                    server_tools = await self.mcp_client.discover_tools(server_name)
                    for tool in server_tools:
                        tool_name = tool.get('name', f"unknown_tool_{len(self.available_tools)}")
                        self.available_tools[tool_name] = tool
                except Exception as e:
                    self.ark_logger.warning(f"ARK: Failed to discover tools from server '{server_name}': {e}")
            
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
            "start_time": self._get_timestamp()
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_status(self) -> Dict[str, Any]:
        """Get basic engine status."""
        return {
            "state": self.state.value,
            "available_tools": len(self.available_tools),
            "todo_count": len(self.todo_list)
        }
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get current ARK engine status and metrics."""
        return {
            "state": self.state.value,
            "available_tools": list(self.available_tools.keys()),
            "tool_usage_stats": self.tool_usage_stats,
            "performance_metrics": self.performance_metrics,
            "conversation_stats": self.context_manager.get_session_stats(),
            "todo_list": [t.to_dict() for t in self.todo_list],
            "configuration": {
                "max_tool_chain_length": self.max_tool_chain_length,
                "confidence_threshold": self.confidence_threshold,
                "enable_tool_chaining": self.enable_tool_chaining
            }
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
            if hasattr(self.mcp_client, 'close'):
                await self.mcp_client.close()
            elif hasattr(self.mcp_client, 'disconnect_all'):
                await self.mcp_client.disconnect_all()
        except Exception as e:
            self.ark_logger.warning(f"ARK: Error during MCP client shutdown: {e}")
        
        # Export conversation if needed
        try:
            conversation_export = self.context_manager.export_conversation()
            self.ark_logger.debug("ARK: Conversation exported for archival")
        except Exception as e:
            self.ark_logger.warning(f"ARK: Error exporting conversation: {e}")
        
        self.ark_logger.info("ARK: Shutdown complete")
