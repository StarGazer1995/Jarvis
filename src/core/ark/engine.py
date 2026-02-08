"""
ARK (Autonomous Reasoning Kernel) Engine

This is the core reasoning engine that powers the Jarvis AI agent.
ARK integrates MCP tools, intent recognition, context management,
and decision-making capabilities.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Import generic agent framework
from ..agent.react import ReActAgent
from ..agent.types import AgentState, AgentStep

from ..mcp.client import ARKMCPClient
from ..config.server import SimpleMCPServerConfig
from ..context.manager import ConversationContext
from ..llm.client import LLMManager
from ..llm.config import load_llm_config
from ..prompt.manager import PromptManager


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
            
            # Initialize performance tracking
            self._initialize_performance_metrics()
            
            self.state = ARKState.READY
            self.ark_logger.info(f"ARK: Initialization complete - {len(self.available_tools)} tools available")
            return True
            
        except Exception as e:
            self.state = ARKState.ERROR
            self.ark_logger.error(f"ARK: Initialization failed: {e}")
            return False
            
    async def process_input(self, user_input: str) -> str:
        """Process user input."""
        # Override to add ARK-specific context updates
        response = await super().process_input(user_input)
        
        # Update Context
        if self.state != ARKState.ERROR:
            try:
                self.context_manager.add_exchange(
                    user_input=user_input,
                    agent_response=response,
                    intent="react_execution",
                    tools_used=[t.description for t in self.todo_list], # simplified
                    metadata={"todo_count": len(self.todo_list)}
                )
            except Exception as e:
                self.ark_logger.warning(f"Failed to update context: {e}")
                
        return response

    def _get_system_prompt(self) -> str:
        """Generate the system prompt for ReAct agent."""
        
        # 1. Todo List Status
        todo_status = "No tasks in todo list."
        if self.todo_list:
            todo_status = "Current Todo List:\n"
            for task in self.todo_list:
                todo_status += f"- [{task.id}] {task.status.value}: {task.description}"
                if task.result:
                    todo_status += f" (Result: {task.result})"
                todo_status += "\n"
        
        # 2. Available Tools
        tools_desc = "Available Tools:\n"
        
        # Internal Tools
        tools_desc += "- manage_tasks: Manage the todo list. Input: {\"action\": \"add\"|\"update\"|\"complete\", ...}\n"
        
        # External Tools
        for name, info in self.available_tools.items():
            desc = info.get('description', 'No description')
            tools_desc += f"- {name}: {desc}\n"
            
        return f"""You are Jarvis, an intelligent agent using the ReAct framework.

{todo_status}

{tools_desc}

Instructions:
1. Analyze the user's request.
2. Break it down into a list of tasks using 'manage_tasks' if needed.
3. Execute tasks one by one.
4. Use available tools to gather information or perform actions.
5. Update task status as you progress.
6. When finished, provide a Final Answer.

Format your response as follows:

Thought: <your reasoning>
Action: <tool_name>
Action Input: <json_or_string_input>

OR

Thought: <your reasoning>
Final Answer: <your final response to the user>
"""

    async def execute_tool(self, name: str, params: Any) -> Any:
        """Execute an action (tool or internal)."""
        
        if name == "manage_tasks":
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except:
                    return "Error: Action Input for manage_tasks must be valid JSON."
            return self._manage_tasks(**params)
            
        # Check external tools
        if name in self.available_tools:
            # Need to ensure parameters is a dict
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except:
                    return f"Error: Tool arguments for {name} must be a JSON object."
            
            try:
                result = await self.mcp_client.execute_tool(name, params)
                # Update stats
                self.tool_usage_stats[name] = self.tool_usage_stats.get(name, 0) + 1
                return result
            except Exception as e:
                return f"Error executing tool {name}: {e}"
                
        return f"Error: Unknown tool '{name}'."

    def _manage_tasks(self, action: str, **kwargs) -> str:
        """Manage the todo list."""
        if action == "add":
            description = kwargs.get("description")
            if not description:
                return "Error: Description required for adding task."
            task_id = str(len(self.todo_list) + 1)
            task = Task(id=task_id, description=description)
            self.todo_list.append(task)
            return f"Task added: [{task_id}] {description}"
            
        elif action == "update":
            task_id = kwargs.get("id") or kwargs.get("task_id")
            status = kwargs.get("status")
            result = kwargs.get("result")
            
            task = next((t for t in self.todo_list if t.id == str(task_id)), None)
            if not task:
                return f"Error: Task {task_id} not found."
                
            if status:
                try:
                    task.status = TaskStatus(status)
                except ValueError:
                    return f"Error: Invalid status {status}."
            
            if result:
                task.result = result
                
            return f"Task {task_id} updated."
            
        elif action == "complete":
            task_id = kwargs.get("id") or kwargs.get("task_id")
            result = kwargs.get("result")
            
            task = next((t for t in self.todo_list if t.id == str(task_id)), None)
            if not task:
                return f"Error: Task {task_id} not found."
                
            task.status = TaskStatus.COMPLETED
            if result:
                task.result = result
                
            return f"Task {task_id} completed."
            
        return f"Error: Unknown action {action}."
    
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
