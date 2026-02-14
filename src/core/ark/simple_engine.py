"""
Simple ARK Engine (Single ReAct Agent Mode)

This module implements a standalone ReAct agent that bypasses the
multi-agent supervisor graph for simpler, single-threaded execution.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from ..agent.react import ReActAgent
from ..agent.types import AgentState

from ..mcp.client import ARKMCPClient
from ..config.server import SimpleMCPServerConfig
from ..context.manager import ConversationContext
from ..prompt.manager import PromptManager

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

class SimpleARKEngine(ReActAgent):
    """
    A simplified version of ARK Engine that uses a single ReAct loop
    instead of the LangGraph supervisor system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.ark_logger = logging.getLogger('ark.simple_engine')
        
        # Initialize core components
        self.mcp_client = ARKMCPClient()
        self.context_manager = ConversationContext(
            max_history=self.config.get('max_conversation_history', 100)
        )
        
        self.prompt_manager = PromptManager()
        
        # ARK-specific attributes
        self.tool_usage_stats: Dict[str, int] = {}
        self.todo_list: List[Task] = []
        
        self.ark_logger.info("Simple ARK Engine initialized")

    async def initialize(self, mcp_servers: Optional[List[SimpleMCPServerConfig]] = None) -> bool:
        """Initialize the engine with MCP servers."""
        if not await super().initialize():
            return False
            
        try:
            self.ark_logger.info("SimpleARK: Starting initialization")
            
            # Initialize MCP client
            if mcp_servers:
                for server_config in mcp_servers:
                    success = await self.mcp_client.connect_to_server(server_config)
                    if success:
                        self.ark_logger.info(f"SimpleARK: Connected to '{server_config.name}'")
                    else:
                        self.ark_logger.warning(f"SimpleARK: Failed to connect to '{server_config.name}'")
            
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
            self.available_tools = {}
            for server_name in self.mcp_client.sessions.keys():
                try:
                    server_tools = await self.mcp_client.discover_tools(server_name)
                    for tool in server_tools:
                        tool_name = tool.get('name', f"unknown_{len(self.available_tools)}")
                        self.available_tools[tool_name] = tool
                except Exception as e:
                    self.ark_logger.warning(f"Error discovering tools from {server_name}: {e}")
            
            # Add built-in task management tool
            self.available_tools["manage_tasks"] = {
                "name": "manage_tasks",
                "description": "Manage the todo list (add, update, complete tasks). Params: action (add/update/complete), description, id, status",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["add", "update", "complete"]},
                        "description": {"type": "string"},
                        "id": {"type": "string"},
                        "status": {"type": "string"}
                    }
                }
            }
            
            self.ark_logger.info(f"SimpleARK: Discovered {len(self.available_tools)} tools")
            
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
        if isinstance(params, str):
            import json
            try:
                params = json.loads(params)
            except:
                return "Error: Invalid params format"
                
        action = params.get("action")
        
        if action == "add":
            description = params.get("description")
            if not description:
                return "Error: Description required for add action"
            
            task_id = str(len(self.todo_list) + 1)
            task = Task(id=task_id, description=description)
            self.todo_list.append(task)
            return f"Task added: {task_id} - {description}"
            
        elif action == "complete":
            task_id = params.get("id")
            for task in self.todo_list:
                if task.id == task_id:
                    task.status = TaskStatus.COMPLETED
                    return f"Task completed: {task_id}"
            return f"Error: Task {task_id} not found"
            
        return f"Action {action} not supported yet"
        
    async def shutdown(self) -> None:
        """Shutdown resources."""
        await super().shutdown()
        if hasattr(self.mcp_client, 'close'):
            await self.mcp_client.close()
        elif hasattr(self.mcp_client, 'disconnect_all'):
            await self.mcp_client.disconnect_all()
