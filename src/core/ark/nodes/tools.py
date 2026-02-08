import json
import logging
import asyncio
from typing import Dict, Any, List
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from ..state import JarvisState
from ...mcp.client import ARKMCPClient

logger = logging.getLogger("ark.nodes.tools")

class ToolsNode:
    """
    Node responsible for executing tool calls (both internal and MCP).
    """
    def __init__(self, mcp_client: ARKMCPClient):
        self.mcp_client = mcp_client
        self.local_tools: Dict[str, Any] = {}

    def register_tool(self, name: str, func: Any):
        """Register a local tool function."""
        self.local_tools[name] = func

    async def __call__(self, state: JarvisState, config: RunnableConfig) -> Dict[str, Any]:
        """
        Execute tools requested in the last message.
        """
        last_message = state["messages"][-1]
        
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            logger.warning("ToolsNode called but no tool_calls found in last message.")
            return {"sender": "tools"} 
            
        tool_calls = last_message.tool_calls
        results = []
        # Copy list to ensure immutability if needed, though TypedDict is mutable
        updated_todo_list = [t.copy() for t in state.get("todo_list", [])]
        
        for tool_call in tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            tool_call_id = tool_call["id"]
            
            logger.info(f"Executing tool: {name}")
            
            try:
                result = None
                
                # 1. Internal Hardcoded Tool: manage_tasks
                if name == "manage_tasks":
                    # Ensure args is dict
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            pass
                    
                    if isinstance(args, dict):
                        result = self._manage_tasks(args, updated_todo_list)
                    else:
                        result = "Error: manage_tasks arguments must be a dictionary/JSON."
                
                # 2. Registered Local Tools
                elif name in self.local_tools:
                    func = self.local_tools[name]
                    if asyncio.iscoroutinefunction(func):
                        result = await func(**args) if isinstance(args, dict) else await func(args)
                    else:
                        result = func(**args) if isinstance(args, dict) else func(args)

                # 3. External Tools: MCP
                else:
                    # Execute MCP tool
                    result = await self.mcp_client.execute_tool(name, args)
                    
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                result = f"Error executing tool {name}: {str(e)}"
                
            results.append(ToolMessage(
                tool_call_id=tool_call_id,
                name=name,
                content=str(result)
            ))
            
        return {
            "messages": results,
            "sender": "tools",
            "todo_list": updated_todo_list # Update state with modified list
        }

    def _manage_tasks(self, args: Dict, todo_list: List[Dict]) -> str:
        """
        Manage the todo list and return a status message.
        Modifies todo_list in-place.
        """
        action = args.get("action")
        
        if action == "add":
            description = args.get("description")
            if not description:
                return "Error: Description required for adding task."
            task_id = str(len(todo_list) + 1)
            task = {
                "id": task_id,
                "description": description,
                "status": "pending",
                "result": None
            }
            todo_list.append(task)
            return f"Task added: [{task_id}] {description}"
            
        elif action == "update":
            task_id = str(args.get("id") or args.get("task_id"))
            status = args.get("status")
            result = args.get("result")
            
            task = next((t for t in todo_list if t["id"] == task_id), None)
            if not task:
                return f"Error: Task {task_id} not found."
                
            if status:
                task["status"] = status
            if result:
                task["result"] = result
                
            return f"Task {task_id} updated."
            
        elif action == "complete":
            task_id = str(args.get("id") or args.get("task_id"))
            result = args.get("result")
            
            task = next((t for t in todo_list if t["id"] == task_id), None)
            if not task:
                return f"Error: Task {task_id} not found."
                
            task["status"] = "completed"
            if result:
                task["result"] = result
                
            return f"Task {task_id} completed."
            
        return f"Error: Unknown action {action}."
