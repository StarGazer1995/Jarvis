import asyncio
import logging
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from ...mcp.client import ARKMCPClient
from ..state import JarvisState
from ..tasks import execute_manage_tasks

logger = logging.getLogger("ark.nodes.tools")


class ToolsNode:
    """
    Node responsible for executing tool calls (both internal and MCP).
    """

    def __init__(self, mcp_client: ARKMCPClient):
        self.mcp_client = mcp_client
        self.local_tools: dict[str, Any] = {}

    def register_tool(self, name: str, func: Any):
        """Register a local tool function."""
        self.local_tools[name] = func

    async def _invoke_local_tool(self, func: Any, args: Any) -> Any:
        if asyncio.iscoroutinefunction(func):
            return await (func(**args) if isinstance(args, dict) else func(args))
        return func(**args) if isinstance(args, dict) else func(args)

    async def _execute_tool_call(
        self, name: str, args: Any, updated_todo_list: list[dict[str, Any]]
    ) -> Any:
        if name == "manage_tasks":
            return execute_manage_tasks(
                args,
                updated_todo_list,
                invalid_params_message=(
                    "Error: manage_tasks arguments must be a dictionary/JSON."
                ),
            )
        if name in self.local_tools:
            return await self._invoke_local_tool(self.local_tools[name], args)
        return await self.mcp_client.execute_tool(name, args)

    async def __call__(
        self, state: JarvisState, config: RunnableConfig
    ) -> dict[str, Any]:
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
                result = await self._execute_tool_call(name, args, updated_todo_list)

            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                result = f"Error executing tool {name}: {str(e)}"

            results.append(
                ToolMessage(tool_call_id=tool_call_id, name=name, content=str(result))
            )

        return {
            "messages": results,
            "sender": "tools",
            "todo_list": updated_todo_list,  # Update state with modified list
        }
