import asyncio
import logging
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from ...execution.engine import ParallelExecutor, ToolCall
from ...mcp.client import ARKMCPClient
from ..state import JarvisState
from ..tasks import execute_manage_tasks

logger = logging.getLogger("ark.nodes.tools")


class ToolsNode:
    """
    Node responsible for executing tool calls (both internal and MCP).

    Supports parallel execution: independent tool calls within a single
    LLM response are executed concurrently via ``asyncio.gather``.
    Tool calls can declare dependencies via a ``depends_on`` field
    in their arguments (parsed from extended tool_call metadata).
    """

    def __init__(self, mcp_client: ARKMCPClient, max_concurrency: int = 10):
        """
        Args:
            mcp_client: The MCP client for executing remote tools.
            max_concurrency: Maximum number of parallel tool executions.
        """
        self.mcp_client = mcp_client
        self.local_tools: dict[str, Any] = {}
        self.max_concurrency = max_concurrency

    def register_tool(self, name: str, func: Any):
        """Register a local tool function."""
        self.local_tools[name] = func

    async def _invoke_local_tool(self, func: Any, args: Any) -> Any:
        if asyncio.iscoroutinefunction(func):
            return await (func(**args) if isinstance(args, dict) else func(args))
        return func(**args) if isinstance(args, dict) else func(args)

    async def _execute_tool_call(
        self,
        name: str,
        args: Any,
        updated_todo_list: list[dict[str, Any]] | None = None,
    ) -> Any:
        if name == "manage_tasks":
            if updated_todo_list is None:
                updated_todo_list = []
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

        Tool calls from the LLM are converted to ``ToolCall`` objects.
        If multiple tool calls are present and have no dependencies between
        them, they are executed in parallel.
        """
        last_message = state["messages"][-1]

        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            logger.warning("ToolsNode called but no tool_calls found in last message.")
            return {"sender": "tools"}

        tool_calls_data = last_message.tool_calls
        updated_todo_list = [t.copy() for t in state.get("todo_list", [])]

        tool_calls: list[ToolCall] = []
        for tc in tool_calls_data:
            name = tc["name"]
            args = dict(tc.get("args", {}))
            tool_call_id = tc.get("id", f"{name}_{len(tool_calls)}")

            # Extract dependency info if present (e.g., from LLM that supports
            # multi-tool responses with dependency annotations)
            depends_on = self._extract_depends_on(args)

            tool_calls.append(
                ToolCall(
                    name=name,
                    arguments=args,
                    id=tool_call_id,
                    depends_on=depends_on,
                )
            )

        # Execute using the ParallelExecutor
        executor = ParallelExecutor(
            execute_fn=lambda n, a: self._execute_tool_call(n, a, updated_todo_list),
            max_concurrency=self.max_concurrency,
        )

        try:
            exec_results = await executor.run(tool_calls)
        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            # Fall back to sequential execution on error
            exec_results = await self._run_sequential(tool_calls, updated_todo_list)

        # Convert results to ToolMessages in original order
        results = []
        for tc in tool_calls:
            tid = tc.id
            result = exec_results.get(
                tid, f"Error: tool {tc.name} did not return a result"
            )
            if isinstance(result, tuple):
                # ParallelExecutor returns (tid, result, error) but we store just results
                pass

            results.append(
                ToolMessage(
                    tool_call_id=tc.id,
                    name=tc.name,
                    content=str(result),
                )
            )

        return {
            "messages": results,
            "sender": "tools",
            "todo_list": updated_todo_list,
        }

    async def _run_sequential(
        self,
        tool_calls: list[ToolCall],
        updated_todo_list: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Fallback: execute tool calls one at a time."""
        results = {}
        for tc in tool_calls:
            try:
                result = await self._execute_tool_call(
                    tc.name, tc.arguments, updated_todo_list
                )
                results[tc.id] = result
            except Exception as e:
                logger.error(f"Sequential fallback failed for {tc.name}: {e}")
                results[tc.id] = f"Error executing tool {tc.name}: {str(e)}"
        return results

    @staticmethod
    def _extract_depends_on(args: dict[str, Any]) -> list[str]:
        """Extract dependency list from tool arguments if annotated."""
        # The LLM can annotate dependencies via a special _depends_on key
        # which is stripped before passing to the actual tool.
        depends = args.pop("_depends_on", [])
        if isinstance(depends, list):
            return [str(d) for d in depends]
        return []
