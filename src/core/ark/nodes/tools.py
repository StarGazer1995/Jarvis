import asyncio
import logging
import time
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from ...execution.engine import ParallelExecutor, ToolCall
from ...mcp.client import ARKMCPClient
from ...observability import MetricsRegistry
from ...security.manager import (
    ARKSecurityManager,
    ValidationResult,
)
from ...security.manager import (
    ValidationResponse as _ValidationResponse,
)
from ..state import JarvisState
from ..tasks import execute_manage_tasks

logger = logging.getLogger("ark.nodes.tools")

# ── Normalised security result templates ────────────────────────

_DENIED_MESSAGE = "Security: tool '{tool_name}' execution denied by policy '{policy}'."

_REQUIRES_APPROVAL_MESSAGE = (
    "Security: tool '{tool_name}' requires approval before execution. "
    "Please confirm to proceed."
)

_RATE_LIMITED_MESSAGE = (
    "Security: tool '{tool_name}' is rate limited. "
    "Retry after {retry_after:.0f} seconds."
)

_VALIDATION_ERROR_MESSAGE = "Security: tool '{tool_name}' validation error: {error}"


class ToolsNode:
    """
    Node responsible for executing tool calls (both internal and MCP).

    Supports parallel execution: independent tool calls within a single
    LLM response are executed concurrently via ``asyncio.gather``.
    Tool calls can declare dependencies via a ``depends_on`` field
    in their arguments (parsed from extended tool_call metadata).

    Security: every tool call is validated through a ``SecurityManager``
    before execution. Denied, approval-required, and rate-limited calls
    return a normalised tool result instead of executing.
    """

    def __init__(
        self,
        mcp_client: ARKMCPClient,
        max_concurrency: int = 10,
        security_manager: ARKSecurityManager | None = None,
    ):
        """
        Args:
            mcp_client: The MCP client for executing remote tools.
            max_concurrency: Maximum number of parallel tool executions.
            security_manager: Optional security manager for tool validation.
                If ``None``, security validation is skipped (legacy mode).
        """
        self.mcp_client = mcp_client
        self.local_tools: dict[str, Any] = {}
        self.max_concurrency = max_concurrency
        self.security_manager = security_manager
        self._session_id: str = "unknown"
        self._user_id: str = "anonymous"

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
        """
        Execute a single tool call (no security gate – gate already applied).

        Args:
            name: Tool name.
            args: Tool arguments.
            updated_todo_list: Mutable todo list for ``manage_tasks``.

        Returns:
            Tool execution result.
        """
        # ── Actual execution ─────────────────────────────────────
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

    async def _validate_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        execution_id: str,
    ) -> Any:
        """
        Build and run security validation for a single tool call.

        Args:
            tool_name: Name of the tool.
            arguments: Tool arguments.
            execution_id: Unique execution ID for this turn.

        Returns:
            ``ValidationResponse`` from the security manager, or ``None``
            if no security manager is configured.
        """
        if self.security_manager is None:
            return None

        request = ARKSecurityManager.build_validation_request(
            user_id=self._user_id,
            session_id=self._session_id,
            execution_id=execution_id,
            tool_name=tool_name,
            arguments=arguments,
        )

        # Choose policy based on tool name
        policy_name = self._policy_for_tool(tool_name)
        return await self.security_manager.validate_tool_execution(
            request, policy_name=policy_name
        )

    @staticmethod
    def _policy_for_tool(tool_name: str) -> str:
        """
        Map a tool name to a security policy.

        Override this method or provide a custom mapper to change
        the policy assignment strategy.
        """
        # Tools with filesystem or system access → high security
        _HIGH_RISK_TOOLS = {
            "execute_command",
            "run_code",
            "write_file",
            "edit_file",
            "create_file",
            "database_query",
            "network_request",
        }
        # Tools that modify state → medium security
        _MEDIUM_RISK_TOOLS = {
            "manage_tasks",
            "read_file",
            "write_file",
            "create_file",
            "edit_file",
            "list_dir",
            "github_api",
        }

        if tool_name in _HIGH_RISK_TOOLS:
            return "high"
        if tool_name in _MEDIUM_RISK_TOOLS:
            return "medium"
        return "low"

    async def __call__(
        self, state: JarvisState, config: RunnableConfig
    ) -> dict[str, Any]:
        """
        Execute tools requested in the last message.

        Each tool call is validated through the security manager before
        execution. Denied, approval-required, and rate-limited calls
        return a normalised tool result instead of executing.

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

        # Generate a single execution_id for this turn
        execution_id = f"exec_{time.time_ns()}"

        tool_calls: list[ToolCall] = []
        # Pre-compute security responses for every tool call
        security_responses: dict[str, Any] = {}
        # Pre-compute security result messages for denied tools
        security_messages: dict[str, str] = {}

        for tc in tool_calls_data:
            name = tc["name"]
            args = dict(tc.get("args", {}))
            tool_call_id = tc.get("id", f"{name}_{len(tool_calls)}")

            # Extract dependency info if present
            depends_on = self._extract_depends_on(args)

            tool_calls.append(
                ToolCall(
                    name=name,
                    arguments=args,
                    id=tool_call_id,
                    depends_on=depends_on,
                )
            )

            # ── Security validation ──────────────────────────────────
            try:
                sec_resp = await self._validate_tool_call(name, args, execution_id)
                security_responses[tool_call_id] = sec_resp

                if sec_resp is not None and sec_resp.result != ValidationResult.ALLOWED:
                    if sec_resp.result == ValidationResult.DENIED:
                        security_messages[tool_call_id] = _DENIED_MESSAGE.format(
                            tool_name=name, policy=sec_resp.policy_applied
                        )
                    elif sec_resp.result == ValidationResult.REQUIRES_APPROVAL:
                        security_messages[tool_call_id] = (
                            _REQUIRES_APPROVAL_MESSAGE.format(tool_name=name)
                        )
                    elif sec_resp.result == ValidationResult.RATE_LIMITED:
                        security_messages[tool_call_id] = _RATE_LIMITED_MESSAGE.format(
                            tool_name=name,
                            retry_after=sec_resp.retry_after or 60,
                        )
                    else:
                        security_messages[tool_call_id] = (
                            _VALIDATION_ERROR_MESSAGE.format(
                                tool_name=name, error="Unknown validation result"
                            )
                        )
            except Exception as e:
                logger.error(f"Security validation failed for {name}: {e}")
                security_responses[tool_call_id] = _ValidationResponse(
                    result=ValidationResult.DENIED,
                    policy_applied="default",
                    message=f"Validation error: {e}",
                )
                security_messages[tool_call_id] = _VALIDATION_ERROR_MESSAGE.format(
                    tool_name=name, error=str(e)
                )

        # ── Separate allowed and denied tool calls ────────────────
        allowed_calls: list[ToolCall] = []
        for tc in tool_calls:
            sec_resp = security_responses.get(tc.id)
            if sec_resp is None or sec_resp.result == ValidationResult.ALLOWED:
                allowed_calls.append(tc)

        # Execute only allowed tool calls using the ParallelExecutor
        exec_results: dict[str, Any] = {}
        if allowed_calls:
            executor = ParallelExecutor(
                execute_fn=lambda n, a: self._execute_tool_call(
                    n, a, updated_todo_list
                ),
                max_concurrency=self.max_concurrency,
            )
            try:
                exec_results = await executor.run(allowed_calls)
            except Exception as e:
                logger.error(f"Parallel execution failed: {e}")
                exec_results = await self._run_sequential(
                    allowed_calls, updated_todo_list
                )

        # Merge security messages with actual execution results
        all_results: dict[str, Any] = {}
        all_results.update(exec_results)
        all_results.update(security_messages)

        # ── Convert results to ToolMessages & record metrics ──────
        results = []
        for tc in tool_calls:
            tid = tc.id
            result = all_results.get(
                tid, f"Error: tool {tc.name} did not return a result"
            )
            result_str = str(result)

            # Determine execution status for metrics
            sec_resp = security_responses.get(tc.id)
            if tid in security_messages:
                metric_status = sec_resp.result.value if sec_resp else "denied"
            elif result_str.startswith("Error:"):
                metric_status = "error"
            else:
                metric_status = "success"

            # ── Observability: record tool metrics ────────────────
            try:
                server = "local" if tc.name in self.local_tools else "mcp"
                MetricsRegistry.record_tool_call(
                    tool_name=tc.name,
                    server=server,
                    status=metric_status,
                    latency=0.0,
                )
                MetricsRegistry.graph_iterations_total.labels(node="tools").inc()
            except Exception:
                logger.debug("Failed to record tool metrics", exc_info=True)

            # ── Structured logging ────────────────────────────────
            logger.info(
                "Tool execution completed",
                extra={
                    "component": "tools_node",
                    "event": "tool_execution",
                    "tool_name": tc.name,
                    "status": metric_status,
                    "session_id": self._session_id,
                    "policy": (sec_resp.policy_applied if sec_resp else "none"),
                },
            )

            results.append(
                ToolMessage(
                    tool_call_id=tc.id,
                    name=tc.name,
                    content=result_str,
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
        """Fallback: execute tool calls one at a time (security already checked)."""
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
