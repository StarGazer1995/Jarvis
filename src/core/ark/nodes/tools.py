import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import ToolMessage, messages_from_dict, messages_to_dict
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
from ..runtime_store import ApprovalRuntimeStore
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


@dataclass
class PendingApproval:
    """
    Represents a single tool call waiting for user approval.

    Attributes:
        approval_id: Stable identifier used to approve or reject the request.
        tool_call_id: Original tool call ID emitted by the model.
        tool_name: Tool name awaiting approval.
        arguments: Tool arguments to execute once approved.
        session_id: Session associated with the pending request.
        user_id: User associated with the pending request.
        policy_name: Policy that required approval.
        execution_id: Execution identifier for the original validation turn.
        created_at: Unix timestamp when the pending request was created.
        validation_request: Original validation request object stored for auditing.
    """

    approval_id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    session_id: str
    user_id: str
    policy_name: str
    execution_id: str
    created_at: float
    validation_request: Any

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the pending approval record to a user-facing dictionary.

        Returns:
            Serializable dictionary containing approval metadata.
        """
        return {
            "approval_id": self.approval_id,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "policy_name": self.policy_name,
            "execution_id": self.execution_id,
            "created_at": self.created_at,
        }

    def to_persisted_dict(self) -> dict[str, Any]:
        """
        Convert the approval record to its durable JSON representation.

        Returns:
            Serializable dictionary used by the runtime store.
        """
        return self.to_dict()

    @classmethod
    def from_persisted_dict(cls, data: dict[str, Any]) -> "PendingApproval":
        """
        Restore a pending approval from durable storage.

        Args:
            data: Persisted approval payload.

        Returns:
            Restored pending approval instance.
        """
        validation_request = ARKSecurityManager.build_validation_request(
            user_id=str(data.get("user_id") or "anonymous"),
            session_id=str(data.get("session_id") or "unknown"),
            execution_id=str(data.get("execution_id") or f"exec_{time.time_ns()}"),
            tool_name=str(data["tool_name"]),
            arguments=dict(data.get("arguments", {})),
        )
        return cls(
            approval_id=str(data["approval_id"]),
            tool_call_id=str(data["tool_call_id"]),
            tool_name=str(data["tool_name"]),
            arguments=dict(data.get("arguments", {})),
            session_id=str(data.get("session_id") or "unknown"),
            user_id=str(data.get("user_id") or "anonymous"),
            policy_name=str(data.get("policy_name") or "unknown"),
            execution_id=str(
                data.get("execution_id") or validation_request.context.execution_id
            ),
            created_at=float(data.get("created_at", time.time())),
            validation_request=validation_request,
        )


@dataclass
class PendingContinuation:
    """
    Stores the minimum state required to resume graph execution after approval.

    Attributes:
        approval_id: Pending approval identifier associated with this continuation.
        tool_call_id: Tool call that should be replaced with the approved result.
        session_id: Session associated with the saved graph state.
        user_id: User associated with the saved graph state.
        state_snapshot: Post-tools graph state captured before reasoning continues.
    """

    approval_id: str
    tool_call_id: str
    session_id: str
    user_id: str
    state_snapshot: JarvisState

    def to_persisted_dict(self) -> dict[str, Any]:
        """
        Convert the continuation state to a durable JSON representation.

        Returns:
            Serializable continuation payload.
        """
        snapshot = {
            "messages": messages_to_dict(self.state_snapshot["messages"]),
            "user_input": self.state_snapshot.get("user_input", ""),
            "todo_list": [
                task.copy() for task in self.state_snapshot.get("todo_list", [])
            ],
            "available_tools": dict(self.state_snapshot.get("available_tools", {})),
            "scratchpad": dict(self.state_snapshot.get("scratchpad", {})),
            "sender": self.state_snapshot.get("sender", "tools"),
            "iteration_count": self.state_snapshot.get("iteration_count", 0),
            "termination_reason": self.state_snapshot.get("termination_reason"),
        }
        return {
            "approval_id": self.approval_id,
            "tool_call_id": self.tool_call_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "state_snapshot": snapshot,
        }

    @classmethod
    def from_persisted_dict(cls, data: dict[str, Any]) -> "PendingContinuation":
        """
        Restore a continuation state from durable storage.

        Args:
            data: Persisted continuation payload.

        Returns:
            Restored continuation instance.
        """
        snapshot = dict(data.get("state_snapshot", {}))
        restored_state: JarvisState = {
            "messages": messages_from_dict(snapshot.get("messages", [])),
            "user_input": str(snapshot.get("user_input", "")),
            "todo_list": [dict(task) for task in snapshot.get("todo_list", [])],
            "available_tools": dict(snapshot.get("available_tools", {})),
            "scratchpad": dict(snapshot.get("scratchpad", {})),
            "sender": str(snapshot.get("sender", "tools")),
            "iteration_count": int(snapshot.get("iteration_count", 0)),
            "termination_reason": snapshot.get("termination_reason"),
        }
        return cls(
            approval_id=str(data["approval_id"]),
            tool_call_id=str(data["tool_call_id"]),
            session_id=str(data.get("session_id") or "unknown"),
            user_id=str(data.get("user_id") or "anonymous"),
            state_snapshot=restored_state,
        )


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
        state_store_path: str | None = None,
    ):
        """
        Args:
            mcp_client: The MCP client for executing remote tools.
            max_concurrency: Maximum number of parallel tool executions.
            security_manager: Optional security manager for tool validation.
                If ``None``, security validation is skipped (legacy mode).
            state_store_path: Optional path used to persist approval runtime state.
        """
        self.mcp_client = mcp_client
        self.local_tools: dict[str, Any] = {}
        self.max_concurrency = max_concurrency
        self.security_manager = security_manager
        self.runtime_store = ApprovalRuntimeStore(state_store_path)
        self._session_id: str = "unknown"
        self._user_id: str = "anonymous"
        self._execution_latencies: dict[str, float] = {}
        self._pending_approvals: dict[str, PendingApproval] = {}
        self._pending_continuations: dict[str, PendingContinuation] = {}
        self._restore_persistent_state()

    def register_tool(self, name: str, func: Any):
        """Register a local tool function."""
        self.local_tools[name] = func

    def set_execution_context(
        self, session_id: str, user_id: str | None = None
    ) -> None:
        """
        Update execution-scoped identity used by security and logging.

        Args:
            session_id: Active conversation session identifier
            user_id: Optional user identifier for audit and rate limiting
        """
        self._session_id = session_id
        self._user_id = user_id or "anonymous"

    async def _invoke_local_tool(self, func: Any, args: Any) -> Any:
        if asyncio.iscoroutinefunction(func):
            return await (func(**args) if isinstance(args, dict) else func(args))
        return func(**args) if isinstance(args, dict) else func(args)

    async def _execute_tool_call(
        self,
        name: str,
        args: Any,
        updated_todo_list: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
    ) -> Any:
        """
        Execute a single tool call (no security gate – gate already applied).

        Args:
            name: Tool name.
            args: Tool arguments.
            updated_todo_list: Mutable todo list for ``manage_tasks``.
            tool_call_id: Optional tool call identifier used for latency tracking.

        Returns:
            Tool execution result.
        """
        start_time = time.perf_counter()
        try:
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
        finally:
            if tool_call_id is not None:
                self._execution_latencies[tool_call_id] = (
                    time.perf_counter() - start_time
                )

    async def _validate_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        execution_id: str,
    ) -> tuple[Any, Any]:
        """
        Build and run security validation for a single tool call.

        Args:
            tool_name: Name of the tool.
            arguments: Tool arguments.
            execution_id: Unique execution ID for this turn.

        Returns:
            Tuple of ``(ValidationResponse, ValidationRequest)``, or ``(None, None)``
            if no security manager is configured.
        """
        if self.security_manager is None:
            return None, None

        request = ARKSecurityManager.build_validation_request(
            user_id=self._user_id,
            session_id=self._session_id,
            execution_id=execution_id,
            tool_name=tool_name,
            arguments=arguments,
        )

        # Choose policy based on tool name
        policy_name = self._policy_for_tool(tool_name)
        response = await self.security_manager.validate_tool_execution(
            request, policy_name=policy_name
        )
        return response, request

    def _store_pending_approval(
        self,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        execution_id: str,
        policy_name: str,
        validation_request: Any,
    ) -> PendingApproval:
        """
        Persist a pending approval record for later user action.

        Args:
            tool_call_id: Original tool call ID from the model
            tool_name: Tool name awaiting approval
            arguments: Tool arguments to reuse after approval
            execution_id: Execution identifier for the current turn
            policy_name: Security policy that requested approval
            validation_request: Validation request stored by the security manager

        Returns:
            The newly stored pending approval object.
        """
        approval = PendingApproval(
            approval_id=f"approval_{uuid.uuid4().hex[:12]}",
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=dict(arguments),
            session_id=self._session_id,
            user_id=self._user_id,
            policy_name=policy_name,
            execution_id=execution_id,
            created_at=time.time(),
            validation_request=validation_request,
        )
        self._pending_approvals[approval.approval_id] = approval
        self._sync_persistent_state()
        return approval

    def _store_pending_continuation(
        self,
        approval_ids: list[str],
        state: JarvisState,
        results: list[ToolMessage],
        updated_todo_list: list[dict[str, Any]],
    ) -> None:
        """
        Capture post-tools graph state for each approval created in the current turn.

        Args:
            approval_ids: Approval identifiers created during the current node call
            state: Incoming graph state before tool results were appended
            results: Tool messages produced for this turn
            updated_todo_list: Updated todo list after tool execution
        """
        if not approval_ids:
            return

        state_snapshot: JarvisState = {
            "messages": list(state["messages"]) + list(results),
            "user_input": state.get("user_input", ""),
            "todo_list": [task.copy() for task in updated_todo_list],
            "available_tools": dict(state.get("available_tools", {})),
            "scratchpad": dict(state.get("scratchpad", {})),
            "sender": "tools",
            "iteration_count": state.get("iteration_count", 0),
            "termination_reason": None,
        }
        for approval_id in approval_ids:
            approval = self._pending_approvals.get(approval_id)
            if approval is None:
                continue
            self._pending_continuations[approval_id] = PendingContinuation(
                approval_id=approval_id,
                tool_call_id=approval.tool_call_id,
                session_id=approval.session_id,
                user_id=approval.user_id,
                state_snapshot=state_snapshot,
            )
        self._sync_persistent_state()

    def _build_resume_state(
        self,
        continuation: PendingContinuation,
        tool_name: str,
        result: Any,
    ) -> JarvisState:
        """
        Replace the approval placeholder message with the approved tool result.

        Args:
            continuation: Stored continuation state for the pending approval
            tool_name: Tool name used to rebuild the approved ToolMessage
            result: Actual tool execution result

        Returns:
            Updated graph state ready to resume from the master node.
        """
        updated_messages = []
        for message in continuation.state_snapshot["messages"]:
            if (
                isinstance(message, ToolMessage)
                and message.tool_call_id == continuation.tool_call_id
            ):
                updated_messages.append(
                    ToolMessage(
                        tool_call_id=message.tool_call_id,
                        name=tool_name,
                        content=str(result),
                    )
                )
                continue
            updated_messages.append(message)

        return {
            **continuation.state_snapshot,
            "messages": updated_messages,
            "sender": "tools",
            "termination_reason": None,
        }

    def list_pending_approvals(
        self,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List pending approvals, optionally filtered by session.

        Args:
            session_id: Optional session identifier filter

        Returns:
            Serializable list of pending approvals.
        """
        approvals = self._pending_approvals.values()
        if session_id is not None:
            approvals = [
                approval for approval in approvals if approval.session_id == session_id
            ]
        return [approval.to_dict() for approval in approvals]

    def reject_pending_approval(self, approval_id: str) -> dict[str, Any]:
        """
        Reject and remove a pending approval request.

        Args:
            approval_id: Pending approval identifier

        Returns:
            Result payload describing the rejected request.

        Raises:
            ValueError: If the approval ID does not exist.
        """
        approval = self._pending_approvals.pop(approval_id, None)
        if approval is None:
            raise ValueError(f"Pending approval '{approval_id}' not found")
        self._pending_continuations.pop(approval_id, None)

        if (
            self.security_manager is not None
            and approval.validation_request in self.security_manager.approval_queue
        ):
            self.security_manager.approval_queue.remove(approval.validation_request)
        self._sync_persistent_state()

        return {
            "approval_id": approval_id,
            "status": "rejected",
            "tool_name": approval.tool_name,
            "session_id": approval.session_id,
        }

    async def approve_pending_approval(self, approval_id: str) -> dict[str, Any]:
        """
        Approve and execute a pending tool call under its original context.

        Args:
            approval_id: Pending approval identifier

        Returns:
            Result payload describing the executed tool call.

        Raises:
            ValueError: If the approval ID does not exist.
        """
        approval = self._pending_approvals.pop(approval_id, None)
        if approval is None:
            raise ValueError(f"Pending approval '{approval_id}' not found")
        continuation = self._pending_continuations.pop(approval_id, None)

        if (
            self.security_manager is not None
            and approval.validation_request in self.security_manager.approval_queue
        ):
            self.security_manager.approval_queue.remove(approval.validation_request)
        self._sync_persistent_state()

        previous_session_id = self._session_id
        previous_user_id = self._user_id
        try:
            self.set_execution_context(
                session_id=approval.session_id,
                user_id=approval.user_id,
            )
            result = await self._execute_tool_call(
                approval.tool_name,
                dict(approval.arguments),
                tool_call_id=approval.tool_call_id,
            )
        finally:
            self.set_execution_context(previous_session_id, previous_user_id)

        return {
            "approval_id": approval_id,
            "status": "approved",
            "tool_name": approval.tool_name,
            "tool_call_id": approval.tool_call_id,
            "session_id": approval.session_id,
            "user_id": approval.user_id,
            "result": result,
            "resume_state": (
                self._build_resume_state(continuation, approval.tool_name, result)
                if continuation is not None
                else None
            ),
        }

    def _restore_persistent_state(self) -> None:
        """
        Load persisted approvals and continuations into the current runtime.
        """
        payload = self.runtime_store.load()
        restored_approvals: dict[str, PendingApproval] = {}
        for item in payload.get("pending_approvals", []):
            try:
                approval = PendingApproval.from_persisted_dict(item)
            except Exception as exc:
                logger.warning("Failed to restore pending approval: %s", exc)
                continue
            restored_approvals[approval.approval_id] = approval
            if (
                self.security_manager is not None
                and approval.validation_request
                not in self.security_manager.approval_queue
            ):
                self.security_manager.approval_queue.append(approval.validation_request)

        restored_continuations: dict[str, PendingContinuation] = {}
        for item in payload.get("pending_continuations", []):
            try:
                continuation = PendingContinuation.from_persisted_dict(item)
            except Exception as exc:
                logger.warning("Failed to restore pending continuation: %s", exc)
                continue
            restored_continuations[continuation.approval_id] = continuation

        self._pending_approvals = restored_approvals
        self._pending_continuations = restored_continuations

    def _sync_persistent_state(self) -> None:
        """
        Persist the current approval runtime state to disk.
        """
        self.runtime_store.save(
            {
                "pending_approvals": [
                    approval.to_persisted_dict()
                    for approval in self._pending_approvals.values()
                ],
                "pending_continuations": [
                    continuation.to_persisted_dict()
                    for continuation in self._pending_continuations.values()
                ],
            }
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
        self._execution_latencies = {}

        # Generate a single execution_id for this turn
        execution_id = f"exec_{time.time_ns()}"

        tool_calls: list[ToolCall] = []
        # Pre-compute security responses for every tool call
        security_responses: dict[str, Any] = {}
        security_requests: dict[str, Any] = {}
        # Pre-compute security result messages for denied tools
        security_messages: dict[str, str] = {}
        created_approval_ids: list[str] = []

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
                sec_resp, validation_request = await self._validate_tool_call(
                    name, args, execution_id
                )
                security_responses[tool_call_id] = sec_resp
                security_requests[tool_call_id] = validation_request

                if sec_resp is not None and sec_resp.result != ValidationResult.ALLOWED:
                    if sec_resp.result == ValidationResult.DENIED:
                        security_messages[tool_call_id] = _DENIED_MESSAGE.format(
                            tool_name=name, policy=sec_resp.policy_applied
                        )
                    elif sec_resp.result == ValidationResult.REQUIRES_APPROVAL:
                        pending_approval = self._store_pending_approval(
                            tool_call_id=tool_call_id,
                            tool_name=name,
                            arguments=args,
                            execution_id=execution_id,
                            policy_name=sec_resp.policy_applied,
                            validation_request=validation_request,
                        )
                        created_approval_ids.append(pending_approval.approval_id)
                        security_messages[tool_call_id] = (
                            _REQUIRES_APPROVAL_MESSAGE.format(tool_name=name)
                            + f" Approval ID: {pending_approval.approval_id}"
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
                execute_fn=lambda n, a, tid=None: self._execute_tool_call(
                    n, a, updated_todo_list, tid
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
                    latency=self._execution_latencies.get(tid, 0.0),
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

        self._store_pending_continuation(
            approval_ids=created_approval_ids,
            state=state,
            results=results,
            updated_todo_list=updated_todo_list,
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
                    tc.name, tc.arguments, updated_todo_list, tc.id
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
