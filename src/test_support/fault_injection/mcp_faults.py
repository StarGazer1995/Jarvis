"""
Pre-built fault scenarios for the MCP layer.

Provides convenient factory functions for common MCP failure patterns
that can be used directly as mock side_effects.
"""

from collections.abc import Callable
from typing import Any

from src.core.common.exceptions import MCPError, ToolError

from .base import FaultPattern, build_side_effect

# ── Connection-level faults ────────────────────────────────────────


def mcp_connection_failure(
    server_name: str = "test_server",
) -> Callable:
    """Create a side_effect that always fails to connect.

    Simulates the MCP server being unreachable (e.g., process crashed,
    port not listening).
    """
    return build_side_effect(
        MCPError,
        fail_count=1,
        kwargs={
            "message": f"Failed to connect to MCP server '{server_name}': connection refused",
            "details": {"server": server_name, "reason": "connection_refused"},
        },
    )


def mcp_server_disconnect(
    server_name: str = "test_server",
) -> Callable:
    """Create a side_effect that simulates mid-operation server disconnect.

    The server connects initially but drops mid-operation.
    """
    return build_side_effect(
        MCPError,
        fail_count=1,
        kwargs={
            "message": f"MCP server '{server_name}' disconnected during operation",
            "details": {"server": server_name, "reason": "unexpected_disconnect"},
        },
    )


# ── Tool-level faults ──────────────────────────────────────────────


def mcp_tool_not_found(tool_name: str = "unknown_tool") -> Callable:
    """Create a side_effect that reports a tool does not exist."""
    return build_side_effect(
        ToolError,
        fail_count=1,
        kwargs={
            "message": f"Tool '{tool_name}' not found on any connected MCP server",
            "tool_name": tool_name,
        },
    )


def mcp_tool_execution_error(
    tool_name: str = "test_tool",
    error_message: str = "Internal tool error",
) -> Callable:
    """Create a side_effect that simulates a tool execution failure.

    The tool exists but fails when called (e.g., invalid arguments,
    runtime error in the tool implementation).
    """
    return build_side_effect(
        ToolError,
        fail_count=1,
        kwargs={
            "message": f"Tool '{tool_name}' execution failed: {error_message}",
            "tool_name": tool_name,
            "details": {"error": error_message},
        },
    )


def mcp_tool_timeout(tool_name: str = "test_tool") -> Callable:
    """Create a side_effect that simulates a tool call timing out."""
    return build_side_effect(
        ToolError,
        fail_count=1,
        kwargs={
            "message": f"Tool '{tool_name}' call timed out",
            "tool_name": tool_name,
            "details": {"error": "timeout"},
        },
    )


def mcp_tool_invalid_response(tool_name: str = "test_tool") -> Callable:
    """Create a side_effect that simulates a malformed tool response.

    The tool executes but returns data that cannot be parsed.
    """
    return build_side_effect(
        ToolError,
        fail_count=1,
        kwargs={
            "message": f"Tool '{tool_name}' returned invalid response format",
            "tool_name": tool_name,
            "details": {"error": "invalid_response_format"},
        },
    )


# ── Tool discovery faults ──────────────────────────────────────────


def mcp_discovery_failure(server_name: str = "test_server") -> Callable:
    """Create a side_effect that simulates tool discovery failing.

    The server connects but fails to list its tools.
    """
    return build_side_effect(
        MCPError,
        fail_count=1,
        kwargs={
            "message": f"Failed to discover tools on server '{server_name}'",
            "details": {"server": server_name, "phase": "tool_discovery"},
        },
    )


# ── Retry patterns ─────────────────────────────────────────────────


def mcp_then_succeed(
    fail_count: int = 2,
    success_value: Any = None,
) -> Callable:
    """Create a side_effect that fails N times then succeeds.

    Useful for testing that the MCP client retries tool execution
    after transient failures.

    Args:
        fail_count: Number of failures before success.
        success_value: Value returned after failures are exhausted.

    Returns:
        A callable suitable for ``mock.execute_tool.side_effect``.
    """
    if success_value is None:
        success_value = {"status": "success", "result": "Tool completed after retry"}
    return build_side_effect(
        ToolError,
        fail_count=fail_count,
        kwargs={
            "message": "Transient tool execution failure",
            "tool_name": "test_tool",
        },
        success_value=success_value,
    )


def mcp_cascading_failure(
    success_value: Any = None,
) -> Callable:
    """Simulate a cascade: tool not found → tool error → execution error → success.

    This pattern tests the MCP client's ability to handle multiple
    different failure modes in sequence, such as might happen during
    server rollout or partial outages.
    """
    if success_value is None:
        success_value = {"status": "success", "result": "Recovered from cascade"}

    pattern = FaultPattern("mcp_cascading_failure")
    pattern.add_fault(
        MCPError,
        fail_count=1,
        kwargs={
            "message": "Server not ready",
            "details": {"reason": "initializing"},
        },
    )
    pattern.add_fault(
        ToolError,
        fail_count=2,
        kwargs={
            "message": "Tool temporarily unavailable",
            "tool_name": "test_tool",
            "details": {"reason": "degraded"},
        },
    )
    pattern.then_succeed(success_value)
    return pattern.build()
