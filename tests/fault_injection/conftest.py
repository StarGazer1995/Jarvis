"""
Pytest fixtures for fault injection testing.

Provides fixtures that make it easy to set up fault scenarios
across all layers of the ARK engine.
"""

from unittest.mock import AsyncMock

import pytest

from src.core.common.exceptions import MCPError
from src.core.llm.types import LLMResponse
from src.core.security.manager import ValidationResponse, ValidationResult

from .llm_faults import (
    llm_rate_limit_then_succeed,
    llm_timeout_then_succeed,
    mixed_llm_faults,
)
from .mcp_faults import mcp_cascading_failure, mcp_then_succeed
from .security_faults import security_then_succeed

# ── LLM fault injection fixtures ───────────────────────────────────


@pytest.fixture
def mock_llm_with_timeout_retry():
    """Provide a mock LLM client that times out N times then succeeds.

    The returned mock has ``generate_response`` and ``stream_response``
    configured with timeout → success pattern. Use this to test that
    the retry handler correctly retries on timeout.

    The ``fail_count`` attribute can be adjusted before use.

    Returns:
        A configured ``AsyncMock`` with ``fail_count`` attribute.
    """
    mock = AsyncMock()
    mock.fail_count = 3

    def _apply_side_effect():
        mock.generate_response.side_effect = llm_timeout_then_succeed(
            fail_count=mock.fail_count,
            success_value=LLMResponse(content="Recovered from timeout", model="test"),
        )

    mock._apply_side_effect = _apply_side_effect
    _apply_side_effect()
    return mock


@pytest.fixture
def mock_llm_with_rate_limit_retry():
    """Provide a mock LLM client that rate-limits N times then succeeds."""
    mock = AsyncMock()
    mock.fail_count = 2

    def _apply_side_effect():
        mock.generate_response.side_effect = llm_rate_limit_then_succeed(
            fail_count=mock.fail_count,
            retry_after=1,
            success_value=LLMResponse(
                content="Recovered from rate limit", model="test"
            ),
        )

    mock._apply_side_effect = _apply_side_effect
    _apply_side_effect()
    return mock


@pytest.fixture
def mock_llm_with_mixed_faults():
    """Provide a mock LLM client with a realistic failure cascade.

    Sequence: timeout (×2) → rate limit (×1) → success.
    """
    mock = AsyncMock()
    mock.generate_response.side_effect = mixed_llm_faults(
        success_value=LLMResponse(content="Recovered from cascade", model="test"),
    )
    return mock


# ── MCP fault injection fixtures ───────────────────────────────────


@pytest.fixture
def mock_mcp_with_retry():
    """Provide a mock MCP client that fails N times then succeeds on execute_tool.

    The returned mock has ``execute_tool`` configured with a fault pattern.
    Adjust ``fail_count`` before use.

    Returns:
        A configured ``AsyncMock`` with ``fail_count`` attribute.
    """
    mock = AsyncMock()
    mock.fail_count = 2

    def _apply_side_effect():
        mock.execute_tool.side_effect = mcp_then_succeed(
            fail_count=mock.fail_count,
            success_value={"status": "success", "result": "Done"},
        )

    mock._apply_side_effect = _apply_side_effect
    _apply_side_effect()
    return mock


@pytest.fixture
def mock_mcp_with_cascade():
    """Provide a mock MCP client with cascading failure pattern.

    Sequence: MCP connection error → ToolError (×2) → success.
    """
    mock = AsyncMock()
    mock.execute_tool.side_effect = mcp_cascading_failure(
        success_value={"status": "success", "result": "Recovered"},
    )
    return mock


@pytest.fixture
def mock_mcp_with_disconnect():
    """Provide a mock MCP client where the server disconnects mid-operation.

    ``execute_tool`` raises MCPError (disconnect) on every call.
    """
    mock = AsyncMock()
    mock.execute_tool.side_effect = MCPError(
        message="Server disconnected",
        details={"reason": "unexpected_disconnect"},
    )
    mock.get_server_status.return_value = {"connected": False, "tools_count": 0}
    return mock


# ── Security fault injection fixtures ──────────────────────────────


@pytest.fixture
def mock_security_with_validation_retry():
    """Provide a mock security manager that fails N times then succeeds.

    The returned mock has ``validate_tool_execution`` configured with
    a fault pattern. Adjust ``fail_count`` before use.

    Returns:
        A configured ``AsyncMock`` with ``fail_count`` attribute.
    """
    mock = AsyncMock()
    mock.fail_count = 2

    def _apply_side_effect():
        mock.validate_tool_execution.side_effect = security_then_succeed(
            fail_count=mock.fail_count,
            success_value=ValidationResponse(
                result=ValidationResult.ALLOWED,
                reason="All checks passed",
            ),
        )

    mock._apply_side_effect = _apply_side_effect
    _apply_side_effect()
    return mock


@pytest.fixture
def mock_security_always_deny():
    """Provide a mock security manager that always denies requests."""
    mock = AsyncMock()
    mock.validate_tool_execution.return_value = ValidationResponse(
        result=ValidationResult.DENIED,
        reason="Policy blocked",
    )
    return mock


@pytest.fixture
def mock_security_always_rate_limit():
    """Provide a mock security manager that always rate-limits."""
    mock = AsyncMock()
    mock.validate_tool_execution.return_value = ValidationResponse(
        result=ValidationResult.RATE_LIMITED,
        reason="Rate limit exceeded",
    )
    return mock


@pytest.fixture
def mock_security_requires_approval():
    """Provide a mock security manager that always requires approval."""
    mock = AsyncMock()
    mock.validate_tool_execution.return_value = ValidationResponse(
        result=ValidationResult.REQUIRES_APPROVAL,
        reason="Tool requires user approval",
    )
    return mock


# ── Combined scenario fixtures ─────────────────────────────────────


@pytest.fixture
def mock_full_pipeline_with_faults():
    """Provide a tuple of (mock_llm, mock_mcp, mock_security) all pre-configured
    with fault patterns for end-to-end resilience testing."""
    mock_llm = AsyncMock()
    mock_llm.generate_response.side_effect = llm_timeout_then_succeed(
        fail_count=2,
        success_value=LLMResponse(content="Recovered", model="test"),
    )

    mock_mcp = AsyncMock()
    mock_mcp.execute_tool.side_effect = mcp_then_succeed(
        fail_count=1,
        success_value={"status": "success", "result": "Done"},
    )

    mock_security = AsyncMock()
    mock_security.validate_tool_execution.return_value = ValidationResponse(
        result=ValidationResult.ALLOWED,
        reason="All checks passed",
    )

    return mock_llm, mock_mcp, mock_security
