"""
Pre-built fault scenarios for the Security layer.

Provides convenient factory functions for common security failure patterns
that can be used directly as mock side_effects.
"""

from collections.abc import Callable
from typing import Any

from src.core.common.exceptions import SecurityError
from src.core.security.manager import ValidationResponse, ValidationResult

from .base import build_side_effect

# ── Validation faults ──────────────────────────────────────────────


def rate_limit_exceeded(
    reason: str = "Rate limit exceeded: 60 requests per minute",
) -> Callable:
    """Create a side_effect that always returns a rate-limited response."""
    return build_side_effect(
        SecurityError,
        fail_count=1,
        kwargs={
            "message": reason,
            "details": {"reason": "rate_limited"},
        },
    )


def permission_denied(
    tool_name: str = "test_tool",
    reason: str = "Tool not in allowed permissions list",
) -> Callable:
    """Create a side_effect that always returns a permission denied response."""
    return build_side_effect(
        SecurityError,
        fail_count=1,
        kwargs={
            "message": f"Permission denied for tool '{tool_name}': {reason}",
            "details": {"tool": tool_name, "reason": "permission_denied"},
        },
    )


def input_validation_failed(
    field: str = "input",
    reason: str = "Input exceeds maximum allowed size of 10000 characters",
) -> Callable:
    """Create a side_effect that always returns an input validation failure."""
    return build_side_effect(
        SecurityError,
        fail_count=1,
        kwargs={
            "message": f"Input validation failed for '{field}': {reason}",
            "details": {"field": field, "reason": "validation_failed"},
        },
    )


def requires_approval(
    tool_name: str = "dangerous_tool",
) -> Callable:
    """Create a side_effect that returns a 'requires approval' response."""
    return build_side_effect(
        SecurityError,
        fail_count=1,
        kwargs={
            "message": f"Tool '{tool_name}' requires user approval",
            "details": {"tool": tool_name, "reason": "requires_approval"},
        },
    )


def security_audit_failure(
    reason: str = "Audit log write failed",
) -> Callable:
    """Create a side_effect that simulates an audit log failure.

    Tests that the security manager handles audit infrastructure
    failures gracefully without blocking valid operations.
    """
    return build_side_effect(
        SecurityError,
        fail_count=1,
        kwargs={
            "message": f"Security audit failed: {reason}",
            "details": {"reason": "audit_failure"},
        },
    )


# ── Retry patterns ─────────────────────────────────────────────────


def security_then_succeed(
    fail_count: int = 2,
    success_value: Any = None,
) -> Callable:
    """Create a side_effect that fails N times then succeeds.

    Useful for testing that callers retry validation after transient
    security infrastructure failures (e.g., audit log temporarily down).

    Args:
        fail_count: Number of failures before success.
        success_value: Value returned after failures are exhausted.

    Returns:
        A callable suitable for ``mock.validate_tool_execution.side_effect``.
    """
    if success_value is None:
        success_value = ValidationResponse(
            result=ValidationResult.ALLOWED,
            policy_applied="test_policy",
            message="All checks passed",
        )
    return build_side_effect(
        SecurityError,
        fail_count=fail_count,
        kwargs={
            "message": "Security infrastructure temporarily unavailable",
            "details": {"reason": "infrastructure_failure"},
        },
        success_value=success_value,
    )


# ── Validation response factories ──────────────────────────────────


def allowed_response(
    policy_applied: str = "test_policy",
) -> ValidationResponse:
    """Create a successful validation response."""
    return ValidationResponse(
        result=ValidationResult.ALLOWED,
        policy_applied=policy_applied,
        message="All checks passed",
    )


def denied_response(
    message: str = "Blocked by security policy",
    policy_applied: str = "test_policy",
) -> ValidationResponse:
    """Create a denied validation response."""
    return ValidationResponse(
        result=ValidationResult.DENIED,
        policy_applied=policy_applied,
        message=message,
    )


def rate_limited_response(
    message: str = "Rate limit exceeded",
    policy_applied: str = "test_policy",
) -> ValidationResponse:
    """Create a rate-limited validation response."""
    return ValidationResponse(
        result=ValidationResult.RATE_LIMITED,
        policy_applied=policy_applied,
        message=message,
    )


def requires_approval_response(
    message: str = "Requires user approval",
    policy_applied: str = "test_policy",
) -> ValidationResponse:
    """Create a requires-approval validation response."""
    return ValidationResponse(
        result=ValidationResult.REQUIRES_APPROVAL,
        policy_applied=policy_applied,
        message=message,
    )
