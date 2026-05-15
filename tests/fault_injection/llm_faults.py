"""
Pre-built fault scenarios for the LLM layer.

Provides convenient factory functions for common LLM failure patterns
that can be used directly as mock side_effects.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from src.core.llm.utils.error_handler import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from src.core.llm.types import LLMResponse

from .base import FaultConfig, FaultInjector, FaultPattern, build_side_effect


# ── Single fault patterns ──────────────────────────────────────────


def llm_timeout(
    message: str = "Request timed out",
    timeout_duration: float = 30.0,
) -> Callable:
    """Create a side_effect that always raises LLMTimeoutError.

    Useful for testing that timeouts are NOT retried when outside
    the retry window, or for testing the error handler directly.
    """
    return build_side_effect(
        LLMTimeoutError,
        fail_count=1,
        kwargs={"message": message, "timeout_duration": timeout_duration},
    )


def llm_rate_limit(
    message: str = "Rate limit exceeded",
    retry_after: int = 60,
) -> Callable:
    """Create a side_effect that always raises LLMRateLimitError."""
    return build_side_effect(
        LLMRateLimitError,
        fail_count=1,
        kwargs={"message": message, "retry_after": retry_after},
    )


def llm_api_error(
    message: str = "API error",
    status_code: int = 500,
) -> Callable:
    """Create a side_effect that always raises LLMAPIError."""
    return build_side_effect(
        LLMAPIError,
        fail_count=1,
        kwargs={"message": message, "status_code": status_code},
    )


def llm_auth_error(message: str = "Authentication failed") -> Callable:
    """Create a side_effect that always raises LLMAuthenticationError."""
    return build_side_effect(
        LLMAuthenticationError,
        fail_count=1,
        kwargs={"message": message},
    )


def llm_config_error(
    message: str = "Invalid configuration",
    config_field: str = "api_key",
) -> Callable:
    """Create a side_effect that always raises LLMConfigurationError."""
    return build_side_effect(
        LLMConfigurationError,
        fail_count=1,
        kwargs={"message": message, "config_field": config_field},
    )


# ── Retry patterns (fail N times then succeed) ─────────────────────


def llm_timeout_then_succeed(
    fail_count: int = 3,
    success_value: Any = None,
) -> Callable:
    """Create a side_effect that times out N times then succeeds.

    This is the most common pattern for testing retry logic: the LLM
    call times out a few times (triggering retry), then finally returns
    a valid response.

    Args:
        fail_count: Number of timeouts before success.
        success_value: Value returned after all timeouts are exhausted.
            Defaults to a simple LLMResponse.

    Returns:
        A callable suitable for ``mock.generate_response.side_effect``.
    """
    if success_value is None:
        success_value = LLMResponse(
            content="Retry succeeded",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
    return build_side_effect(
        LLMTimeoutError,
        fail_count=fail_count,
        kwargs={"message": "Request timed out", "timeout_duration": 30.0},
        success_value=success_value,
    )


def llm_rate_limit_then_succeed(
    fail_count: int = 2,
    retry_after: int = 1,
    success_value: Any = None,
) -> Callable:
    """Create a side_effect that hits rate limits N times then succeeds.

    Args:
        fail_count: Number of rate limit errors before success.
        retry_after: Seconds to wait before retry (used by RateLimitHandler).
        success_value: Value returned after rate limits are exhausted.

    Returns:
        A callable suitable for ``mock.generate_response.side_effect``.
    """
    if success_value is None:
        success_value = LLMResponse(
            content="Rate limited then succeeded",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
    return build_side_effect(
        LLMRateLimitError,
        fail_count=fail_count,
        kwargs={"message": "Rate limit exceeded", "retry_after": retry_after},
        success_value=success_value,
    )


def llm_api_error_then_succeed(
    fail_count: int = 3,
    status_code: int = 502,
    success_value: Any = None,
) -> Callable:
    """Create a side_effect that returns API errors N times then succeeds."""
    if success_value is None:
        success_value = LLMResponse(
            content="API recovered",
            model="test-model",
        )
    return build_side_effect(
        LLMAPIError,
        fail_count=fail_count,
        kwargs={"message": "Bad gateway", "status_code": status_code},
        success_value=success_value,
    )


# ── Mixed / chained fault patterns ─────────────────────────────────


def mixed_llm_faults(
    success_value: Any = None,
) -> Callable:
    """Create a realistic failure cascade: timeout → rate limit → success.

    Simulates an overloaded LLM service that first times out, then
    starts rate-limiting retries, and finally recovers.
    """
    if success_value is None:
        success_value = LLMResponse(content="Recovered from cascading failures")

    pattern = FaultPattern("mixed_llm_faults")
    pattern.add_fault(
        LLMTimeoutError,
        fail_count=2,
        kwargs={"message": "Upstream timeout", "timeout_duration": 30.0},
    )
    pattern.add_fault(
        LLMRateLimitError,
        fail_count=1,
        kwargs={"message": "Now being rate limited", "retry_after": 5},
    )
    pattern.then_succeed(success_value)
    return pattern.build()
