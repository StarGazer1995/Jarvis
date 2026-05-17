"""
Jarvis Fault Injection Framework

A systematic fault injection testing framework for validating resilience
and error handling across all layers of the ARK engine.

Usage:
    from src.test_support.fault_injection import (
        FaultInjector, FaultConfig, FaultPattern,
        llm_faults, mcp_faults, security_faults,
    )

    # Inject a fault that fails 3 times then succeeds
    mock_client.generate_response.side_effect = FaultInjector(
        FaultConfig(
            exception=LLMAPIError,
            error_kwargs={"message": "API failure"},
            fail_count=3,
            success_value=LLMResponse(content="ok"),
        )
    ).build()

    # Use pre-built patterns
    from src.test_support.fault_injection.llm_faults import llm_timeout_then_succeed

    mock_client.generate_response.side_effect = llm_timeout_then_succeed(
        fail_count=2, success_value=LLMResponse(content="ok")
    )
"""

from .base import FaultConfig, FaultInjector, FaultPattern

__all__ = [
    "FaultInjector",
    "FaultConfig",
    "FaultPattern",
]
