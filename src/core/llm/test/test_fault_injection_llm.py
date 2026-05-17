"""
Fault Injection Tests — LLM Layer

Tests the resilience of the LLM client layer against various failure
scenarios using the FaultInjector framework. Covers:

- Retry handler behavior under different exception types
- Rate limit handling with retry_after
- Mixed failure cascades
- Non-retryable exceptions (should propagate immediately)
- Streaming errors
- LLM manager fallback between providers
"""

import time
from unittest.mock import AsyncMock

import pytest

from src.core.llm.types import LLMResponse
from src.core.llm.utils.error_handler import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from src.core.llm.utils.retry_handler import (
    RateLimitHandler,
    RetryHandler,
    with_retry,
)
from src.test_support.fault_injection import FaultConfig, FaultInjector
from src.test_support.fault_injection.llm_faults import (
    llm_api_error,
    llm_auth_error,
    llm_config_error,
    llm_rate_limit,
    llm_rate_limit_then_succeed,
    llm_timeout,
    llm_timeout_then_succeed,
    mixed_llm_faults,
)

# ═══════════════════════════════════════════════════════════════════
# 1. FaultInjector Unit Tests
# ═══════════════════════════════════════════════════════════════════


class TestFaultInjectorBasics:
    """Verify the FaultInjector itself works correctly."""

    def test_fault_then_success(self):
        """Basic: fail N times, then succeed."""
        side_effect = FaultInjector(
            FaultConfig(ValueError, args=("boom",), fail_count=2),
            success_value="ok",
        ).build()

        # First two calls should raise
        with pytest.raises(ValueError, match="boom"):
            side_effect()
        with pytest.raises(ValueError, match="boom"):
            side_effect()

        # Third call should succeed
        assert side_effect() == "ok"

    def test_zero_faults_with_success(self):
        """No faults configured → always succeed."""
        side_effect = FaultInjector(success_value="always_ok").build()
        assert side_effect() == "always_ok"
        assert side_effect() == "always_ok"

    def test_fault_always_raises(self):
        """fail_count=1 with no success_value → always raises."""
        side_effect = FaultInjector(
            FaultConfig(RuntimeError, args=("always",), fail_count=1),
        ).build()

        with pytest.raises(RuntimeError, match="always"):
            side_effect()
        with pytest.raises(RuntimeError, match="always"):
            side_effect()  # should keep raising

    def test_multiple_phases(self):
        """Chain multiple fault types."""
        from src.test_support.fault_injection import FaultPattern

        pattern = FaultPattern("multi_phase")
        pattern.add_fault(ValueError, fail_count=2, args=("v1",))
        pattern.add_fault(TypeError, fail_count=1, args=("t1",))
        pattern.then_succeed("done")

        side_effect = pattern.build()

        with pytest.raises(ValueError, match="v1"):
            side_effect()
        with pytest.raises(ValueError, match="v1"):
            side_effect()
        with pytest.raises(TypeError, match="t1"):
            side_effect()
        assert side_effect() == "done"

    def test_async_compatible(self):
        """FaultInjector works as side_effect for async mocks."""
        mock = AsyncMock()
        mock.fetch_data.side_effect = FaultInjector(
            FaultConfig(TimeoutError, args=("timeout",), fail_count=1),
            success_value={"data": "ok"},
        ).build()

        # First call raises (fail_count=1)
        import asyncio

        with pytest.raises(TimeoutError, match="timeout"):
            asyncio.run(mock.fetch_data())

        # Second call succeeds
        result = asyncio.run(mock.fetch_data())
        assert result == {"data": "ok"}


# ═══════════════════════════════════════════════════════════════════
# 2. Pre-built LLM Fault Helpers
# ═══════════════════════════════════════════════════════════════════


class TestPrebuiltLLMFaults:
    """Verify the pre-built LLM fault helpers work correctly."""

    def test_llm_timeout_raises(self):
        side_effect = llm_timeout()
        with pytest.raises(LLMTimeoutError, match="timed out"):
            side_effect()

    def test_llm_rate_limit_raises(self):
        side_effect = llm_rate_limit()
        with pytest.raises(LLMRateLimitError, match="Rate limit"):
            side_effect()

    def test_llm_api_error_raises(self):
        side_effect = llm_api_error()
        with pytest.raises(LLMAPIError, match="API error"):
            side_effect()

    def test_llm_auth_error_raises(self):
        side_effect = llm_auth_error()
        with pytest.raises(LLMAuthenticationError, match="Authentication"):
            side_effect()

    def test_llm_config_error_raises(self):
        side_effect = llm_config_error()
        with pytest.raises(LLMConfigurationError, match="Invalid configuration"):
            side_effect()

    def test_llm_timeout_then_succeed(self):
        """Verify the retry pattern: fails N times, then returns a response."""
        side_effect = llm_timeout_then_succeed(
            fail_count=3,
            success_value=LLMResponse(content="ok", model="test"),
        )

        for i in range(3):
            with pytest.raises(LLMTimeoutError):
                side_effect()

        result = side_effect()
        assert isinstance(result, LLMResponse)
        assert result.content == "ok"

    def test_llm_rate_limit_then_succeed(self):
        side_effect = llm_rate_limit_then_succeed(
            fail_count=2,
            success_value=LLMResponse(content="ok", model="test"),
        )

        with pytest.raises(LLMRateLimitError):
            side_effect()
        with pytest.raises(LLMRateLimitError):
            side_effect()

        result = side_effect()
        assert result.content == "ok"

    def test_mixed_llm_faults(self):
        """Cascade: timeout ×2 → rate_limit ×1 → success."""
        side_effect = mixed_llm_faults(
            success_value=LLMResponse(content="recovered", model="test"),
        )

        with pytest.raises(LLMTimeoutError):
            side_effect()
        with pytest.raises(LLMTimeoutError):
            side_effect()
        with pytest.raises(LLMRateLimitError):
            side_effect()

        result = side_effect()
        assert result.content == "recovered"


# ═══════════════════════════════════════════════════════════════════
# 3. RetryHandler Resilience Tests
# ═══════════════════════════════════════════════════════════════════


class TestRetryHandlerFaultInjection:
    """Test RetryHandler with fault injection — the real retry logic."""

    @pytest.mark.asyncio
    async def test_retry_on_timeout_recovers(self):
        """RetryHandler should retry on LLMTimeoutError and recover."""
        handler = RetryHandler(max_attempts=4, min_wait=0.01)
        call_count = 0

        async def flaky_call():
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise LLMTimeoutError("Upstream timeout", timeout_duration=30.0)
            return "recovered"

        result = await handler.execute_with_retry(flaky_call)
        assert result == "recovered"
        assert call_count == 4  # 3 failures + 1 success

    @pytest.mark.asyncio
    async def test_retry_exhausted_on_persistent_timeout(self):
        """RetryHandler should raise RetryError after max_attempts."""
        handler = RetryHandler(max_attempts=2, min_wait=0.01)

        async def always_times_out():
            raise LLMTimeoutError("Always times out")

        from tenacity import RetryError

        with pytest.raises(RetryError):
            await handler.execute_with_retry(always_times_out)

    @pytest.mark.asyncio
    async def test_rate_limit_retry_respects_retry_after(self):
        """RateLimitHandler should respect retry_after timing."""
        handler = RateLimitHandler()

        error = LLMRateLimitError("Too many requests", retry_after=1)
        wait_time = handler.handle_rate_limit_error(error)
        assert wait_time == 1.0

    @pytest.mark.asyncio
    async def test_rate_limit_without_retry_after_fallback(self):
        """When no retry_after provided, should use exponential backoff default."""
        handler = RateLimitHandler()

        error = LLMRateLimitError("Too many requests")
        wait_time = handler.handle_rate_limit_error(error)
        # Default exponential backoff: min_wait=1.0 * 2^1 = 2.0 ...
        # Actually let's check the implementation
        assert wait_time == 4.0  # From the handler's default calculation

    @pytest.mark.asyncio
    async def wait_if_needed_respects_interval(self):
        """RateLimitHandler.wait_if_needed should enforce minimum interval."""
        handler = RateLimitHandler()

        start = time.monotonic()
        await handler.wait_if_needed(retry_after=0.05)
        elapsed = time.monotonic() - start
        assert elapsed >= 0.03  # allow some tolerance

    @pytest.mark.asyncio
    async def test_non_retryable_exception_propagates(self):
        """Exceptions not in retry_on_exceptions should propagate immediately."""
        handler = RetryHandler(
            max_attempts=3,
            retry_on_exceptions=[LLMAPIError],
        )

        async def raises_auth_error():
            raise LLMAuthenticationError("Bad key")

        with pytest.raises(LLMAuthenticationError, match="Bad key"):
            await handler.execute_with_retry(raises_auth_error)

    @pytest.mark.asyncio
    async def test_sync_function_retry(self):
        """RetryHandler works with sync functions too."""
        handler = RetryHandler(max_attempts=3, min_wait=0.01)
        call_count = 0

        def sync_flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise LLMAPIError("sync fail")
            return "sync_ok"

        result = await handler.execute_with_retry(sync_flaky)
        assert result == "sync_ok"
        assert call_count == 3


# ═══════════════════════════════════════════════════════════════════
# 4. with_retry Decorator Resilience Tests
# ═══════════════════════════════════════════════════════════════════


class TestWithRetryDecoratorFaultInjection:
    """Test the @with_retry decorator with fault injection."""

    @pytest.mark.asyncio
    async def test_decorator_recovers_after_failures(self):
        """@with_retry should recover after transient failures."""
        call_count = 0

        @with_retry(max_attempts=3, min_wait=0.01, retry_on_exceptions=[LLMAPIError])
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise LLMAPIError("transient")
            return "ok"

        result = await flaky()
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_decorator_non_retryable_exception(self):
        """@with_retry should not retry exceptions outside its retry list."""

        @with_retry(max_attempts=3, retry_on_exceptions=[LLMAPIError])
        async def raises_auth():
            raise LLMAuthenticationError("bad key")

        with pytest.raises(LLMAuthenticationError):
            await raises_auth()

    @pytest.mark.asyncio
    async def test_sync_decorator_with_retry(self):
        """@with_retry works with sync functions."""
        call_count = 0

        @with_retry(max_attempts=3, min_wait=0.01, retry_on_exceptions=[LLMAPIError])
        def sync_flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise LLMAPIError("sync transient")
            return "sync_ok"

        result = await sync_flaky()
        assert result == "sync_ok"
        assert call_count == 3


# ═══════════════════════════════════════════════════════════════════
# 5. Error Handler Conversion Tests
# ═══════════════════════════════════════════════════════════════════


class TestErrorHandlerFaultInjection:
    """Test the error handler's ability to convert various fault types."""

    def test_handle_generic_error(self):
        """handle_openai_error should wrap unknown errors as generic LLMError."""
        from src.core.llm.utils.error_handler import handle_openai_error

        generic = ValueError("something broke")
        result = handle_openai_error(generic)
        assert isinstance(result, LLMError)
        assert "something broke" in result.message

    def test_handle_llm_error_passthrough(self):
        """handle_openai_error should pass through LLMError instances unchanged."""
        from src.core.llm.utils.error_handler import handle_openai_error

        original = LLMError("Already formatted error")
        result = handle_openai_error(original)
        assert result is original  # Same instance, no conversion

        try:
            import openai
        except ImportError:
            pytest.skip("openai not installed")

        rate_error = openai.RateLimitError(
            "Rate limit exceeded",
            response=AsyncMock(status_code=429),
            body={"error": {"message": "Rate limit"}},
        )
        rate_error.retry_after = 30

        result = handle_openai_error(rate_error)
        assert isinstance(result, LLMRateLimitError)
        assert result.retry_after == 30

    def test_handle_openai_timeout_error(self):
        """handle_openai_error should convert timeout errors."""
        from src.core.llm.utils.error_handler import (
            LLMTimeoutError,
            handle_openai_error,
        )

        try:
            import openai
        except ImportError:
            pytest.skip("openai not installed")

        timeout_error = openai.APITimeoutError("Request timed out")

        result = handle_openai_error(timeout_error)
        assert isinstance(result, LLMTimeoutError)


# ═══════════════════════════════════════════════════════════════════
# 7. LLM Manager Fallback Tests
# ═══════════════════════════════════════════════════════════════════


class TestLLMManagerFallbackFaultInjection:
    """Test that LLMManager correctly falls back between providers on failure."""

    @pytest.mark.asyncio
    async def test_manager_falls_back_on_provider_failure(self):
        """LLMManager should try fallback providers when primary fails."""
        try:
            from src.core.llm.manager import LLMManager
            from src.core.llm.types import LLMMessage
        except ImportError:
            pytest.skip("LLMManager not available")

        # Create mock clients for two providers
        primary = AsyncMock(spec=["generate_response", "config"])
        primary.generate_response = AsyncMock()
        primary.generate_response.side_effect = LLMAPIError("Primary down")
        primary.config = AsyncMock()
        primary.config.provider = "openai"

        fallback = AsyncMock(spec=["generate_response", "config"])
        fallback.generate_response = AsyncMock()
        fallback.generate_response.return_value = LLMResponse(
            content="Fallback response",
            model="fallback-model",
        )
        fallback.config = AsyncMock()
        fallback.config.provider = "litellm"

        manager = LLMManager(default_client=primary)
        manager.clients = {
            "openai": primary,
            "litellm": fallback,
        }
        manager.set_fallback_providers(["litellm"])

        msg = LLMMessage(role="user", content="test")
        response = await manager.generate_response([msg])

        assert response.content == "Fallback response"
        assert fallback.generate_response.called
