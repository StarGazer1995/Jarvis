"""
Performance Benchmarks

Establishes baseline performance metrics for critical operations
using pytest-benchmark. Run with:

    uv run pytest tests/test_performance.py --benchmark-only
    uv run pytest tests/test_performance.py --benchmark-compare  # compare with saved
"""

import time
from unittest.mock import AsyncMock

import pytest

from src.core.llm.utils.retry_handler import RetryHandler
from src.core.llm.utils.error_handler import LLMAPIError, LLMTimeoutError
from src.core.llm.types import LLMResponse

from tests.fault_injection import FaultConfig, FaultInjector
from tests.fault_injection.llm_faults import (
    llm_timeout_then_succeed,
    llm_timeout,
)


# ═══════════════════════════════════════════════════════════════════
# 1. FaultInjector Overhead
# ═══════════════════════════════════════════════════════════════════


class TestFaultInjectorBenchmarks:
    """Measure the overhead of the FaultInjector framework itself."""

    def test_bare_mock_call(self, benchmark):
        """Baseline: how fast is a raw AsyncMock call?"""
        mock = AsyncMock()
        mock.func = AsyncMock(return_value="ok")

        def run():
            import asyncio

            async def _run():
                for _ in range(100):
                    await mock.func()
                return "done"

            return asyncio.run(_run())

        result = benchmark(run)
        assert result == "done"

    def test_fault_injector_success_path(self, benchmark):
        """FaultInjector with no faults (always succeeds)."""
        mock = AsyncMock()
        mock.func.side_effect = FaultInjector(success_value="ok").build()

        def run():
            import asyncio

            async def _run():
                for _ in range(100):
                    await mock.func()
                return "done"

            return asyncio.run(_run())

        result = benchmark(run)
        assert result == "done"

    def test_fault_injector_one_fault(self, benchmark):
        """FaultInjector with 1 fault then success — per-call overhead."""
        mock = AsyncMock()
        mock.func.side_effect = FaultInjector(
            FaultConfig(ValueError, args=("boom",), fail_count=1),
            success_value="ok",
        ).build()

        def run():
            import asyncio

            async def _run():
                try:
                    await mock.func()
                except ValueError:
                    pass
                return await mock.func()

            return asyncio.run(_run())

        result = benchmark(run)
        assert result == "ok"

    def test_llm_timeout_then_succeed_overhead(self, benchmark):
        """Pre-built LLM fault helper overhead."""
        mock = AsyncMock()
        mock.generate_response.side_effect = llm_timeout_then_succeed(
            fail_count=3,
            success_value=LLMResponse(content="ok", model="test"),
        )

        def run():
            import asyncio

            async def _run():
                for _ in range(3):
                    try:
                        await mock.generate_response()
                    except LLMTimeoutError:
                        pass
                return await mock.generate_response()

            return asyncio.run(_run())

        result = benchmark(run)
        assert result.content == "ok"


# ═══════════════════════════════════════════════════════════════════
# 2. RetryHandler Overhead
# ═══════════════════════════════════════════════════════════════════


class TestRetryHandlerBenchmarks:
    """Measure the overhead of the RetryHandler with tenacity."""

    def test_retry_success_first_try(self, benchmark):
        """Best case: no retry needed, just pass through."""
        handler = RetryHandler(max_attempts=3, min_wait=0.001)

        def run():
            import asyncio

            async def _run():
                for _ in range(50):
                    await handler.execute_with_retry(lambda: "ok")
                return "done"

            return asyncio.run(_run())

        result = benchmark(run)
        assert result == "done"

    def test_retry_with_failures(self, benchmark):
        """Retry handler with actual retries (worst-case overhead)."""
        handler = RetryHandler(max_attempts=4, min_wait=0.001)

        def run():
            import asyncio

            call_count = 0

            async def flaky():
                nonlocal call_count
                call_count += 1
                if call_count < 4:
                    raise LLMAPIError("transient")
                return "recovered"

            async def _run():
                nonlocal call_count
                call_count = 0
                return await handler.execute_with_retry(flaky)

            return asyncio.run(_run())

        result = benchmark(run)
        assert result == "recovered"

    def test_retry_exhausted_overhead(self, benchmark):
        """Measure overhead when retry ultimately fails (max attempts)."""
        handler = RetryHandler(max_attempts=3, min_wait=0.001)

        from tenacity import RetryError

        def run():
            import asyncio

            async def always_fails():
                raise LLMTimeoutError("Always times out")

            async def _run():
                try:
                    await handler.execute_with_retry(always_fails)
                except RetryError:
                    pass

            asyncio.run(_run())

        benchmark(run)


# ═══════════════════════════════════════════════════════════════════
# 3. Config Loading Overhead
# ═══════════════════════════════════════════════════════════════════


class TestConfigLoadingBenchmarks:
    """Measure config loading and parsing overhead."""

    def test_config_loading(self, benchmark):
        """Time to load the full production config file."""
        from src.core.config.loader import load_llm_config

        # First load to warm cache, then benchmark with reload
        load_llm_config(reload=True)

        def run():
            return load_llm_config(reload=True)

        config = benchmark(run)
        assert config is not None

    def test_config_parsing_overhead(self, benchmark):
        """Time to parse a config dict into dataclasses."""
        from src.core.config.loader import (
            ConfigLoader,
            Environment,
        )

        raw_config = {
            "global": {
                "default_provider": "openai",
                "retry": {
                    "max_attempts": 3,
                    "initial_delay": 1.0,
                    "max_delay": 60.0,
                    "exponential_base": 2.0,
                },
                "timeout": {"connect": 30.0, "read": 120.0, "total": 180.0},
                "log_level": "INFO",
            },
            "providers": {
                "openai": {
                    "type": "openai",
                    "enabled": True,
                    "api_key": "${OPENAI_API_KEY}",
                    "default_model": "gpt-4",
                    "models": {
                        "gpt-4": {"max_tokens": 4096, "temperature": 0.7},
                        "gpt-4-turbo": {"max_tokens": 4096, "temperature": 0.7},
                        "gpt-3.5-turbo": {"max_tokens": 4096, "temperature": 0.7},
                    },
                },
                "anthropic": {
                    "type": "anthropic",
                    "enabled": False,
                    "api_key": "${ANTHROPIC_API_KEY}",
                    "default_model": "claude-3-opus",
                    "models": {
                        "claude-3-opus": {"max_tokens": 4096, "temperature": 0.7},
                    },
                },
            },
            "features": {
                "streaming": {"enabled": True, "chunk_size": 1024},
                "context": {"max_history": 10, "max_tokens": 8192},
            },
        }

        loader = ConfigLoader()
        env = Environment.DEVELOPMENT

        def run():
            return loader._parse_config(raw_config, env)

        config = benchmark(run)
        assert config is not None


# ═══════════════════════════════════════════════════════════════════
# 4. Schema Validation Overhead
# ═══════════════════════════════════════════════════════════════════


class TestSchemaValidationBenchmarks:
    """Measure the overhead of Pydantic-based schema validation."""

    def test_validate_full_config(self, benchmark):
        """Time to validate a complete provider configuration through Pydantic."""
        from src.core.config.schema import ProviderConfigSchema, ModelConfigSchema

        def run():
            return ProviderConfigSchema(
                type="openai",
                enabled=True,
                api_key="sk-test",
                base_url="https://api.openai.com/v1",
                default_model="gpt-4",
                models={
                    "gpt-4": ModelConfigSchema(max_tokens=4096, temperature=0.7),
                    "gpt-4-turbo": ModelConfigSchema(max_tokens=8192, temperature=0.5),
                    "gpt-3.5-turbo": ModelConfigSchema(
                        max_tokens=4096, temperature=0.7
                    ),
                },
            )

        config = benchmark(run)
        assert config.default_model == "gpt-4"

    def test_validate_faulty_config_overhead(self, benchmark):
        """Time to reject an invalid config (failure path)."""
        from src.core.config.schema import ProviderConfigSchema, ModelConfigSchema

        def run():
            try:
                ProviderConfigSchema(
                    type="openai",
                    base_url="not-a-url",
                    default_model="missing-model",
                    models={
                        "gpt-4": ModelConfigSchema(),
                    },
                )
            except ValueError:
                pass

        benchmark(run)

    def test_validate_config_dict_overhead(self, benchmark):
        """Time for the full validate_config_dict bridge function."""
        from src.core.config.schema import validate_config_dict
        from src.core.config.loader import ProviderConfig, ModelConfig

        provider = ProviderConfig(
            type="openai",
            enabled=True,
            api_key="sk-test",
            default_model="gpt-4",
            models={"gpt-4": ModelConfig(max_tokens=4096)},
        )

        def run():
            return validate_config_dict(
                raw_config={
                    "global": {
                        "default_provider": "test",
                        "log_level": "INFO",
                    },
                },
                provider_configs={"test": provider},
                features={},
                environment="development",
            )

        result = benchmark(run)
        assert result.is_valid


# ═══════════════════════════════════════════════════════════════════
# 5. MCP Client Overhead
# ═══════════════════════════════════════════════════════════════════


class TestMCPBenchmarks:
    """Measure MCP client operation overhead."""

    def test_list_tools_with_builtins(self, benchmark):
        """Time to list built-in tools (no MCP servers)."""
        from src.core.mcp.client import ARKMCPClient

        client = ARKMCPClient()

        import asyncio

        def run():
            return asyncio.run(client.list_tools())

        tools = benchmark(run)
        assert len(tools) >= 2
