"""
Tests for the Observability Module

Covers:
- MetricsRegistry Prometheus metrics definition correctness
- ObservabilityServer HTTP endpoint responses
- MonitoredLLMClient Prometheus integration
- Integration with existing MetricsCollector
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, patch

import pytest

from src.core.observability import MetricsRegistry, ObservabilityServer


# ═══════════════════════════════════════════════════════════════════
# 1. MetricsRegistry — Metric Definitions
# ═══════════════════════════════════════════════════════════════════


class TestMetricsRegistry:
    """Verify that all Prometheus metrics are correctly defined."""

    def test_llm_calls_counter(self):
        """llm_calls counter should exist with correct labels."""
        # Note: prometheus_client strips _total suffix internally
        assert MetricsRegistry.llm_calls._name == "ark_llm_calls"
        assert MetricsRegistry.llm_calls._labelnames == ("provider", "model", "status")

    def test_llm_latency_histogram(self):
        """llm_latency_seconds should be a Histogram."""
        assert MetricsRegistry.llm_latency_seconds._name == "ark_llm_latency_seconds"
        assert MetricsRegistry.llm_latency_seconds._labelnames == ("provider", "model")

    def test_tool_calls_counter(self):
        """tool_calls counter should exist with correct labels."""
        assert MetricsRegistry.tool_calls_total._name == "ark_tool_calls"
        assert MetricsRegistry.tool_calls_total._labelnames == (
            "tool_name",
            "server",
            "status",
        )

    def test_engine_ready_gauge(self):
        """engine_ready gauge should start at 0."""
        assert MetricsRegistry.engine_ready._name == "ark_engine_ready"
        assert MetricsRegistry.engine_ready._value.get() == 0.0

    def test_init_engine_info(self):
        """init_engine_info should set engine metadata (no crash)."""
        # Should not raise
        MetricsRegistry.init_engine_info(version="2.0.0")
        assert True

    def test_mark_engine_ready(self):
        """mark_engine_ready should set the gauge to 1."""
        MetricsRegistry.mark_engine_ready()
        assert MetricsRegistry.engine_ready._value.get() == 1.0

    def test_record_llm_call(self):
        """record_llm_call should increment counters."""
        before = MetricsRegistry.llm_calls.labels(
            provider="openai", model="gpt-4", status="success"
        )._value.get()

        MetricsRegistry.record_llm_call(
            provider="openai",
            model="gpt-4",
            status="success",
            latency=0.5,
            prompt_tokens=100,
            completion_tokens=50,
        )

        after = MetricsRegistry.llm_calls.labels(
            provider="openai", model="gpt-4", status="success"
        )._value.get()
        assert after == before + 1

    def test_record_tool_call(self):
        """record_tool_call should increment tool counters."""
        before = MetricsRegistry.tool_calls_total.labels(
            tool_name="web_search", server="mcp", status="success"
        )._value.get()

        MetricsRegistry.record_tool_call(
            tool_name="web_search",
            server="mcp",
            status="success",
            latency=0.3,
        )

        after = MetricsRegistry.tool_calls_total.labels(
            tool_name="web_search", server="mcp", status="success"
        )._value.get()
        assert after == before + 1

    def test_record_error(self):
        """record_error should increment error counters."""
        before = MetricsRegistry.errors_total.labels(
            component="llm", error_type="TimeoutError"
        )._value.get()

        MetricsRegistry.record_error(component="llm", error_type="TimeoutError")

        after = MetricsRegistry.errors_total.labels(
            component="llm", error_type="TimeoutError"
        )._value.get()
        assert after == before + 1


# ═══════════════════════════════════════════════════════════════════
# 2. ObservabilityServer — HTTP Endpoints
# ═══════════════════════════════════════════════════════════════════


class TestObservabilityServer:
    """Test the metrics HTTP server."""

    @pytest.mark.asyncio
    async def test_server_start_stop(self):
        """Server should start and stop cleanly."""
        server = ObservabilityServer(port=9101)
        assert not server.is_running

        await server.start()
        assert server.is_running

        await server.stop()
        assert not server.is_running

    @pytest.mark.asyncio
    async def test_start_twice_does_not_error(self):
        """Starting an already-running server should log a warning, not crash."""
        server = ObservabilityServer(port=9102)
        await server.start()
        await server.start()  # Should not raise
        await server.stop()

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Health endpoint should return JSON with status ok."""
        server = ObservabilityServer(port=9103)
        await server.start()

        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", 9103)
            writer.write(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()

            response = b""
            while True:
                chunk = await asyncio.wait_for(reader.read(4096), timeout=2.0)
                if not chunk:
                    break
                response += chunk

            assert b"200 OK" in response or b"200" in response
            # Check that body contains JSON
            assert b"status" in response
            assert b"ok" in response
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self):
        """Metrics endpoint should return Prometheus format."""
        server = ObservabilityServer(port=9104)
        await server.start()

        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", 9104)
            writer.write(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()

            response = b""
            while True:
                chunk = await asyncio.wait_for(reader.read(4096), timeout=2.0)
                if not chunk:
                    break
                response += chunk

            assert b"200 OK" in response or b"200" in response
            # Prometheus metrics output should start with # HELP or include metric names
            assert b"ark_" in response
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_unknown_path_returns_404(self):
        """Unknown paths should return 404."""
        server = ObservabilityServer(port=9105)
        await server.start()

        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", 9105)
            writer.write(b"GET /unknown HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()

            response = b""
            while True:
                chunk = await asyncio.wait_for(reader.read(4096), timeout=2.0)
                if not chunk:
                    break
                response += chunk

            assert b"404" in response
            assert b"Not Found" in response
        finally:
            await server.stop()

    def test_get_server_singleton(self):
        """get_observability_server should return the same instance."""
        from src.core.observability import get_observability_server

        s1 = get_observability_server()
        s2 = get_observability_server()
        assert s1 is s2

    def test_get_server_with_different_port(self):
        """get_observability_server should return singleton regardless of port arg."""
        from src.core.observability import get_observability_server

        # Both calls return the same singleton instance
        s1 = get_observability_server(port=9999)
        s2 = get_observability_server(port=8888)
        assert s1 is s2


# ═══════════════════════════════════════════════════════════════════
# 3. MonitoredLLMClient — Prometheus Integration
# ═══════════════════════════════════════════════════════════════════


class TestMonitoredLLMClientObservability:
    """Test that MonitoredLLMClient records Prometheus metrics."""

    @pytest.mark.asyncio
    async def test_successful_call_records_metrics(self):
        """A successful LLM call should increment counters and record latency."""
        from src.core.llm.client import MonitoredLLMClient, BaseLLMClient
        from src.core.llm.types import LLMConfig, LLMProvider, LLMMessage, LLMResponse

        # Create a mock client
        mock_inner = AsyncMock(spec=BaseLLMClient)
        mock_inner.generate_response = AsyncMock(
            return_value=LLMResponse(
                content="test response",
                model="gpt-4",
                usage={"prompt_tokens": 50, "completion_tokens": 30},
            )
        )
        mock_inner.config = LLMConfig(
            provider=LLMProvider.OPENAI, model="gpt-4", api_key="test"
        )
        mock_inner.logger = None
        mock_inner._initialized = True

        before = MetricsRegistry.llm_calls.labels(
            provider="openai", model="gpt-4", status="success"
        )._value.get()

        client = MonitoredLLMClient(mock_inner)
        msg = LLMMessage(role="user", content="hello")
        response = await client.generate_response([msg])

        assert response.content == "test response"

        after = MetricsRegistry.llm_calls.labels(
            provider="openai", model="gpt-4", status="success"
        )._value.get()
        assert after == before + 1

    @pytest.mark.asyncio
    async def test_failed_call_records_error_metric(self):
        """A failed LLM call should record error metrics."""
        from src.core.llm.client import MonitoredLLMClient, BaseLLMClient
        from src.core.llm.types import LLMConfig, LLMProvider, LLMMessage

        mock_inner = AsyncMock(spec=BaseLLMClient)
        mock_inner.generate_response = AsyncMock(side_effect=ValueError("API error"))
        mock_inner.config = LLMConfig(
            provider=LLMProvider.OPENAI, model="gpt-4", api_key="test"
        )
        mock_inner.logger = None
        mock_inner._initialized = True

        before = MetricsRegistry.errors_total.labels(
            component="llm", error_type="ValueError"
        )._value.get()

        client = MonitoredLLMClient(mock_inner)
        msg = LLMMessage(role="user", content="hello")

        with pytest.raises(ValueError, match="API error"):
            await client.generate_response([msg])

        after = MetricsRegistry.errors_total.labels(
            component="llm", error_type="ValueError"
        )._value.get()
        assert after == before + 1


# ═══════════════════════════════════════════════════════════════════
# 4. Prometheus Output Format Validation
# ═══════════════════════════════════════════════════════════════════


class TestPrometheusOutput:
    """Validate that Prometheus output is well-formed."""

    def test_generate_latest_contains_ark_metrics(self):
        """generate_latest() should include our custom metrics."""
        from prometheus_client import generate_latest

        output = generate_latest().decode("utf-8")

        # Should contain our namespace prefix
        assert "ark_" in output

        # Should contain metric definitions
        assert "ark_llm_calls_total" in output
        assert "ark_tool_calls_total" in output
        assert "ark_errors_total" in output
