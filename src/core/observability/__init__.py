"""
Observability Module

Provides Prometheus-based metrics collection, health check endpoints,
and structured monitoring for the ARK engine.

Usage:
    from src.core.observability import MetricsRegistry, observability_server

    # Record an LLM call
    MetricsRegistry.llm_calls.labels(provider="openai", status="success").inc()

    # Start the metrics server (auto-started by web app)
    await observability_server.start()

Components:
    - MetricsRegistry: Central registry of Prometheus metrics
    - ObservabilityServer: HTTP server exposing /metrics and /health
    - LLMObservabilityMiddleware: Tracks LLM call metrics transparently
"""

from .metrics import MetricsRegistry
from .server import ObservabilityServer, get_observability_server

observability_server = get_observability_server()

__all__ = [
    "MetricsRegistry",
    "ObservabilityServer",
    "observability_server",
    "get_observability_server",
]
