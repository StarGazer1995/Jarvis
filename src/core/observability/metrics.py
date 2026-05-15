"""
Prometheus Metrics Definitions

Defines all Prometheus metrics used across the ARK engine.
Uses a central MetricsRegistry for consistent naming and labeling.
"""

from prometheus_client import Counter, Histogram, Gauge, Info


class MetricsRegistry:
    """
    Central registry of all Prometheus metrics for the ARK engine.

    All metrics use the ``ark_`` namespace prefix and follow
    Prometheus naming conventions (snake_case).
    """

    # ── LLM Metrics ──────────────────────────────────────────────

    llm_calls: Counter = Counter(
        "ark_llm_calls_total",
        "Total number of LLM calls",
        labelnames=["provider", "model", "status"],
    )

    llm_latency_seconds: Histogram = Histogram(
        "ark_llm_latency_seconds",
        "LLM call latency in seconds",
        labelnames=["provider", "model"],
        buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
    )

    llm_tokens_total: Counter = Counter(
        "ark_llm_tokens_total",
        "Total tokens consumed by LLM calls",
        labelnames=["provider", "model", "type"],  # type: prompt, completion
    )

    llm_in_flight: Gauge = Gauge(
        "ark_llm_in_flight",
        "Number of LLM calls currently in flight",
        labelnames=["provider"],
    )

    # ── Tool Execution Metrics ───────────────────────────────────

    tool_calls_total: Counter = Counter(
        "ark_tool_calls_total",
        "Total number of tool executions",
        labelnames=["tool_name", "server", "status"],
    )

    tool_latency_seconds: Histogram = Histogram(
        "ark_tool_latency_seconds",
        "Tool execution latency in seconds",
        labelnames=["tool_name", "server"],
        buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
    )

    tool_in_flight: Gauge = Gauge(
        "ark_tool_in_flight",
        "Number of tool executions currently in flight",
        labelnames=["server"],
    )

    # ── Conversation Metrics ─────────────────────────────────────

    conversations_total: Counter = Counter(
        "ark_conversations_total",
        "Total number of conversations",
        labelnames=["status"],  # status: active, completed, error
    )

    turns_per_conversation: Histogram = Histogram(
        "ark_conversation_turns",
        "Number of turns per conversation",
        buckets=(1, 2, 5, 10, 20, 50, 100),
    )

    messages_total: Counter = Counter(
        "ark_messages_total",
        "Total number of messages processed",
        labelnames=["role"],  # role: user, assistant, tool, system
    )

    # ── Engine Health Metrics ────────────────────────────────────

    engine_info: Info = Info(
        "ark_engine",
        "ARK engine metadata (version, status)",
    )

    engine_uptime_seconds: Gauge = Gauge(
        "ark_engine_uptime_seconds",
        "Time since the engine started",
    )

    engine_ready: Gauge = Gauge(
        "ark_engine_ready",
        "Whether the engine is ready (1) or not (0)",
    )

    graph_iterations_total: Counter = Counter(
        "ark_graph_iterations_total",
        "Total number of LangGraph iterations",
        labelnames=["node"],  # node: master, tools
    )

    # ── MCP Server Metrics ───────────────────────────────────────

    mcp_servers_total: Gauge = Gauge(
        "ark_mcp_servers_total",
        "Number of connected MCP servers",
        labelnames=["status"],  # status: connected, disconnected, error
    )

    mcp_tools_total: Gauge = Gauge(
        "ark_mcp_tools_total",
        "Number of available MCP tools",
    )

    # ── Error Metrics ────────────────────────────────────────────

    errors_total: Counter = Counter(
        "ark_errors_total",
        "Total number of errors by type",
        labelnames=["component", "error_type"],
    )

    retries_total: Counter = Counter(
        "ark_retries_total",
        "Total number of retry attempts",
        labelnames=["component"],
    )

    # ── Method ──────────────────────────────────────────────────

    @classmethod
    def init_engine_info(cls, version: str = "1.0.0") -> None:
        """Set the engine info metric once at startup."""
        cls.engine_info.info({"version": version, "status": "initializing"})

    @classmethod
    def mark_engine_ready(cls, version: str = "1.0.0") -> None:
        """Mark the engine as ready."""
        cls.engine_ready.set(1)
        cls.engine_info.info({"version": version, "status": "ready"})

    @classmethod
    def record_llm_call(
        cls,
        provider: str,
        model: str,
        status: str,
        latency: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Record a complete LLM call with all associated metrics."""
        cls.llm_calls.labels(provider=provider, model=model, status=status).inc()
        cls.llm_latency_seconds.labels(provider=provider, model=model).observe(latency)
        if prompt_tokens:
            cls.llm_tokens_total.labels(
                provider=provider, model=model, type="prompt"
            ).inc(prompt_tokens)
        if completion_tokens:
            cls.llm_tokens_total.labels(
                provider=provider, model=model, type="completion"
            ).inc(completion_tokens)

    @classmethod
    def record_tool_call(
        cls,
        tool_name: str,
        server: str,
        status: str,
        latency: float,
    ) -> None:
        """Record a tool execution with timing."""
        cls.tool_calls_total.labels(
            tool_name=tool_name, server=server, status=status
        ).inc()
        cls.tool_latency_seconds.labels(tool_name=tool_name, server=server).observe(
            latency
        )

    @classmethod
    def record_error(cls, component: str, error_type: str) -> None:
        """Record an error occurrence."""
        cls.errors_total.labels(component=component, error_type=error_type).inc()
