# Harness Engineering Standards

*Version 1.2 — Last Updated: May 20, 2026*

This document defines the **harness engineering standards** for Project Jarvis. It serves as the authoritative reference for how the system is designed, tested, observed, and operated as an integrated engineering harness.

---

## Table of Contents

1. [What is Harness Engineering?](#1-what-is-harness-engineering)
2. [The Six Pillars](#2-the-six-pillars)
   - [2.1 Test Harness](#21-test-harness)
   - [2.2 Integration Harness](#22-integration-harness)
   - [2.3 Configuration Harness](#23-configuration-harness)
   - [2.4 Security Harness](#24-security-harness)
   - [2.5 Observability Harness](#25-observability-harness)
   - [2.6 Orchestration Harness](#26-orchestration-harness)
3. [Implementation Checklist](#3-implementation-checklist)
4. [Review Criteria](#4-review-criteria)

---

## 1. What is Harness Engineering?

Harness engineering is the discipline of building **integrated control and measurement frameworks** around software systems. In the context of Project Jarvis, a "harness" is not a single component — it is the sum of all the infrastructure that:

- **Integrates** components together reliably (Integration Harness)
- **Validates** that the system behaves correctly (Test Harness)
- **Configures** the system without code changes (Configuration Harness)
- **Protects** the system from misuse and failure (Security Harness)
- **Observes** what the system is doing in real time (Observability Harness)
- **Orchestrates** the flow of data and control (Orchestration Harness)

A well-engineered harness means the difference between a demo that works on a laptop and a system that can be deployed, maintained, and evolved with confidence.

---

## 2. The Six Pillars

### 2.1 Test Harness

**Principle**: Every component must be testable in isolation, and the system as a whole must be testable deterministically.

#### Standards

| Standard | Requirement | Verification |
|----------|-------------|--------------|
| **Mock Coverage** | Every external dependency (LLM provider, MCP server, database) must have a mock fixture in the root `conftest.py` | Code review — grep for `Mock(spec=` |
| **Fault Injection** | Every retry-able operation must have tests that simulate: timeout, rate limit, malformed response, and service unavailable | Check `test_retry_handler.py` and equivalent per-module tests |
| **Determinism** | Tests must not depend on real network calls, real API keys, or wall-clock timing | Run `pytest --offline` (planned) |
| **Coverage Floor** | Effective immediately, newly added or modified lines in the current diff must maintain 100% diff coverage | `diff-cover coverage.xml --compare-branch=origin/main --fail-under=100` |
| **Coverage Integrity** | Tests must not be invented solely to inflate coverage; every case must validate a meaningful behavior, contract, edge case, or failure mode | Code review of test intent and assertions |
| **Async Correctness** | All async tests must use `pytest-asyncio` with proper event loop management | CI check |

#### Mock Architecture

```
conftest.py (project root) provides:
├── MockLLMClient        → Simulates any LLM provider response
├── MockMCPClient        → Simulates MCP server interactions
├── MockContextManager   → Simulates conversation state
├── MockToolRegistry     → Simulates tool discovery/registration
└── MockServerConfigMgr  → Simulates server configuration loading
```

All mocks use `unittest.mock.Mock` / `AsyncMock` with `spec=` to ensure interface compliance.

#### Fault Injection Pattern

```python
# Example: Testing retry behavior on LLM timeout
async def test_llm_retry_on_timeout(mock_llm_client):
    """LLMManager should retry on timeout up to max_attempts."""
    mock_llm_client.ainvoke.side_effect = [
        TimeoutError("Connection timed out"),  # 1st attempt fails
        TimeoutError("Connection timed out"),  # 2nd attempt fails
        {"choices": [{"message": {"content": "Success"}}]}  # 3rd succeeds
    ]
    result = await llm_manager.ainvoke("test prompt")
    assert mock_llm_client.ainvoke.call_count == 3
    assert result["choices"][0]["message"]["content"] == "Success"
```

---

### 2.2 Integration Harness

**Principle**: All external capabilities are integrated through standardized protocols, not ad-hoc connections.

#### Standards

| Standard | Requirement | Verification |
|----------|-------------|--------------|
| **MCP-First** | All external tool integration MUST use the Model Context Protocol | No direct HTTP/gRPC calls from business logic |
| **Abstraction Layer** | LLM providers must be accessed through `LLMManager`, not provider SDKs directly | Import check — no `openai.ChatCompletion` outside `src/core/llm/` |
| **Lifecycle Management** | All integration resources (MCP servers, DB connections) must have explicit startup/shutdown lifecycle | Check `__aenter__`/`__aexit__` or explicit `startup()`/`shutdown()` methods |
| **Graceful Degradation** | If an MCP server disconnects, the system must continue functioning (with reduced capability) rather than crash | Test with `mock_mcp_client.get_server_status.return_value = {"connected": False}` |

#### Integration Architecture

```
┌──────────────┐     MCP Protocol      ┌────────────────┐
│  ARKMCPClient │ ◄──────────────────► │  MCP Server A   │
│  (Tool Router) │                      │  (filesystem)   │
└──────┬───────┘                       └────────────────┘
       │                                 ┌────────────────┐
       ├──────────────────────────────► │  MCP Server B   │
       │                                 │  (web search)   │
       │                                 └────────────────┘
       │                                 ┌────────────────┐
       └──────────────────────────────► │  MCP Server C   │
                                         │  (database)     │
                                         └────────────────┘
```

---

### 2.3 Configuration Harness

**Principle**: The system must be configurable without code changes, at multiple granularity levels.

#### Three-Tier Configuration Model

| Tier | Mechanism | Scope | Override Priority |
|------|-----------|-------|-------------------|
| 1 — Defaults | YAML files (`config/llm_config.yaml`) | Global | Lowest |
| 2 — Environment | `${VARIABLE_NAME}` substitution | Per-deployment | Medium |
| 3 — Runtime | In-memory overrides via `JarvisConfig` | Per-session | Highest |

#### Standards

| Standard | Requirement | Verification |
|----------|-------------|--------------|
| **Schema Validation** | All configuration YAML files MUST have a corresponding Pydantic/dataclass schema validated at load time | Check config loader raises clear error on unknown keys |
| **Secrets Management** | API keys and tokens MUST use `${ENV_VAR}` syntax, never hardcoded values | grep for `api_key:` — must match `${*}` pattern |
| **Environment Profiles** | Must support `development`, `testing`, and `production` profiles with appropriate defaults | Check `config/examples/` |
| **Hot-Reload Readiness** | Configuration objects should be designed for eventual hot-reload (immutable snapshots, not global singletons) | Review per-pattern |

#### Configuration Loading Flow

```
config/llm_config.yaml
        │
        ▼
  YAML Parser (env var substitution)
        │
        ▼
  Dataclass/Pydantic Validation
        │
        ▼
  Runtime Overrides (per-session)
        │
        ▼
  Immutable Config Snapshot
```

---

### 2.4 Security Harness

**Principle**: Every operation must be validated, rate-limited, and audited.

#### Standards

| Standard | Requirement | Verification |
|----------|-------------|--------------|
| **Tool Validation** | Every tool execution must pass through `SecurityManager.validate_tool_execution()` | Static analysis — security manager invoked before any `execute_tool` |
| **Code Execution** | Python code MUST execute in Docker or E2B sandbox only — never on the host | Review `src/capabilities/interpreter.py` |
| **Rate Limiting** | Per-minute and per-hour rate limits must be enforced at the security layer | Test with rate-limit-exceeding scenarios |
| **Audit Logging** | All tool executions, security violations, and auth failures must be logged with user ID, tool name, timestamp, and result | Check `audit_logger` usage |
| **Input Validation** | All user-supplied tool arguments must be validated for size, type, and allowed patterns | Review `ValidationRequest` handling |

#### Security Flow

```
User Request
    │
    ▼
SecurityManager.validate_tool_execution()
    ├── RateLimiter.check()      → Reject if exceeded
    ├── PermissionChecker.check() → Reject if unauthorized
    ├── InputValidator.check()    → Reject if malformed
    │
    ▼
AuditLogger.log(user, tool, args, result)
    │
    ▼
Tool Execution (via MCP or Local)
```

---

### 2.5 Observability Harness

**Principle**: The system must be transparent — you cannot manage what you cannot measure.

#### Standards

| Standard | Requirement | Verification |
|----------|-------------|--------------|
| **Structured Logging** | All logs must use structured format with consistent fields (`component`, `event`, `duration_ms`, `error`) | Review log statements |
| **Metrics Collection** | Key operations must emit metrics: LLM call count/latency/tokens, tool execution count/duration, error rate by type | Check `metrics_collector.py` usage |
| **Audit Trail** | All security-relevant events must be persisted to the audit log with immutable timestamps | Check `audit_logger` |
| **Tool Usage Tracking** | The `ToolUsageStats` collector must track which tools are called, how often, and their success/failure rate | Review `ARKEngine` stats |

#### Key Metrics

| Metric | Source | Why it matters |
|--------|--------|----------------|
| `llm.call.count` | `LLMManager` | Usage tracking and cost estimation |
| `llm.call.latency_ms` | `LLMManager` | Performance regression detection |
| `llm.token.total` | `MetricsCollector` | Cost and context window management |
| `tool.execution.count` | `ARKEngine` | Feature adoption measurement |
| `tool.execution.error_rate` | `ARKEngine` | Reliability monitoring |
| `security.rate_limit.hits` | `SecurityManager` | Abuse pattern detection |

---

### 2.6 Orchestration Harness

**Principle**: The flow of data and control between components must be explicit, observable, and recoverable.

#### LangGraph as Orchestration Backplane

The ARK engine uses **LangGraph** as the core orchestration harness, implementing a **Master-Tools** pattern:

```
User Input
    │
    ▼
┌──────────┐
│ MasterNode│ ←── (Reasoning: should I use a tool or respond?)
└─────┬────┘
      │
      ├── "respond" ──► Send final answer to user
      │
      └── "use_tool" ──► ┌──────────┐
                          │ ToolsNode │ ←── (Execute tool, collect result)
                          └─────┬────┘
                                │
                                └──► Back to MasterNode (next reasoning cycle)
```

#### Standards

| Standard | Requirement | Verification |
|----------|-------------|--------------|
| **State Machine** | All orchestration must be defined as a state graph with explicit states and transitions | Review `StateGraph(JarvisState)` |
| **Max Iterations** | The orchestration loop must have a configurable maximum iteration count to prevent infinite loops | Check `max_iterations` parameter |
| **State Checkpointing** | The graph state must be serializable for future checkpoint/recovery support | Review `JarvisState` dataclass |
| **Conditional Routing** | Routing decisions must be explicit (conditional edges), not implicit | Review `should_continue()` function |

---

## 3. Implementation Checklist

When adding any new feature or component, verify against this checklist:

- [ ] **Test Harness**: Mock fixtures created? Fault injection tests written? Current diff at 100% diff coverage? Test cases reflect real behavior instead of coverage-padding?
- [ ] **Integration Harness**: Does it use MCP or `LLMManager`? Lifecycle methods defined? Graceful degradation handled?
- [ ] **Configuration Harness**: Parameters externalized? Schema validation? Environment variable support?
- [ ] **Security Harness**: Tool validation path? Rate limits? Audit logging? Input validation?
- [ ] **Observability Harness**: Structured logs? Metrics collected? Errors tracked?
- [ ] **Orchestration Harness**: State transitions explicit? Max iterations bounded? State serializable?

---

## 4. Review Criteria

Code reviews must evaluate:

1. **Does this change introduce a new external dependency without a mock?** → Reject
2. **Does this change hardcode a configuration value?** → Reject
3. **Does this change call an LLM or tool outside the established abstraction layer?** → Reject
4. **Does this change skip error handling for known failure modes?** → Reject
5. **Does this change operate without structured logging or metrics?** → Request changes
6. **Does this change lack tests for failure paths?** → Request changes
7. **Does this change add tests that exist only to boost coverage without validating real behavior?** → Reject

---

*This document is a living standard. It should be reviewed and updated as the project evolves.*
