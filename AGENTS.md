# Project Jarvis — Agent Instructions

This file defines project-wide guidelines for AI coding agents working on the Jarvis AI agent framework. All agents must follow these harness engineering standards.

---

## Architecture

### Core Pattern: Layered Harness

Jarvis follows a **layered integration harness** architecture:

```
UI/CLI → Agent → ARK Engine → LLM Providers + MCP Tools
         ↑_____________________________|
                (LangGraph loop)
```

- **JarvisAgent**: Conversational interface layer (Chainlit web UI or CLI)
- **ARK Engine** (`src/core/ark/`): Core orchestration harness using LangGraph
  - `MasterNode`: Reasoning/decision node
  - `ToolsNode`: Execution node for MCP + local tools
- **LLM Layer** (`src/core/llm/`): Multi-provider abstraction (OpenAI, LiteLLM, NVIDIA NIM)
- **MCP Layer** (`src/core/mcp/`): Tool integration via Model Context Protocol
- **Security Manager** (`src/core/security/`): Policy enforcement harness
- **Context Manager** (`src/core/context/`): Conversation state & memory harness

### Design Principles

1. **Every external dependency MUST have a mock counterpart** in `tests/conftest.py` — this is non-negotiable for test harness integrity.
2. **Configuration is always externalized** — never hardcode API keys, model names, or thresholds.
3. **All LLM interaction goes through `LLMManager`** — never call a provider SDK directly from business logic.
4. **All tool execution goes through `ARKMCPClient`** — never call external tools directly.

---

## Code Style

- **Language**: Python 3.10+
- **Formatting**: Ruff with line length 88 (`pyproject.toml` config)
- **Type hints**: Required on all function signatures and dataclass fields
- **Docstrings**: Google-style docstrings for all public classes and methods
- **Async**: Use `async/await` for all I/O-bound operations
- **Error handling**: Use custom exception hierarchy (`JarvisError`, `LLMError`, `ProviderError`, etc.) — never raise bare `Exception`
- **Logging**: Use structured logging via the project's logger — never `print()`

---

## Build and Test

```bash
# Install dependencies
uv sync

# Run full test suite
uv run pytest

# Run with coverage
uv run pytest --cov

# Run specific test file
uv run pytest tests/core/ark/test_engine.py -v

# Run linter
uv run ruff check .
```

### Test Harness Requirements

Every agent MUST adhere to these standards when writing or modifying code:

1. **Every new feature MUST include tests** — unit tests for logic, integration tests for MCP/LLM boundaries.
2. **Use the existing mock fixtures** (`MockLLMClient`, `MockMCPClient`, `MockContextManager`) from `tests/conftest.py`. Never hit real APIs in unit tests.
3. **All async tests must use `pytest-asyncio`** with `asyncio_mode = "auto"` (already configured).
4. **Test both success and failure paths** — include tests for rate limits, timeouts, malformed responses.
5. **Coverage target**: ≥90% for new code, measured via `pytest-cov`.

---

## Conventions

### File Organization

| Path | Purpose |
|------|---------|
| `src/core/ark/` | LangGraph orchestration (MasterNode, ToolsNode, graph definition) |
| `src/core/llm/` | LLM provider clients, cache, metrics |
| `src/core/mcp/` | MCP client, tool registry, server config |
| `src/core/context/` | Conversation history, memory management |
| `src/core/config/` | Configuration loading (YAML + env vars) |
| `src/core/security/` | Tool validation, rate limiting, audit |
| `src/core/prompt/` | Prompt templates and management |
| `src/capabilities/` | Tool/server implementations |
| `src/web/` | Chainlit web UI |
| `tests/` | Mirrors `src/` structure |

### Harness Engineering Standards

When adding or modifying any component:

1. **Test Harness First**: Write the mock and test fixture before implementing the real component.
2. **Fault Injection**: Include tests that simulate failures (timeouts, bad responses, crashes) — not just happy paths.
3. **Observability**: Every new feature MUST emit structured logs. Metrics should be added for rate-limited or failure-prone operations.
4. **Configuration**: New configurable parameters MUST have defaults, environment variable overrides, and schema validation.
5. **Security**: Any tool that executes external commands or accesses the filesystem MUST go through `SecurityManager.validate_tool_execution()`.

### Git Conventions

- Write descriptive commit messages: `component: brief description`
- Keep commits atomic — one logical change per commit
- Run `ruff check` before committing

---

## Key Reference Files

| File | What it covers |
|------|----------------|
| `docs/engineering-standards.md` | Full harness engineering standards |
| `docs/ai-agent-architecture.md` | ARK + LangGraph architecture deep-dive |
| `docs/creating-new-capabilities.md` | Adding MCP tools and servers |
| `docs/prompt_engineering_insights.md` | Prompt design and management |
| `docs/lessons_learned.md` | Historical decisions and learnings |
| `config/llm_config.yaml` | LLM provider and model configuration |
| `pyproject.toml` | Build, test, and lint configuration |
