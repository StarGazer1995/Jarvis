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

1. **Every external dependency MUST have a mock counterpart** in `conftest.py` — this is non-negotiable for test harness integrity.
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

# Run only diff-related tests locally
bash scripts/run_diff_tests.sh

# Run diff-related tests with 100% diff coverage
bash scripts/run_diff_tests.sh --coverage

# Run specific test file
uv run pytest src/core/ark/test/test_engine.py -v

# Run linter
uv run ruff check .
```

### Test Harness Requirements

Every agent MUST adhere to these standards when writing or modifying code:

1. **Tests are co-located** with source code under `src/**/test/` directories. Write tests alongside the module being tested (e.g., `src/core/ark/nodes/tools.py` → `src/core/ark/nodes/test/test_tools.py`).
2. **Every new feature MUST include tests** — unit tests for logic, integration tests for MCP/LLM boundaries.
3. **Use the existing mock fixtures** (`MockLLMClient`, `MockMCPClient`, `MockContextManager`) from the root `conftest.py`. Never hit real APIs in unit tests.
4. **All async tests must use `pytest-asyncio`** with `asyncio_mode = "auto"` (already configured).
5. **Test both success and failure paths** — include tests for rate limits, timeouts, malformed responses.
6. **Coverage target**: Effective immediately, all newly added or modified lines in the current diff must reach **100% diff coverage**.
7. **No coverage gaming**: Do not add contrived, implementation-shaped, or behavior-free test cases solely to raise coverage numbers. Every new test must validate a meaningful runtime behavior, contract, edge case, or failure mode.

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

### Harness Engineering Standards

When adding or modifying any component, follow these five standards (see `docs/engineering-standards.md` for the full six-pillar framework):

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
| `docs/engineering-standards.md` | Full harness engineering standards (six pillars) |
| `docs/ai-agent-architecture.md` | ARK + LangGraph architecture deep-dive |
| `docs/creating-new-capabilities.md` | Adding MCP tools and servers |
| `docs/prompt_engineering_insights.md` | Prompt design and management |
| `docs/lessons_learned.md` | Historical decisions and learnings |
| `config/llm_config.yaml` | LLM provider and model configuration |
| `pyproject.toml` | Build, test, and lint configuration |
| `conftest.py` | Shared test fixtures and mocks |
| `scripts/run_diff_tests.sh` | Run diff-targeted tests with coverage |
