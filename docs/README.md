# Project Jarvis Documentation

## Getting Started

Welcome to Project Jarvis! This documentation will help you understand and extend the AI agent framework.

> **New to the project? Start with [`AGENTS.md`](../AGENTS.md) at the project root for agent instructions and quick-start standards, then read [`engineering-standards.md`](engineering-standards.md) for the full harness engineering framework.**

## Document Index

| Document | Description |
|----------|-------------|
| [`engineering-standards.md`](engineering-standards.md) | **Harness Engineering Standards** — The six-pillar framework for building reliable, observable, and testable AI systems |
| [`ai-agent-architecture.md`](ai-agent-architecture.md) | Architecture deep-dive: ARK engine, LangGraph orchestration, MCP integration patterns |
| [`prompt_engineering_insights.md`](prompt_engineering_insights.md) | Evolution of prompt strategies (XML → JSON), centralized prompt management, and lessons learned |
| [`prompt_system_comparison.md`](prompt_system_comparison.md) | Comparison of Jarvis prompt system vs LangChain prompt system (Chinese) |
| [`creating-new-capabilities.md`](creating-new-capabilities.md) | Guide for adding new MCP tools and capabilities |
| [`lessons_learned.md`](lessons_learned.md) | Historical architectural decisions and key takeaways |
| [`ark_evolution.mmd`](ark_evolution.mmd) | Mermaid diagram of ARK engine evolution |
| [`../AGENTS.md`](../AGENTS.md) | **Project-wide Agent Instructions** — VS Code Copilot agent guidelines for this workspace |

## Core Components

### JarvisAgent Class

The `JarvisAgent` class is the heart of the framework. It provides:

- **Initialization**: Set up the agent with a custom name
- **Main Loop**: Interactive conversation handling
- **Input Processing**: Extensible input/output processing
- **Logging**: Structured logging for debugging

### Main Entry Point

The `main.py` file provides:

- Command-line argument parsing
- Logging configuration
- Agent lifecycle management

## Quick Reference: Harness Engineering

This project follows **harness engineering** principles. The six pillars are:

1. **Test Harness** — Mocks for every dependency, fault injection testing, 100% diff coverage for newly added or modified lines, and no coverage-padding test cases
2. **Integration Harness** — MCP-first tool integration, abstraction layers, graceful degradation
3. **Configuration Harness** — Three-tier config (YAML + env vars + runtime overrides)
4. **Security Harness** — Tool validation, rate limiting, sandboxed execution, audit logging
5. **Observability Harness** — Structured logging, metrics collection, usage tracking
6. **Orchestration Harness** — LangGraph state machine, explicit routing, bounded iterations

See [`engineering-standards.md`](engineering-standards.md) for the full standards document.

## Extending the Framework

### Adding New Capabilities

To add new capabilities to your Jarvis agent:

1. Create new modules in the `src/` directory
2. Extend the `JarvisAgent` class or create new components
3. Add corresponding tests (with mocks from the root `conftest.py`)
4. Update this documentation

### Example Extension

```python
# src/capabilities/weather.py
class WeatherCapability:
    def get_weather(self, location: str) -> str:
        # Implementation here
        return f"Weather in {location}: Sunny"

# In src/main.py, extend the agent:
class EnhancedJarvisAgent(JarvisAgent):
    def __init__(self, name: str = "Jarvis"):
        super().__init__(name)
        self.weather = WeatherCapability()
    
    def process_input(self, user_input: str) -> None:
        if "weather" in user_input.lower():
            # Handle weather requests
            pass
        else:
            super().process_input(user_input)
```

## Testing Guidelines

All testing follows the **Test Harness** standard:

- **Every external dependency MUST have a mock** — use `MockLLMClient`, `MockMCPClient`, etc. from the root `conftest.py`
- **Test both success AND failure paths** — include rate limits, timeouts, malformed responses
- **Write fault injection tests** — verify retry logic handles partial failures correctly
- **Write descriptive test names** that document the scenario being tested
- **Maintain 100% diff coverage** on all newly added or modified lines in the current change set
- **Do not add contrived coverage-padding tests** — every test must validate a meaningful behavior, contract, edge case, or failure mode

See [`engineering-standards.md`](engineering-standards.md#21-test-harness) for detailed test harness standards.
