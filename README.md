# Project Jarvis

An AI agent framework for building intelligent assistants.

## Overview

Project Jarvis is a Python-based AI agent framework designed to create intelligent, conversational assistants. This project provides a foundation for building AI agents with extensible capabilities.

## 📂 Project Structure

```
Jarvis/
├── src/
│   ├── core/               # Core framework components
│   │   ├── agent/          # Agent implementations (ReAct)
│   │   ├── ark/            # Autonomous Reasoning Kernel (LangGraph)
│   │   ├── config/         # Configuration management
│   │   ├── context/        # Context and memory management
│   │   ├── llm/            # LLM providers and utilities
│   │   ├── mcp/            # Model Context Protocol client/registry
│   │   ├── prompt/         # Prompt engineering and management
│   │   └── security/       # Security and validation
│   ├── web/                # Web interface (Chainlit)
│   ├── jarvis_agent.py     # Main agent entry point
│   └── main.py             # CLI entry point
├── config/                 # Configuration files
├── examples/               # Usage examples and demos
├── tests/                  # Comprehensive test suite
├── docs/                   # Documentation
├── .chainlit/              # Chainlit configuration
└── pyproject.toml          # Project dependencies and metadata
```

## 🚀 Getting Started

### Prerequisites

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) (recommended for package management)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd 00_Jarvis
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Configure environment:
   Ensure you have the necessary API keys set up (e.g., `OPENAI_API_KEY`). You can refer to `config/` directory for configuration examples.

## 📖 Usage

### Running the Web Interface

Start the Chainlit-based web UI:

```bash
uv run chainlit run src/web/app.py -w
```

### Running the CLI Agent

Start the agent in command-line mode:

```bash
uv run python src/main.py
```

### Running Examples

Explore the capabilities with provided examples:

```bash
# Complete demo (Recommended)
uv run python examples/complete_demo.py

# Conversation demo
uv run python examples/conversation_demo.py

# LangGraph example
uv run python examples/langgraph_example.py
```

## 🛠 Development

### Running Tests

Execute the test suite using pytest:

```bash
uv run pytest
```

With coverage report:

```bash
uv run pytest --cov=src tests/
```

### Adding New Features

1. Add new modules in the `src/` directory
2. Write corresponding tests in the `tests/` directory
3. Update documentation in the `docs/` directory
4. Add any new dependencies:
```bash
uv add <package_name>
```

### Testing

- Use `pytest` for testing
- Write tests for all new functionality
- Maintain test coverage above 80%

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Document functions and classes with docstrings

## Architecture

The project follows a modular architecture:

- **Core Agent**: `JarvisAgent` class handles main conversation loop
- **Logging**: Structured logging for debugging and monitoring
- **Extensible**: Easy to add new capabilities and integrations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

[Add your license here]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
