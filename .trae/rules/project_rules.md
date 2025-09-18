# Instructions for This Project

## Overview
This repository contains a Python-based AI agent framework called "Project Jarvis". The project provides a foundation for building intelligent, conversational assistants with extensible capabilities.

## Project Structure
```
00_Jarvis/
├── src/                    # Source code
│   ├── __init__.py        # Package initialization
│   └── main.py            # Main entry point and core agent logic
├── tests/                 # Test files using pytest
│   └── test_main.py       # Tests for main functionality
├── docs/                  # Documentation
│   └── README.md          # Detailed documentation
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Python project configuration
├── .github/              # GitHub workflows and configurations
├── .gitignore            # Git ignore rules
└── README.md             # Main project documentation
```

## Guidance for AI Coding Agents

### Development Workflow
- **Source Code**: Place all Python modules in `src/` directory
- **Testing**: Use `pytest` for tests in `tests/` directory - run with `pytest` or `pytest --cov=src tests/` for coverage
- **Documentation**: Update `docs/README.md` for API documentation, main `README.md` for project overview
- **Dependencies**: Add new Python packages to `requirements.txt`
- **Entry Point**: Run the agent with `python src/main.py` (supports `--log-level` argument)

### Code Conventions
- Use python3.12 for development
- Follow PEP 8 Python style guidelines
- Use type hints where appropriate
- Document all functions and classes with docstrings
- Maintain test coverage to 100%
- Use structured logging via the built-in logging configuration

### Architecture Patterns
- **Core Agent**: `JarvisAgent` class handles main conversation loop and agent lifecycle
- **Extensible Design**: Easy to add new capabilities by extending the agent or creating new modules
- **Logging**: Centralized logging configuration with configurable levels
- **Testing**: Comprehensive test coverage using pytest with mocking for external dependencies

### Adding New Features
1. Create new modules in `src/` directory
2. Extend `JarvisAgent` class or create new capability classes
3. Write comprehensive tests in `tests/` directory
4. Update documentation in `docs/README.md`
5. Add any new dependencies to `requirements.txt`
6. Consider adding command-line arguments if needed

### External Integrations
- Currently minimal dependencies (pytest for testing, typing-extensions for compatibility)
- When adding new integrations, document their purpose and configuration requirements
- Consider creating separate capability modules for different integrations (e.g., weather, calendar, etc.)

## Development Commands
- **Run Agent**: `python src/main.py [--log-level DEBUG|INFO|WARNING|ERROR]`
- **Run Tests**: `pytest` or `pytest --cov=src tests/` for coverage
- **Install Dependencies**: `pip install -r requirements.txt`
- **Package Management**: Force use `uv` for package management

## Documentations
- Update `docs/README.md` for detailed API and usage instructions
- Maintain the main `README.md` for project overview and quick start guide
- Each discussion should be documented under `docs/` for clarity

---
_Last updated: 2025-09-18_

---
_Last updated: 2025-09-18_
