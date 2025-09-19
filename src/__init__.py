"""
Project Jarvis - AI Agent Framework

This package provides the foundation for building intelligent AI agents
powered by the ARK (Agent Reactor Kernel) engine with MCP integration.
"""

__version__ = "0.1.0"
__author__ = "Project Jarvis Team"

from .jarvis_agent import JarvisAgent
from .core.ark_engine import ARKEngine

__all__ = ["JarvisAgent", "ARKEngine"]