"""
Agent Framework Types
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class AgentState(Enum):
    """Operational states for an agent."""

    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    TOOL_EXECUTION = "tool_execution"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class AgentStep:
    """Represents a single step in an agent's reasoning loop."""

    thought: str
    action: str | None = None
    action_input: dict[str, Any] | str | None = None
    observation: str | None = None
    observation_data: Any | None = None
