"""
Agent Framework Types
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any


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
    action: Optional[str] = None
    action_input: Optional[Union[Dict[str, Any], str]] = None
    observation: Optional[str] = None
