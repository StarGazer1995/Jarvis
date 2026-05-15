from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class JarvisState(TypedDict):
    """
    The shared state for the Jarvis Agent graph.
    """

    # Messages history, using add_messages reducer to append new messages
    messages: Annotated[list[BaseMessage], add_messages]

    # The original user input for the current turn
    user_input: str

    # Current todo list state (synced with ARK engine)
    todo_list: list[dict[str, Any]]

    # Available tools definitions (for prompt injection)
    available_tools: dict[str, Any]

    # Intermediate reasoning steps or context
    scratchpad: dict[str, Any]

    # The last node that executed (useful for routing)
    sender: str


class MultiAgentState(JarvisState):
    """
    Extended state for Multi-Agent Supervisor System.
    """

    # The next agent to route to
    next: str

    # Shared structured data for inter-agent protocol
    structured_data: dict[str, Any]
