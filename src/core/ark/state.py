from typing import TypedDict, List, Dict, Any, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class JarvisState(TypedDict):
    """
    The shared state for the Jarvis Agent graph.
    """

    # Messages history, using add_messages reducer to append new messages
    messages: Annotated[List[BaseMessage], add_messages]

    # The original user input for the current turn
    user_input: str

    # Current todo list state (synced with ARK engine)
    todo_list: List[Dict[str, Any]]

    # Available tools definitions (for prompt injection)
    available_tools: Dict[str, Any]

    # Intermediate reasoning steps or context
    scratchpad: Dict[str, Any]

    # The last node that executed (useful for routing)
    sender: str


class MultiAgentState(JarvisState):
    """
    Extended state for Multi-Agent Supervisor System.
    """

    # The next agent to route to
    next: str

    # Shared structured data for inter-agent protocol
    structured_data: Dict[str, Any]
