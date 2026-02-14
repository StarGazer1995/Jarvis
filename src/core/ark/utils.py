import re
import inspect
from dataclasses import dataclass
from typing import Callable, Dict, Any, Union
from langchain_core.messages import BaseMessage, HumanMessage
from src.core.ark.state import MultiAgentState

@dataclass
class AgentSpec:
    """
    Specification for a Worker Agent in the Supervisor system.
    """
    name: str
    description: str
    node: Callable[[MultiAgentState], Dict[str, Any]]

def create_agent_node(agent_name: str, agent_func: Callable[[MultiAgentState], Union[Dict, str, BaseMessage]]):
    """
    Helper to wrap a simple function into a Protocol-Compliant Node.
    Ensures the output is correctly formatted as a message tagged with the agent's name.
    """
    async def agent_node(state: MultiAgentState) -> Dict[str, Any]:
        result = agent_func(state)
        if inspect.isawaitable(result):
            result = await result
        
        # Normalize result to dict
        if isinstance(result, str):
            content = result
            data = {}
        elif isinstance(result, BaseMessage):
            content = result.content
            data = {} # Message attributes could be data, but let's keep it simple
        elif isinstance(result, dict):
            # Check if it's already a state update
            if "messages" in result:
                # It's likely a state update, but we need to ensure the message has the name
                if "content" in result:
                    content = result["content"]
                    data = result.get("data", {})
                else:
                    # Assume it's a full state update, just return it
                    # But we MUST ensure sender is set
                    result["sender"] = agent_name
                    return result
            else:
                # It's a data dict, maybe?
                if "content" in result:
                    content = result["content"]
                    data = result.get("data", {})
                else:
                    # Fallback
                    content = str(result)
                    data = {}
        else:
            content = str(result)
            data = {}

        # Create the message
        message = HumanMessage(content=content, name=agent_name)
        
        return {
            "messages": [message],
            "sender": agent_name,
            "structured_data": data # This overwrites. For merging, we'd need a reducer.
        }
    return agent_node

def clean_llm_response(text: str) -> str:
    """
    Clean the LLM response to extract the final decision.
    Removes <think> blocks and looks for the final answer.
    """
    # Remove <think>...</think> blocks
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Remove Markdown code blocks if any
    cleaned = re.sub(r'```.*?```', '', cleaned, flags=re.DOTALL)
    
    # Strip whitespace
    cleaned = cleaned.strip()
    
    # If multiple lines, take the last non-empty line
    lines = [l.strip() for l in cleaned.split('\n') if l.strip()]
    if lines:
        return lines[-1]
    return cleaned
