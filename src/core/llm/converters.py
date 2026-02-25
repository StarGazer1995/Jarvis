from typing import List, Any
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage
)
from src.core.llm.types import LLMMessage

def convert_langchain_to_llm_messages(messages: List[BaseMessage]) -> List[LLMMessage]:
    """
    Convert LangChain messages to internal LLMMessage format.
    
    Args:
        messages: List of LangChain BaseMessage objects
        
    Returns:
        List of LLMMessage objects
    """
    out = []
    for m in messages:
        role = "user"
        content = m.content
        metadata = {}
        
        # Copy additional_kwargs to metadata
        if hasattr(m, "additional_kwargs") and m.additional_kwargs:
            metadata.update(m.additional_kwargs)
            
        if isinstance(m, SystemMessage):
            role = "system"
        elif isinstance(m, AIMessage):
            role = "assistant"
            # Handle tool_calls in AIMessage if present
            if hasattr(m, "tool_calls") and m.tool_calls:
                metadata["tool_calls"] = m.tool_calls
        elif isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, ToolMessage):
            # Treat ToolMessage as User message saying "Observation: ..." for ReAct style
            # Or just pass it as tool role if the LLM provider supports it.
            # However, LLMMessage usually supports system, user, assistant.
            # Current Jarvis implementation treats it as user observation.
            role = "user"
            content = f"Observation: {content}"
            if m.name:
                metadata["tool_name"] = m.name
            if m.tool_call_id:
                metadata["tool_call_id"] = m.tool_call_id
        
        # Handle generic messages with specific types or names
        if m.name:
            metadata["name"] = m.name
            
        out.append(LLMMessage(role=role, content=str(content), metadata=metadata))
            
    return out
