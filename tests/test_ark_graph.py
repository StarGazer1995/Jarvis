import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage, HumanMessage

from src.core.ark.graph import create_ark_graph
from src.core.ark.state import JarvisState
from src.core.llm.client import LLMManager, LLMResponse
from src.core.mcp.client import ARKMCPClient

@pytest.mark.asyncio
async def test_ark_graph_basic_flow():
    # Mock Dependencies
    mock_llm = MagicMock(spec=LLMManager)
    mock_llm.generate_response = AsyncMock()
    
    mock_mcp = MagicMock(spec=ARKMCPClient)
    mock_mcp.execute_tool = AsyncMock()
    
    # 1. Simple Q&A
    mock_llm.generate_response.return_value = LLMResponse(content="Hello World")
    
    graph = create_ark_graph(mock_llm, mock_mcp)
    
    initial_state = {
        "messages": [HumanMessage(content="Hi")],
        "user_input": "Hi",
        "todo_list": [],
        "scratchpad": {},
        "sender": "user"
    }
    
    result = await graph.ainvoke(initial_state)
    
    # Messages: [Human, AI]
    assert len(result["messages"]) == 2
    assert result["messages"][-1].content == "Hello World"
    assert result["sender"] == "master"

@pytest.mark.asyncio
async def test_ark_graph_tool_execution():
    # Mock Dependencies
    mock_llm = MagicMock(spec=LLMManager)
    mock_llm.generate_response = AsyncMock()
    
    mock_mcp = MagicMock(spec=ARKMCPClient)
    mock_mcp.execute_tool = AsyncMock(return_value="Tool Result Success")
    
    # Scenario: User -> LLM (Tool Call) -> Tool -> LLM (Final Answer)
    
    # First LLM call returns a Tool Call
    response1 = LLMResponse(content="Thought: Need to use tool.\nAction: test_tool\nAction Input: {\"arg\": 1}")
    # Second LLM call returns Final Answer
    response2 = LLMResponse(content="Final Answer: Done.")
    
    mock_llm.generate_response.side_effect = [response1, response2]
    
    graph = create_ark_graph(mock_llm, mock_mcp)
    
    initial_state = {
        "messages": [HumanMessage(content="Run tool")],
        "user_input": "Run tool",
        "todo_list": [],
        "scratchpad": {},
        "sender": "user"
    }
    
    # Use recursion_limit to prevent infinite loops if logic is wrong
    result = await graph.ainvoke(initial_state, {"recursion_limit": 10})
    
    # Verify Flow
    # Expected: [Human, AI(ToolCall), Tool, AI(Final)]
    
    messages = result["messages"]
    assert len(messages) == 4
    
    # Check Tool Call Message
    assert isinstance(messages[1], AIMessage)
    assert messages[1].tool_calls[0]["name"] == "test_tool"
    
    # Check Tool Result Message
    # Note: ToolMessage content is string
    assert messages[2].content == "Tool Result Success"
    assert messages[2].name == "test_tool"
    
    # Check Final Answer
    assert messages[3].content == "Final Answer: Done."
