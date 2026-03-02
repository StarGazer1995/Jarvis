import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage, HumanMessage

from src.core.ark.graph import create_ark_graph, create_supervisor_graph
from src.core.ark.state import JarvisState, MultiAgentState
from src.core.ark.utils import AgentSpec, create_agent_node
from src.core.llm.client import LLMManager, LLMResponse
from src.core.mcp.client import ARKMCPClient


@pytest.mark.asyncio
async def create_stream_response(content: str):
    """Helper to create an async generator that yields the content in chunks."""
    chunk_size = 5
    for i in range(0, len(content), chunk_size):
        yield content[i : i + chunk_size]
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_ark_graph_basic_flow():
    # Mock Dependencies
    mock_llm = MagicMock(spec=LLMManager)
    # mock_llm.generate_response = AsyncMock() # Removed
    mock_llm.stream_response = MagicMock()  # Changed to stream_response

    mock_mcp = MagicMock(spec=ARKMCPClient)
    mock_mcp.execute_tool = AsyncMock()

    # 1. Simple Q&A
    # mock_llm.generate_response.return_value = LLMResponse(content="Hello World") # Removed
    # Updated to JSON protocol
    mock_llm.stream_response.return_value = create_stream_response(
        '{"thought": "Hi", "type": "answer", "content": "Hello World"}'
    )

    graph = create_ark_graph(mock_llm, mock_mcp)

    initial_state = {
        "messages": [HumanMessage(content="Hi")],
        "user_input": "Hi",
        "todo_list": [],
        "scratchpad": {},
        "sender": "user",
    }

    result = await graph.ainvoke(initial_state)

    # Messages: [Human, AI]
    assert len(result["messages"]) == 2
    # The content includes thought + content string
    assert "Hello World" in result["messages"][-1].content
    assert result["sender"] == "master"


@pytest.mark.asyncio
async def test_ark_graph_tool_execution():
    # Mock Dependencies
    mock_llm = MagicMock(spec=LLMManager)
    # mock_llm.generate_response = AsyncMock() # Removed
    mock_llm.stream_response = MagicMock()  # Changed

    mock_mcp = MagicMock(spec=ARKMCPClient)
    mock_mcp.execute_tool = AsyncMock(return_value="Tool Result Success")

    # Scenario: User -> LLM (Tool Call) -> Tool -> LLM (Final Answer)

    # First LLM call returns a Tool Call (JSON Protocol)
    content1 = '{"thought": "Need to use tool.", "type": "tool_call", "content": {"name": "test_tool", "arguments": {"arg": 1}}}'
    # Second LLM call returns Final Answer (JSON Protocol)
    content2 = '{"thought": "Done.", "type": "answer", "content": "Done."}'

    # mock_llm.generate_response.side_effect = [response1, response2] # Removed
    mock_llm.stream_response.side_effect = [
        create_stream_response(content1),
        create_stream_response(content2),
    ]  # Changed

    graph = create_ark_graph(mock_llm, mock_mcp)

    initial_state = {
        "messages": [HumanMessage(content="Run tool")],
        "user_input": "Run tool",
        "todo_list": [],
        "scratchpad": {},
        "sender": "user",
    }

    # Use recursion_limit to prevent infinite loops if logic is wrong
    result = await graph.ainvoke(initial_state, {"recursion_limit": 10})

    # Verify Flow
    # Expected: [Human, AI(ToolCall), Tool, AI(Final)]

    messages = result["messages"]
    assert len(messages) == 4

    # Check Tool Call Message
    assert isinstance(messages[1], AIMessage)
    # Note: Depending on how MasterNode parses the stream, tool_calls might be populated
    # If MasterNode aggregates the stream and then parses tool calls, this should work.
    # Assuming MasterNode implementation handles streaming content correctly.
    assert messages[1].tool_calls[0]["name"] == "test_tool"

    # Check Tool Result Message
    # Note: ToolMessage content is string
    # In LangGraph/ARK, tool output is usually ToolMessage, but here we check content.
    # ARK graph node 'tools' likely returns ToolMessage or updates state.
    # If using 'prebuilt' ToolNode, it returns ToolMessage.
    # Let's inspect the message type if needed, but content check is key.
    assert messages[2].content == "Tool Result Success"
    assert messages[2].name == "test_tool"

    # Check Final Answer
    assert "Done." in messages[3].content


@pytest.mark.asyncio
async def test_supervisor_flow():
    # Mock Dependencies
    mock_llm = MagicMock(spec=LLMManager)
    # mock_llm.generate_response = AsyncMock() # Removed
    mock_llm.stream_response = MagicMock()  # Changed

    # Define a Worker Agent
    async def worker_func(state: MultiAgentState):
        return {"content": "Worker Task Done", "data": {"status": "complete"}}

    worker_node = create_agent_node("WorkerA", worker_func)
    agent_spec = AgentSpec(name="WorkerA", description="Does work", node=worker_node)

    # Scenario: User -> Supervisor (Delegate to WorkerA) -> WorkerA -> Supervisor (Final Answer)

    # 1. Supervisor delegates to WorkerA (using tool-call syntax for routing)
    # MasterNode parses "Action: WorkerA" as a tool call (JSON Protocol)
    content1 = '{"thought": "Delegate to WorkerA.", "type": "tool_call", "content": {"name": "WorkerA", "arguments": {}}}'

    # 2. Supervisor receives WorkerA output and finishes (JSON Protocol)
    content2 = (
        '{"thought": "Task Complete.", "type": "answer", "content": "Task Complete."}'
    )

    # mock_llm.generate_response.side_effect = [response1, response2] # Removed
    mock_llm.stream_response.side_effect = [
        create_stream_response(content1),
        create_stream_response(content2),
    ]  # Changed

    graph = create_supervisor_graph(mock_llm, [agent_spec])

    initial_state = {
        "messages": [HumanMessage(content="Do work")],
        "user_input": "Do work",
        "todo_list": [],
        "scratchpad": {},
        "structured_data": {},
        "sender": "user",
    }

    result = await graph.ainvoke(initial_state, {"recursion_limit": 10})

    messages = result["messages"]
    # Expected: [Human, AI(Delegate), WorkerOutput, AI(Final)]

    assert len(messages) == 4

    # Check Delegation (AI Message with Tool Call)
    assert isinstance(messages[1], AIMessage)
    assert messages[1].tool_calls[0]["name"] == "WorkerA"

    # Check Worker Output
    # Worker output is a HumanMessage with name=WorkerA
    assert isinstance(messages[2], HumanMessage)
    assert messages[2].name == "WorkerA"
    assert messages[2].content == "Worker Task Done"

    # Check Final Answer
    assert "Task Complete." in messages[3].content
