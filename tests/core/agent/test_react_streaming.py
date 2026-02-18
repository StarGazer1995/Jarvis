
import pytest
from unittest.mock import MagicMock, AsyncMock
from src.core.agent.react import ReActAgent
from src.core.llm.client import LLMMessage

@pytest.mark.asyncio
async def test_streaming_with_thought_tags():
    """Test streaming with standard <thought> tags."""
    agent = ReActAgent()
    agent.llm_manager = MagicMock()
    
    # Mock stream response
    async def mock_stream(*args, **kwargs):
        yield "<thought>"
        yield "Thinking process..."
        yield "</thought>"
        yield "Final answer."
    
    agent.llm_manager.stream_response.side_effect = mock_stream
    
    callbacks = {
        "on_thought_start": MagicMock(),
        "on_thought_token": MagicMock(),
        "on_thought_end": MagicMock(),
        "on_token": MagicMock()
    }
    
    response = await agent._generate_and_stream_response([], callbacks)
    
    assert response == "<thought>Thinking process...</thought>Final answer."
    callbacks["on_thought_start"].assert_called_once()
    callbacks["on_thought_token"].assert_any_call("Thinking process...")
    callbacks["on_thought_end"].assert_called_once()
    callbacks["on_token"].assert_any_call("Final answer.")

@pytest.mark.asyncio
async def test_streaming_with_think_tags():
    """Test streaming with alternative <think> tags."""
    agent = ReActAgent()
    agent.llm_manager = MagicMock()
    
    # Mock stream response
    async def mock_stream(*args, **kwargs):
        yield "<think>"
        yield "Deep thinking..."
        yield "</think>"
        yield "Result."
    
    agent.llm_manager.stream_response.side_effect = mock_stream
    
    callbacks = {
        "on_thought_start": MagicMock(),
        "on_thought_token": MagicMock(),
        "on_thought_end": MagicMock(),
        "on_token": MagicMock()
    }
    
    response = await agent._generate_and_stream_response([], callbacks)
    
    assert response == "<think>Deep thinking...</think>Result."
    callbacks["on_thought_start"].assert_called_once()
    callbacks["on_thought_token"].assert_any_call("Deep thinking...")
    callbacks["on_thought_end"].assert_called_once()
    callbacks["on_token"].assert_any_call("Result.")

@pytest.mark.asyncio
async def test_streaming_mixed_tags():
    """Test streaming with mixed content and tags."""
    agent = ReActAgent()
    agent.llm_manager = MagicMock()
    
    async def mock_stream(*args, **kwargs):
        yield "Prefix "
        yield "<think>"
        yield "Thought"
        yield "</think>"
        yield " Suffix"
    
    agent.llm_manager.stream_response.side_effect = mock_stream
    
    callbacks = {
        "on_thought_start": MagicMock(),
        "on_thought_token": MagicMock(),
        "on_thought_end": MagicMock(),
        "on_token": MagicMock()
    }
    
    await agent._generate_and_stream_response([], callbacks)
    
    callbacks["on_token"].assert_any_call("Prefix ")
    callbacks["on_thought_start"].assert_called_once()
    callbacks["on_thought_token"].assert_any_call("Thought")
    callbacks["on_thought_end"].assert_called_once()
    callbacks["on_token"].assert_any_call(" Suffix")
