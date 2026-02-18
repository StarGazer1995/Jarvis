
import pytest
from unittest.mock import MagicMock
from src.core.agent.react import ReActAgent
from src.core.llm.client import LLMMessage

@pytest.mark.asyncio
async def test_streaming_missing_flush():
    """Test that remaining buffer content is flushed correctly."""
    agent = ReActAgent()
    agent.llm_manager = MagicMock()
    
    # Mock stream response with partial content at the end
    async def mock_stream(*args, **kwargs):
        yield "Some text"
        yield " <"  # Partial tag start
    
    agent.llm_manager.stream_response.side_effect = mock_stream
    
    callbacks = {
        "on_token": MagicMock(),
        "on_thought_token": MagicMock(),
    }
    
    await agent._generate_and_stream_response([], callbacks)
    
    # Should be called for "Some text"
    callbacks["on_token"].assert_any_call("Some text")
    
    # The parser splits " <" into " " and "<" because it detects the start of a potential tag
    # So we should verify that both parts are flushed
    callbacks["on_token"].assert_any_call(" ")
    callbacks["on_token"].assert_any_call("<")

@pytest.mark.asyncio
async def test_streaming_missing_flush_in_thought():
    """Test that remaining buffer content is flushed correctly inside thought."""
    agent = ReActAgent()
    agent.llm_manager = MagicMock()
    
    # Mock stream response
    async def mock_stream(*args, **kwargs):
        yield "<thought>Thinking"
        yield "..."  # inside thought
    
    agent.llm_manager.stream_response.side_effect = mock_stream
    
    callbacks = {
        "on_thought_start": MagicMock(),
        "on_thought_token": MagicMock(),
        "on_thought_end": MagicMock(),
        "on_token": MagicMock()
    }
    
    await agent._generate_and_stream_response([], callbacks)
    
    callbacks["on_thought_start"].assert_called_once()
    callbacks["on_thought_token"].assert_any_call("Thinking")
    callbacks["on_thought_token"].assert_any_call("...")  # Should be flushed
    # on_thought_end NOT called because stream ended abruptly
    callbacks["on_thought_end"].assert_not_called()
