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
        # "thought": "Thinking", "type": "answer", "content": "Some text"
        # We simulate partial JSON stream
        yield '{"thought": "Thinking", "type": "answer", "content": "Some '
        yield 'text"}'

    agent.llm_manager.stream_response.side_effect = mock_stream

    callbacks = {
        "on_token": MagicMock(),
        "on_thought_token": MagicMock(),
    }

    await agent._generate_and_stream_response([], callbacks)

    # The current parser logic does NOT emit tokens for JSON string values until the closing quote is found
    # to handle escapes correctly.
    # So "Some " is buffered, and then combined with "text" and finally emitted as "Some text"
    # Wait, the parser logic:
    # 1. Finds end of string (unescaped quote).
    # 2. If found, emits content.
    # 3. If NOT found, it emits "safe" part (buffer - 10 chars).

    # "Some " is length 5. Safe buffer is 10. So it won't emit yet.
    # Then "text"}" arrives. Buffer is "Some text"}".
    # It finds the quote. Emits "Some text".

    callbacks["on_token"].assert_any_call("Some text")


@pytest.mark.asyncio
async def test_streaming_missing_flush_in_thought():
    """Test that remaining buffer content is flushed correctly inside thought."""
    agent = ReActAgent()
    agent.llm_manager = MagicMock()

    # Mock stream response
    async def mock_stream(*args, **kwargs):
        yield '{"thought": "Thinking'
        yield '..."'  # inside thought

    agent.llm_manager.stream_response.side_effect = mock_stream

    callbacks = {
        "on_thought_start": MagicMock(),
        "on_thought_token": MagicMock(),
        "on_thought_end": MagicMock(),
        "on_token": MagicMock(),
    }

    await agent._generate_and_stream_response([], callbacks)

    callbacks["on_thought_start"].assert_called_once()
    # Same logic: "Thinking" (8 chars) < 10 chars buffer. Won't emit.
    # Then "..."" arrives. Finds quote. Emits "Thinking..."

    callbacks["on_thought_token"].assert_any_call("Thinking...")
    callbacks["on_thought_end"].assert_called_once()
