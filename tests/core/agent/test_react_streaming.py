import pytest
from unittest.mock import MagicMock, AsyncMock
from src.core.agent.react import ReActAgent
from src.core.llm.client import LLMMessage


@pytest.mark.asyncio
async def test_streaming_with_thought_tags():
    """Test streaming with standard JSON thought field."""
    agent = ReActAgent()
    agent.llm_manager = MagicMock()

    # Mock stream response
    async def mock_stream(*args, **kwargs):
        yield '{"thought": "Thinking process...", "type": "answer", "content": "Final answer."}'

    agent.llm_manager.stream_response.side_effect = mock_stream

    callbacks = {
        "on_thought_start": MagicMock(),
        "on_thought_token": MagicMock(),
        "on_thought_end": MagicMock(),
        "on_token": MagicMock(),
    }

    response = await agent._generate_and_stream_response([], callbacks)

    assert "Thinking process..." in response
    assert "Final answer." in response
    callbacks["on_thought_start"].assert_called_once()
    callbacks["on_thought_token"].assert_any_call("Thinking process...")
    callbacks["on_thought_end"].assert_called_once()
    callbacks["on_token"].assert_any_call("Final answer.")


@pytest.mark.asyncio
async def test_streaming_with_think_tags():
    """Test streaming with alternative JSON thought field (same logic)."""
    # In JSON protocol, we only use "thought" field, so this test is redundant
    # but we can adapt it to test split tokens.
    agent = ReActAgent()
    agent.llm_manager = MagicMock()

    # Mock stream response
    async def mock_stream(*args, **kwargs):
        # "Deep " (5 chars) < 10. Buffer.
        yield '{"thought": "Deep '
        # "thinking..." (11 chars). Buffer is 16 chars.
        # Safe len = 16 - 10 = 6.
        # Emits "Deep t". Buffer becomes "hinking...".
        # Then closing quote comes later? No, wait.
        yield 'thinking...", "type": "answer", "content": "Result."}'

    agent.llm_manager.stream_response.side_effect = mock_stream

    callbacks = {
        "on_thought_start": MagicMock(),
        "on_thought_token": MagicMock(),
        "on_thought_end": MagicMock(),
        "on_token": MagicMock(),
    }

    response = await agent._generate_and_stream_response([], callbacks)

    assert "Deep thinking..." in response
    callbacks["on_thought_start"].assert_called_once()

    # "Deep " (len 5). Buffer "Deep ". Safe 0.
    # "thinking...", ... (len big). Buffer "Deep thinking...", ...".
    # Finds quote. Emits "Deep thinking...".

    callbacks["on_thought_token"].assert_any_call("Deep thinking...")
    callbacks["on_thought_end"].assert_called_once()
    callbacks["on_token"].assert_any_call("Result.")


@pytest.mark.asyncio
async def test_streaming_mixed_tags():
    """Test streaming with mixed content - Not applicable in JSON strictly, but testing robust parsing."""
    # JSON parser expects strict structure. "Prefix " before JSON is invalid unless handled.
    # Our parser handles markdown blocks, but maybe not random text prefix.
    # Let's test standard JSON streaming behavior instead.
    agent = ReActAgent()
    agent.llm_manager = MagicMock()

    async def mock_stream(*args, **kwargs):
        yield '{"thought": "Thought", "type": "answer", "content": "Suffix"}'

    agent.llm_manager.stream_response.side_effect = mock_stream

    callbacks = {
        "on_thought_start": MagicMock(),
        "on_thought_token": MagicMock(),
        "on_thought_end": MagicMock(),
        "on_token": MagicMock(),
    }

    await agent._generate_and_stream_response([], callbacks)

    callbacks["on_thought_start"].assert_called_once()
    callbacks["on_thought_token"].assert_any_call("Thought")
    callbacks["on_thought_end"].assert_called_once()
    callbacks["on_token"].assert_any_call("Suffix")
