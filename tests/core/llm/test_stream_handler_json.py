import pytest
import asyncio
from unittest.mock import Mock, call
from src.core.llm.stream_handler import StreamTokenHandler

async def mock_stream(text, chunk_size=3):
    for i in range(0, len(text), chunk_size):
        yield text[i:i+chunk_size]
        await asyncio.sleep(0)

@pytest.mark.asyncio
async def test_stream_handler_thought_and_answer():
    callbacks = {
        "on_thought_start": Mock(),
        "on_thought_token": Mock(),
        "on_thought_end": Mock(),
        "on_token": Mock()
    }
    handler = StreamTokenHandler(callbacks)
    
    json_text = '{"thought": "Thinking process...", "type": "answer", "content": "Final Answer"}'
    
    await handler.process_stream(mock_stream(json_text))
    
    callbacks["on_thought_start"].assert_called_once()
    assert callbacks["on_thought_end"].call_count == 1
    
    # Check tokens
    thought_calls = [c.args[0] for c in callbacks["on_thought_token"].call_args_list]
    thought_str = "".join(thought_calls)
    assert "Thinking process..." in thought_str
    
    content_calls = [c.args[0] for c in callbacks["on_token"].call_args_list]
    content_str = "".join(content_calls)
    assert "Final Answer" in content_str

@pytest.mark.asyncio
async def test_stream_handler_tool_call():
    callbacks = {
        "on_thought_start": Mock(),
        "on_thought_token": Mock(),
        "on_thought_end": Mock(),
        "on_token": Mock()
    }
    handler = StreamTokenHandler(callbacks)
    
    json_text = '{"thought": "I need a tool", "type": "tool_call", "content": {"name": "test"}}'
    
    await handler.process_stream(mock_stream(json_text))
    
    callbacks["on_thought_start"].assert_called_once()
    callbacks["on_thought_end"].assert_called_once()
    
    # Tool call content should NOT be streamed to on_token
    callbacks["on_token"].assert_not_called()

@pytest.mark.asyncio
async def test_stream_handler_escaped_quotes():
    callbacks = {
        "on_thought_token": Mock()
    }
    handler = StreamTokenHandler(callbacks)
    
    # "thought": "Thinking \"quoted\"..."
    json_text = '{"thought": "Thinking \\"quoted\\"...", "type": "answer", "content": ""}'
    
    await handler.process_stream(mock_stream(json_text))
    
    thought_calls = [c.args[0] for c in callbacks["on_thought_token"].call_args_list]
    thought_str = "".join(thought_calls)
    assert 'Thinking "quoted"...' in thought_str
