
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from src.core.ark.engine import ARKEngine
from langchain_core.messages import HumanMessage, AIMessage

@pytest.mark.asyncio
async def test_json_history_format():
    """
    Integration test to verify that conversation history maintains raw JSON format
    for Assistant messages, ensuring compliance with JSON-based system prompts.
    """
    with patch('src.core.ark.engine.LLMManager') as MockLLMManager, \
         patch('src.core.ark.engine.ARKMCPClient') as MockMCPClient, \
         patch('src.core.ark.graph.StateGraph') as MockStateGraph:
        
        # Setup Mock Engine
        engine = ARKEngine()
        engine.state = engine.state.READY
        
        # Mock the graph execution
        mock_graph = AsyncMock()
        engine.graph = mock_graph
        
        # Simulated raw JSON response from MasterNode
        raw_json_response = '{"thought": "I should greet the user.", "type": "answer", "content": "Hello!"}'
        parsed_response = "I should greet the user.\n\nHello!"
        
        async def side_effect(initial_state, config=None):
            messages = initial_state["messages"]
            
            # Create AIMessage with raw_json in additional_kwargs
            ai_msg = AIMessage(
                content=parsed_response,
                additional_kwargs={"raw_json": raw_json_response}
            )
            
            return {
                "messages": messages + [ai_msg],
                "todo_list": [],
            }
            
        mock_graph.ainvoke.side_effect = side_effect
        
        # Turn 1: User says Hi
        await engine.process_input("Hi")
        
        # Verify context stored correctly
        assert len(engine.context_manager.conversation_history) == 1
        turn = engine.context_manager.conversation_history[0]
        assert turn.agent_response == parsed_response # User facing content
        assert turn.raw_response == raw_json_response # Raw JSON
        
        # Turn 2: User says "How are you?"
        # Verify that the history passed to graph uses raw_json
        await engine.process_input("How are you?")
        
        call_args = mock_graph.ainvoke.call_args_list[1]
        initial_state = call_args[0][0]
        messages = initial_state["messages"]
        
        # Expected: [Human("Hi"), AI(raw_json_response), Human("How are you?")]
        assert len(messages) == 3
        assert messages[0].content == "Hi"
        assert isinstance(messages[1], AIMessage)
        assert messages[1].content == raw_json_response # THIS IS KEY
        assert messages[2].content == "How are you?"
        
        print("JSON history verification passed!")

if __name__ == "__main__":
    asyncio.run(test_json_history_format())
