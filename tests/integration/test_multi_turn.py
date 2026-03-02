import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from src.core.ark.engine import ARKEngine
from langchain_core.messages import HumanMessage, AIMessage


@pytest.mark.asyncio
async def test_multi_turn_conversation_flow():
    """
    Integration test for multi-turn conversation flow.
    Verifies that context is maintained and reset works.
    """
    # Mock LLM Manager to avoid real API calls
    with (
        patch("src.core.ark.engine.LLMManager") as MockLLMManager,
        patch("src.core.ark.engine.ARKMCPClient") as MockMCPClient,
        patch("src.core.ark.graph.StateGraph") as MockStateGraph,
    ):
        # Setup Mock Engine
        engine = ARKEngine()
        engine.state = engine.state.READY  # Force ready state

        # Mock the graph execution to simulate LLM responses based on input history
        mock_graph = AsyncMock()
        engine.graph = mock_graph

        # We need to simulate the state updates that the graph would do
        # Specifically, updating the conversation history via ARKEngine's process_input logic
        # But wait, ARKEngine updates history AFTER graph execution based on the result.
        # So we just need to return a simulated response from the graph.

        # Helper to simulate graph execution
        async def side_effect(initial_state, config=None):
            messages = initial_state["messages"]
            last_message = messages[-1]
            user_input = last_message.content

            # Simple logic to simulate memory
            response_content = "I don't know your name."

            # Check history for name
            # messages[0] might be SystemMessage (if added by MasterNode, but here we are at graph input)
            # The engine passes [History..., CurrentUser]

            history_text = " ".join(
                [
                    m.content
                    for m in messages
                    if isinstance(m, (HumanMessage, AIMessage))
                ]
            )

            if "My name is John" in history_text:
                if "What is my name" in user_input:
                    response_content = "Your name is John."

            if "reset" in user_input.lower():
                response_content = "Conversation reset."

            return {
                "messages": messages + [AIMessage(content=response_content)],
                "todo_list": [],
            }

        mock_graph.ainvoke.side_effect = side_effect

        # Turn 1: Introduce self
        response1 = await engine.process_input("My name is John")
        assert "John" in response1 or "name" in response1  # Just checking flow
        # Verify context updated
        assert len(engine.context_manager.conversation_history) == 1
        assert (
            engine.context_manager.conversation_history[0].user_input
            == "My name is John"
        )

        # Turn 2: Ask name
        # The mock side_effect logic above is a bit flawed because it checks 'history_text'
        # which is constructed from the INPUT messages.
        # In Turn 2, input messages will be [Turn1, Current].
        # So "My name is John" will be in history.

        response2 = await engine.process_input("What is my name?")
        # Our mock side_effect checks if "My name is John" is in history_text.
        # It should be there because engine injects it.

        # We can't easily assert the response content because we mocked the graph to be generic/dynamic
        # based on input.
        # But we CAN verify that the graph was called with the correct history.

        # Check call args for the second call
        call_args = mock_graph.ainvoke.call_args_list[1]
        initial_state = call_args[0][0]
        messages = initial_state["messages"]

        # Expected: [Human("My name is John"), AI("..."), Human("What is my name?")]
        # Note: AIMessage content depends on what the first side_effect returned.
        # Let's see what side_effect returned for first call.
        # It returned "I don't know your name." (default) or "Your name is John" (if condition met).
        # For "My name is John", it likely returned "I don't know your name." because condition "What is my name" was false.

        assert len(messages) == 3  # Turn 1 (Human+AI) + Turn 2 (Human)
        assert messages[0].content == "My name is John"
        assert isinstance(messages[1], AIMessage)
        assert messages[2].content == "What is my name?"

        print("Multi-turn context verification passed!")

        # Turn 3: Reset
        response3 = await engine.process_input("Reset")
        assert "reset" in response3.lower() or "conversation" in response3.lower()

        # Turn 4: Ask name again
        response4 = await engine.process_input("What is my name?")

        # Check call args for the fourth call
        call_args = mock_graph.ainvoke.call_args_list[3]
        initial_state = call_args[0][0]
        messages = initial_state["messages"]

        # Expected: History should be cleared or truncated due to reset.
        # Our get_cleaned_history logic:
        # History has: [T1, T2, T3(Reset)].
        # get_cleaned_history will see Reset at index 2 (0-based pairs? No, flat list).
        # T1: Human, AI
        # T2: Human, AI
        # T3: Human(Reset), AI(Reset)
        # It should return messages AFTER the reset.
        # So it should be empty (or contain T3 AI if we kept it? No, we drop everything up to Reset command).

        # Wait, get_cleaned_history implementation:
        # raw_messages = [H1, A1, H2, A2, H3(Reset), A3]
        # reset_index = 4 (H3)
        # returns raw_messages[5:] -> [A3]
        # Then R6 removes leading AI. -> []
        # So history should be empty.

        # So messages passed to graph should be [Human("What is my name?")]
        assert len(messages) == 1
        assert messages[0].content == "What is my name?"

        print("Reset functionality verification passed!")


if __name__ == "__main__":
    asyncio.run(test_multi_turn_conversation_flow())
