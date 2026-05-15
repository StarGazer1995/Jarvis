from datetime import datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.core.context.manager import ConversationContext, ConversationTurn


class TestConversationContext:
    @pytest.fixture
    def context(self):
        return ConversationContext()

    def test_turn_to_langchain_message(self):
        turn = ConversationTurn(
            user_input="Hello",
            agent_response="Hi there",
            timestamp=datetime.now().timestamp(),
        )
        messages = turn.to_langchain_message()
        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Hello"
        assert isinstance(messages[1], AIMessage)
        assert messages[1].content == "Hi there"

    def test_get_cleaned_history_basic(self, context):
        context.add_exchange("User 1", "AI 1")
        context.add_exchange("User 2", "AI 2")

        # Default behavior: Keep all valid messages
        # Result: User 1, AI 1, User 2, AI 2
        history = context.get_cleaned_history()

        assert len(history) == 4
        assert isinstance(history[0], HumanMessage)
        assert history[0].content == "User 1"
        assert isinstance(history[1], AIMessage)
        assert history[1].content == "AI 1"
        assert isinstance(history[2], HumanMessage)
        assert history[2].content == "User 2"
        assert isinstance(history[3], AIMessage)
        assert history[3].content == "AI 2"

    def test_r1_clear_command(self, context):
        context.add_exchange("User 1", "AI 1")
        context.add_exchange("reset", "AI 2")
        context.add_exchange("User 3", "AI 3")

        # R1: Clear before "reset".
        # "reset" is at index 2 (User message).
        # "AI 2" is at index 3.
        # Messages: [U1, A1, Reset, A2, U3, A3]
        # reset_index = 2.
        # Keep messages from index 3 onwards: [A2, U3, A3]
        # R6: Ensure starts with User. -> Remove A2.
        # Result: [U3, A3]

        history = context.get_cleaned_history()

        assert len(history) == 2
        assert isinstance(history[0], HumanMessage)
        assert history[0].content == "User 3"
        assert isinstance(history[1], AIMessage)
        assert history[1].content == "AI 3"

        # Check that User 1 is NOT in history
        contents = [m.content for m in history]
        assert "User 1" not in contents

    def test_r2_filter_invalid(self, context):
        context.add_exchange("User 1", "AI 1")
        context.add_exchange("User 2", "")  # Empty AI response
        context.add_exchange("User 3", "AI 3")

        # R2: Filter invalid. Empty AI response is invalid.
        # So "User 2", "" -> "User 2" remains, "" is removed.
        # Sequence: U1, A1, U2, U3, A3
        # R4: Merge consecutive user. Merge U2 and U3.
        # Result: U1, A1, U2+U3, A3

        history = context.get_cleaned_history()

        assert len(history) == 4
        assert history[0].content == "User 1"
        assert history[1].content == "AI 1"
        assert isinstance(history[2], HumanMessage)
        assert "User 2" in history[2].content
        assert "User 3" in history[2].content
        assert history[3].content == "AI 3"

    def test_r3_preserve_trailing_assistant(self, context):
        context.add_exchange("User 1", "AI 1")

        history = context.get_cleaned_history()

        # R3 (Removed): Preserve valid AI 1.
        assert len(history) == 2
        assert history[0].content == "User 1"
        assert history[1].content == "AI 1"

    def test_r4_merge_consecutive_user(self, context):
        context.add_exchange("User 1", "AI 1")
        context.add_exchange("User 2", "")  # Invalid AI
        context.add_exchange("User 3", "AI 3")

        # "" removed by R2.
        # Sequence: U1, A1, U2, U3, A3.
        # Merge U2, U3 -> U1, A1, U2+U3, A3.

        history = context.get_cleaned_history()
        assert len(history) == 4
        assert isinstance(history[2], HumanMessage)
        assert "User 2" in history[2].content
        assert "User 3" in history[2].content
        assert history[3].content == "AI 3"

    def test_r5_truncate(self, context):
        for i in range(10):
            context.add_exchange(f"U{i}", f"A{i}")

        # 10 turns = 20 messages.
        # Truncate to e.g. 5 messages.
        # [U0, A0, ..., U9, A9] (20 msgs)
        # Last 5: [A7, U8, A8, U9, A9]
        # R6: Ensure starts with User. -> Remove A7.
        # Result: [U8, A8, U9, A9] (4 messages)

        history = context.get_cleaned_history(max_messages=5)

        assert len(history) == 4
        assert isinstance(history[0], HumanMessage)
        assert history[0].content == "U8"

    def test_r6_start_with_user(self, context):
        context.add_exchange("User 1", "AI 1")
        # Let's simulate by truncating such that AI is first.
        # [U1, A1, U2, A2] -> remove A2 (Wait, R3 removed? No) -> [U1, A1, U2, A2]
        # max_messages=3 -> [A1, U2, A2].
        # R6 -> Remove A1 -> [U2, A2].

        context.add_exchange("User 2", "AI 2")
        history = context.get_cleaned_history(max_messages=3)

        assert len(history) == 2
        assert isinstance(history[0], HumanMessage)
        assert history[0].content == "User 2"
        assert history[1].content == "AI 2"
