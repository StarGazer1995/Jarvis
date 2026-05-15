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


class TestConversationTurnDetailed:
    """ConversationTurn 详细测试 - 覆盖边界情况"""

    def test_processing_time_property(self):
        turn = ConversationTurn(user_input="Hi", agent_response="Hello")
        assert turn.processing_time == 0.0
        turn.processing_time = 1.5
        assert turn.processing_time == 1.5

    def test_to_dict_full(self):
        turn = ConversationTurn(
            user_input="Hello",
            agent_response="World",
            intent="greeting",
            entities={"name": "test"},
            tools_used=["search"],
            metadata={"source": "test"},
            raw_response='{"answer": "World"}',
        )
        d = turn.to_dict()
        assert d["intent"] == "greeting"
        assert d["tools_used"] == ["search"]
        assert d["raw_response"] == '{"answer": "World"}'

    def test_from_dict(self):
        from datetime import datetime

        d = {
            "user_input": "Hi",
            "agent_response": "Hey",
            "intent": "greeting",
            "entities": {"lang": "en"},
            "tools_used": [],
            "metadata": {},
            "raw_response": None,
        }
        turn = ConversationTurn.from_dict(d)
        assert turn.user_input == "Hi"
        assert turn.agent_response == "Hey"
        assert turn.intent == "greeting"

    def test_to_langchain_with_raw_response(self):
        turn = ConversationTurn(
            user_input="Hi",
            agent_response="Hello",
            raw_response='{"answer": "Hello"}',
        )
        messages = turn.to_langchain_message()
        assert len(messages) == 2
        # raw_response should be used as AI content
        assert messages[1].content == '{"answer": "Hello"}'

    def test_to_langchain_empty_input(self):
        turn = ConversationTurn(user_input="", agent_response="")
        messages = turn.to_langchain_message()
        assert len(messages) == 0


class TestConversationContextDetailed:
    """ConversationContext 详细测试"""

    def test_init_with_session_id(self):
        ctx = ConversationContext(max_history=50, session_id="custom-123")
        assert ctx.max_history == 50
        assert ctx.session_id == "custom-123"
        assert len(ctx) == 0

    def test_init_without_session_id(self):
        ctx = ConversationContext()
        assert ctx.session_id is not None
        assert len(ctx.session_id) == 8

    def test_user_memory_alias(self):
        ctx = ConversationContext()
        ctx.user_memory = {"key": "value"}
        assert ctx.user_preferences["key"] == "value"
        assert ctx.user_memory["key"] == "value"

    def test_add_turn(self):
        ctx = ConversationContext()
        turn = ConversationTurn(user_input="Hello", agent_response="World")
        ctx.add_turn(turn)
        assert len(ctx) == 1
        assert ctx.conversation_history[0].user_input == "Hello"

    def test_add_turn_with_history_limit(self):
        ctx = ConversationContext(max_history=2)
        ctx.add_exchange("U1", "A1")
        ctx.add_exchange("U2", "A2")
        ctx.add_exchange("U3", "A3")  # Should evict U1/A1
        assert len(ctx) == 2
        assert ctx.conversation_history[0].user_input == "U2"

    def test_update_memory(self):
        ctx = ConversationContext()
        ctx.update_memory("name", "Alice")
        assert ctx.user_preferences["name"] == "Alice"

    def test_get_memory(self):
        ctx = ConversationContext()
        ctx.update_memory("name", "Bob")
        assert ctx.get_memory("name") == "Bob"
        assert ctx.get_memory("nonexistent", "default") == "default"

    def test_clear_memory(self):
        ctx = ConversationContext()
        ctx.update_memory("key", "value")
        ctx.clear_memory()
        assert ctx.user_preferences == {}

    def test_len_and_iter(self):
        ctx = ConversationContext()
        ctx.add_exchange("U1", "A1")
        ctx.add_exchange("U2", "A2")
        assert len(ctx) == 2
        turns = list(iter(ctx))
        assert len(turns) == 2

    def test_str_and_repr(self):
        ctx = ConversationContext(session_id="test123")
        s = str(ctx)
        assert "test123" in s
        r = repr(ctx)
        assert "test123" in r
        assert "max_history" in r

    def test_reset(self):
        ctx = ConversationContext(session_id="keep-me")
        ctx.add_exchange("U1", "A1")
        old_session_id = ctx.session_id
        ctx.reset()
        assert len(ctx) == 0
        assert ctx.session_id == old_session_id

    def test_get_recent_turns(self):
        ctx = ConversationContext()
        for i in range(10):
            ctx.add_exchange(f"U{i}", f"A{i}")
        recent = ctx.get_recent_turns(3)
        assert len(recent) == 3  # Most recent 3
        assert recent[0].user_input == "U9"  # Most recent first

    def test_get_recent_turns_empty(self):
        ctx = ConversationContext()
        assert ctx.get_recent_turns() == []

    def test_get_conversation_summary(self):
        ctx = ConversationContext()
        ctx.add_exchange("Hello", "Hi", intent="greeting")
        ctx.add_exchange("Weather?", "Sunny", intent="weather")
        summary = ctx.get_conversation_summary()
        assert summary["total_turns"] == 2
        assert "greeting" in summary["intents"]

    def test_get_conversation_summary_empty(self):
        ctx = ConversationContext()
        summary = ctx.get_conversation_summary()
        assert summary["total_turns"] == 0

    def test_update_user_preference(self):
        ctx = ConversationContext()
        ctx.update_user_preference("theme", "dark")
        assert ctx.get_user_preference("theme") == "dark"
        assert ctx.get_user_preference("nonexistent", "light") == "light"

    def test_context_variables(self):
        ctx = ConversationContext()
        ctx.set_context_variable("current_task", "testing")
        assert ctx.get_context_variable("current_task") == "testing"
        assert ctx.get_context_variable("missing", "fallback") == "fallback"
        ctx.clear_context_variables()
        assert ctx.current_context == {}

    def test_get_session_stats_with_data(self):
        ctx = ConversationContext()
        ctx.add_exchange("U1", "A1")
        ctx.add_exchange("U2", "A2", tools_used=["search"])
        stats = ctx.get_session_stats()
        assert stats["total_turns"] == 2
        assert stats["tools_used"] == 1
        assert stats["session_duration"] >= 0

    def test_get_session_stats_empty(self):
        ctx = ConversationContext()
        stats = ctx.get_session_stats()
        assert stats["total_turns"] == 0

    def test_export_json(self):
        ctx = ConversationContext(session_id="export-test")
        ctx.add_exchange("Hello", "Hi")
        exported = ctx.export_conversation(export_format="json")
        import json
        data = json.loads(exported)
        assert data["session_metadata"]["session_id"] == "export-test"
        assert len(data["conversation_history"]) == 1

    def test_export_text(self):
        ctx = ConversationContext(session_id="text-export")
        ctx.add_exchange("Hello", "Hi")
        exported = ctx.export_conversation(export_format="text")
        assert "Hello" in exported
        assert "Hi" in exported
        assert "text-export" in exported

    def test_export_invalid_format(self):
        ctx = ConversationContext()
        with pytest.raises(ValueError, match="Unsupported export format"):
            ctx.export_conversation(export_format="xml")

    def test_reset_session(self):
        ctx = ConversationContext(session_id="old-session")
        ctx.add_exchange("U1", "A1")
        ctx.reset_session()
        assert len(ctx) == 0
        assert ctx.session_id != "old-session"
