"""Additional context manager tests migrated from legacy `src/test` files."""

import logging


class TestContextRemaining:
    """context/manager 最后边界"""

    def test_debug_log_add_exchange(self, caplog):
        caplog.set_level(logging.DEBUG, logger="ark.context")
        from src.core.context.manager import ConversationContext

        ctx = ConversationContext()
        ctx.add_exchange("Hello", "Hi")
        assert any("Added exchange" in record.msg for record in caplog.records)

    def test_debug_log_add_turn(self, caplog):
        caplog.set_level(logging.DEBUG, logger="ark.context")
        from src.core.context.manager import ConversationContext, ConversationTurn

        ctx = ConversationContext()
        turn = ConversationTurn(user_input="Hi", agent_response="Hello")
        ctx.add_turn(turn)
        assert any("Added turn" in record.msg for record in caplog.records)

    def test_compress_history_below_threshold(self):
        import asyncio

        from src.core.context.manager import ConversationContext

        ctx = ConversationContext(max_history=100)
        ctx.add_exchange("U1", "A1")
        asyncio.run(ctx.compress_history(threshold=20))
        assert len(ctx) == 1

    def test_get_conversation_summary_with_intents(self):
        from src.core.context.manager import ConversationContext

        ctx = ConversationContext()
        ctx.add_exchange("Hello", "Hi", intent="greeting")
        ctx.add_exchange("Weather?", "Sunny", intent="weather")
        summary = ctx.get_conversation_summary()
        assert "greeting" in summary["intents"]
        assert "weather" in summary["intents"]

    def test_update_last_exchange_replaces_response(self):
        from src.core.context.manager import ConversationContext

        ctx = ConversationContext()
        ctx.add_exchange(
            "Need approval",
            "Security: tool requires approval",
            tools_used=["write_file"],
            metadata={"todo_count": 1},
        )

        ctx.update_last_exchange(
            agent_response="Final answer",
            raw_response='{"type":"answer"}',
            tools_used=["task-1"],
            metadata={"approval_continued": True},
        )

        turn = ctx.conversation_history[-1]
        assert turn.user_input == "Need approval"
        assert turn.agent_response == "Final answer"
        assert turn.raw_response == '{"type":"answer"}'
        assert turn.tools_used == ["task-1"]
        assert turn.metadata["todo_count"] == 1
        assert turn.metadata["approval_continued"] is True

    def test_update_last_exchange_requires_existing_history(self):
        from src.core.context.manager import ConversationContext

        ctx = ConversationContext()

        try:
            ctx.update_last_exchange(agent_response="Final answer")
        except ValueError as exc:
            assert str(exc) == "No conversation history available to update"
        else:
            raise AssertionError(
                "Expected ValueError when no conversation history exists"
            )

    def test_activate_session_updates_session_metadata(self):
        from src.core.context.manager import ConversationContext

        ctx = ConversationContext(session_id="session-old")
        ctx.activate_session("session-new")

        assert ctx.session_id == "session-new"
        assert ctx.session_metadata["session_id"] == "session-new"
