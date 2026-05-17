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
