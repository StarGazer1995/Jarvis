"""Additional stream handler tests migrated from `src/test`."""


class TestStreamHandlerCoverage:
    """覆盖 stream_handler.py 中未覆盖的边界"""

    def test_emit_callback_none_content(self):
        """测试 _emit_callback 无内容时调用回调"""
        from src.core.llm.stream_handler import StreamTokenHandler

        handler = StreamTokenHandler({"on_thought": lambda: None})
        handler._emit_callback("on_thought")
        handler._emit_callback("on_thought", content="some content")
        handler._emit_callback("nonexistent")  # should return early

    def test_emit_callback_error_handling(self):
        """测试 _emit_callback 异常处理"""
        from src.core.llm.stream_handler import StreamTokenHandler

        def failing_cb():
            raise ValueError("callback failed")

        handler = StreamTokenHandler({"on_thought": failing_cb})
        handler._emit_callback("on_thought")  # should not raise
