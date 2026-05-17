"""Additional LLM error handler tests migrated from `src/test`."""


class TestErrorHandlerCoverage:
    """覆盖 error_handler.py 中未覆盖的边界"""

    def test_log_llm_error_with_details(self):
        """测试 LLM 错误日志"""
        from src.core.llm.utils.error_handler import LLMError, log_llm_error

        error = LLMError("test error", details={"key": "value"})
        log_llm_error(error)

    def test_handle_none_error(self):
        """测试处理 None 错误"""
        from src.core.llm.utils.error_handler import handle_openai_error

        result = handle_openai_error(None)
        assert result is not None
