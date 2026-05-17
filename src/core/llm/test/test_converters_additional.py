"""Additional LLM converter tests migrated from legacy `src/test` files."""


class TestConvertersMoreCoverage:
    """转换器更多测试"""

    def test_convert_funtion_message(self):
        from langchain_core.messages import FunctionMessage

        from src.core.llm.converters import convert_langchain_to_llm_messages

        msg = FunctionMessage(content="result", name="func1")
        result = convert_langchain_to_llm_messages([msg])
        assert len(result) == 1
        # FunctionMessage maps to "user" role in the converter
        assert result[0].role == "user"

    def test_convert_tool_message(self):
        from langchain_core.messages import ToolMessage

        from src.core.llm.converters import convert_langchain_to_llm_messages

        msg = ToolMessage(content="tool output", tool_call_id="call1")
        result = convert_langchain_to_llm_messages([msg])
        assert len(result) == 1
        # ToolMessage maps to "user" role in the converter
        assert result[0].role == "user"
