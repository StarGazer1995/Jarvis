"""
LLM 消息转换器单元测试

测试 LangChain → LLM 消息的转换函数。
"""

from src.core.llm.converters import convert_langchain_to_llm_messages


class TestConvertLangchainToLLM:
    """测试 LangChain → LLM 消息转换"""

    def test_system_message(self):
        from langchain_core.messages import SystemMessage

        lc_msg = SystemMessage(content="You are a helpful assistant")
        result = convert_langchain_to_llm_messages([lc_msg])
        assert len(result) == 1
        assert result[0].role == "system"
        assert result[0].content == "You are a helpful assistant"

    def test_human_message(self):
        from langchain_core.messages import HumanMessage

        lc_msg = HumanMessage(content="Hello!")
        result = convert_langchain_to_llm_messages([lc_msg])
        assert result[0].role == "user"

    def test_ai_message(self):
        from langchain_core.messages import AIMessage

        lc_msg = AIMessage(content="Hi there!")
        result = convert_langchain_to_llm_messages([lc_msg])
        assert result[0].role == "assistant"

    def test_multiple_messages(self):
        from langchain_core.messages import HumanMessage, SystemMessage

        msgs = [
            SystemMessage(content="Be concise"),
            HumanMessage(content="Question?"),
        ]
        result = convert_langchain_to_llm_messages(msgs)
        assert len(result) == 2
        assert result[0].role == "system"
        assert result[1].role == "user"

    def test_empty_list(self):
        result = convert_langchain_to_llm_messages([])
        assert result == []

    def test_tool_message(self):
        from langchain_core.messages import ToolMessage

        lc_msg = ToolMessage(
            content="Tool result", tool_call_id="call_1", name="search"
        )
        result = convert_langchain_to_llm_messages([lc_msg])
        assert result[0].role == "user"
        assert "Observation: Tool result" in result[0].content
        assert result[0].metadata.get("tool_name") == "search"
        assert result[0].metadata.get("tool_call_id") == "call_1"

    def test_ai_message_with_tool_calls(self):
        from langchain_core.messages import AIMessage

        lc_msg = AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {"q": "test"}, "id": "call_1"}],
        )
        result = convert_langchain_to_llm_messages([lc_msg])
        assert result[0].role == "assistant"
        assert "tool_calls" in result[0].metadata
        assert len(result[0].metadata["tool_calls"]) == 1

    def test_message_with_name(self):
        from langchain_core.messages import HumanMessage

        lc_msg = HumanMessage(content="Hello", name="user123")
        result = convert_langchain_to_llm_messages([lc_msg])
        assert result[0].metadata.get("name") == "user123"

    def test_additional_kwargs(self):
        from langchain_core.messages import AIMessage

        lc_msg = AIMessage(
            content="Response",
            additional_kwargs={"custom_field": "value"},
        )
        result = convert_langchain_to_llm_messages([lc_msg])
        assert result[0].metadata.get("custom_field") == "value"
