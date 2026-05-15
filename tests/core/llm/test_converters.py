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
