"""
LLM Client 补充测试
"""

from src.core.llm.types import LLMMessage, LLMResponse, TokenUsage


class TestLLMMessageModel:
    def test_message_repr(self):
        msg = LLMMessage(role="user", content="hello")
        r = repr(msg)
        assert "user" in r

    def test_message_empty_content(self):
        msg = LLMMessage(role="assistant", content="")
        assert msg.content == ""


class TestLLMResponseEdgeCases:
    def test_response_with_token_usage(self):
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        resp = LLMResponse(content="answer", usage=usage)
        assert resp.usage.total_tokens == 15
