"""
LLM 配置类型单元测试

测试 LLMConfig、LLMMessage、LLMResponse、TokenUsage、LLMProvider 的数据类功能。
"""

import os
from unittest.mock import patch

from src.core.llm.types import (
    LLMConfig,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    TokenUsage,
)


class TestLLMProvider:
    def test_enum_values(self):
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.AZURE_OPENAI.value == "azure_openai"
        assert LLMProvider.OLLAMA.value == "ollama"


class TestLLMMessage:
    def test_create_message(self):
        msg = LLMMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_with_metadata(self):
        msg = LLMMessage(role="assistant", content="Hi", metadata={"source": "test"})
        assert msg.metadata["source"] == "test"


class TestTokenUsage:
    def test_defaults(self):
        usage = TokenUsage()
        assert usage.total_tokens == 0

    def test_custom_values(self):
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20


class TestLLMResponse:
    def test_create_response(self):
        resp = LLMResponse(content="Answer", model="gpt-4")
        assert resp.content == "Answer"
        assert resp.model == "gpt-4"

    def test_response_with_usage(self):
        resp = LLMResponse(content="X", usage={"total_tokens": 50})
        assert resp.usage["total_tokens"] == 50


class TestLLMConfigFromDict:
    def test_minimal_dict(self):
        config = LLMConfig.from_dict({"provider": "openai", "api_key": "key"})
        assert config.provider == LLMProvider.OPENAI
        assert config.api_key == "key"
        assert config.model == "gpt-3.5-turbo"

    def test_anthropic_provider(self):
        config = LLMConfig.from_dict({"provider": "anthropic", "api_key": "ant-key"})
        assert config.provider == LLMProvider.ANTHROPIC

    def test_unknown_provider_fallsback(self):
        config = LLMConfig.from_dict({"provider": "nonexistent", "api_key": "key"})
        assert config.provider == LLMProvider.OPENAI

    def test_full_config(self):
        config = LLMConfig.from_dict(
            {
                "provider": "azure_openai",
                "api_key": "az-key",
                "model": "gpt-4",
                "base_url": "https://azure.com",
                "max_tokens": 2000,
                "temperature": 0.3,
                "timeout": 60,
                "stream": True,
            }
        )
        assert config.provider == LLMProvider.AZURE_OPENAI
        assert config.model == "gpt-4"
        assert config.max_tokens == 2000
        assert config.temperature == 0.3

    def test_retry_attempts_populates_retry(self):
        config = LLMConfig.from_dict(
            {"provider": "openai", "api_key": "k", "retry_attempts": 5}
        )
        assert config.retry["max_attempts"] == 5


class TestLLMConfigFromEnv:
    def test_from_env_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            config = LLMConfig.from_env()
            assert config.provider == LLMProvider.OPENAI
            assert config.api_key is None

    def test_from_env_with_values(self):
        env = {
            "LLM_PROVIDER": "anthropic",
            "OPENAI_API_KEY": "sk-test",
            "LLM_MODEL": "claude-3",
            "LLM_MAX_TOKENS": "2000",
            "LLM_TEMPERATURE": "0.5",
            "LLM_TIMEOUT": "60",
        }
        with patch.dict(os.environ, env, clear=True):
            config = LLMConfig.from_env()
            assert config.provider == LLMProvider.ANTHROPIC
            assert config.api_key == "sk-test"
            assert config.model == "claude-3"
            assert config.max_tokens == 2000
            assert config.temperature == 0.5
            assert config.timeout == 60


class TestLLMConfigPostInit:
    def test_retry_dict_from_retry_attempts(self):
        config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4", retry_attempts=5)
        assert config.retry == {"max_attempts": 5}

    def test_retry_dict_preserved_when_provided(self):
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            retry={"max_attempts": 10},
            retry_attempts=3,
        )
        assert config.retry == {"max_attempts": 10}
