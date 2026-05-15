"""
OpenAI客户端单元测试

测试OpenAI客户端的各种功能和错误处理。
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.llm.client import LLMConfig, LLMMessage, LLMProvider
from src.core.llm.providers.openai_client import OpenAILLMClient
from src.core.llm.utils.error_handler import (
    LLMConfigurationError,
)


class TestOpenAILLMClient:
    """OpenAI客户端测试类"""

    @pytest.fixture
    def valid_config(self):
        """有效的配置"""
        return LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="test-api-key",
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000,
            timeout=30.0,
        )

    @pytest.fixture
    def invalid_config(self):
        """无效的配置"""
        return LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="",  # 空的API密钥
            model="gpt-3.5-turbo",
        )

    @pytest.fixture
    def test_messages(self):
        """测试消息"""
        return [LLMMessage(role="user", content="Hello, how are you?")]

    def test_client_initialization(self, valid_config):
        """测试客户端初始化"""
        client = OpenAILLMClient(valid_config)
        assert client.config == valid_config
        assert not client._initialized
        assert client._request_count == 0
        assert client._total_tokens == 0

    def test_config_validation_missing_api_key(self, invalid_config):
        """测试缺少API密钥的配置验证"""
        client = OpenAILLMClient(invalid_config)

        with pytest.raises(LLMConfigurationError) as exc_info:
            client._validate_config()

        assert "API密钥未设置" in str(exc_info.value)
        assert exc_info.value.config_field == "api_key"

    def test_config_validation_invalid_temperature(self, valid_config):
        """测试无效温度的配置验证"""
        valid_config.temperature = 3.0  # 超出范围
        client = OpenAILLMClient(valid_config)

        with pytest.raises(LLMConfigurationError) as exc_info:
            client._validate_config()

        assert "temperature必须在0-2之间" in str(exc_info.value)

    def test_config_validation_invalid_max_tokens(self, valid_config):
        """测试无效最大tokens的配置验证"""
        valid_config.max_tokens = -1
        client = OpenAILLMClient(valid_config)

        with pytest.raises(LLMConfigurationError) as exc_info:
            client._validate_config()

        assert "max_tokens必须大于0" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_initialization_success(self, valid_config):
        """测试成功初始化"""
        client = OpenAILLMClient(valid_config)

        # Mock OpenAI库
        mock_openai = MagicMock()
        mock_async_client = AsyncMock()
        mock_openai.AsyncOpenAI.return_value = mock_async_client

        # Mock测试请求
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_async_client.chat.completions.create.return_value = mock_response

        # Mock httpx
        mock_httpx = MagicMock()

        # Patch sys.modules to return mocks for openai and httpx
        with patch.dict("sys.modules", {"openai": mock_openai, "httpx": mock_httpx}):
            result = await client.initialize()

        assert result is True
        assert client._initialized is True
        assert client._client is not None

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        "openai" not in sys.modules,
        reason="Only runs when openai is already imported (isolation guarantee)",
    )
    async def test_initialization_import_error(self, valid_config):
        """测试OpenAI库导入错误

        注意: 此测试依赖 import hook 机制，需要在 openai 已加载的环境下运行。
        通过删除 sys.modules 中的 openai 条目 + 拦截 __import__ 来模拟导入失败。
        """
        client = OpenAILLMClient(valid_config)

        # Remove openai from sys.modules so __import__ is triggered
        saved_modules = {}
        for mod in list(sys.modules.keys()):
            if mod == "openai" or mod.startswith("openai."):
                saved_modules[mod] = sys.modules.pop(mod)

        try:

            def raise_import_error(name, *args, **kwargs):
                if name == "openai" or name.startswith("openai."):
                    raise ImportError(f"No module named '{name}'")
                return __import__(name, *args, **kwargs)

            import builtins

            with patch.object(builtins, "__import__", side_effect=raise_import_error):
                result = await client.initialize()

            assert result is False
            assert client._initialized is False
        finally:
            # Restore modules
            sys.modules.update(saved_modules)

    @pytest.mark.asyncio
    async def test_initialization_authentication_error(self, valid_config):
        """测试认证错误"""
        client = OpenAILLMClient(valid_config)

        # Mock OpenAI库
        mock_openai = MagicMock()
        mock_async_client = AsyncMock()
        mock_openai.AsyncOpenAI.return_value = mock_async_client

        # Mock认证错误类
        class MockAuthenticationError(Exception):
            pass

        class MockRateLimitError(Exception):
            pass

        class MockAPITimeoutError(Exception):
            pass

        class MockAPIError(Exception):
            pass

        mock_openai.AuthenticationError = MockAuthenticationError
        mock_openai.RateLimitError = MockRateLimitError
        mock_openai.APITimeoutError = MockAPITimeoutError
        mock_openai.APIError = MockAPIError
        mock_async_client.chat.completions.create.side_effect = MockAuthenticationError(
            "Invalid API key"
        )

        mock_httpx = MagicMock()

        # Patch sys.modules for client's import
        # And patch error_handler's openai reference
        with patch.dict("sys.modules", {"openai": mock_openai, "httpx": mock_httpx}):
            with patch("src.core.llm.utils.error_handler.openai", mock_openai):
                result = await client.initialize()

        assert result is False
        assert client._initialized is False

    def test_convert_messages_to_openai(self, valid_config, test_messages):
        """测试消息格式转换"""
        client = OpenAILLMClient(valid_config)
        openai_messages = client._convert_messages_to_openai(test_messages)

        assert len(openai_messages) == 1
        assert openai_messages[0]["role"] == "user"
        assert openai_messages[0]["content"] == "Hello, how are you?"

    def test_prepare_request_params(self, valid_config):
        """测试请求参数准备"""
        client = OpenAILLMClient(valid_config)
        messages = [{"role": "user", "content": "Hello"}]

        params = client._prepare_request_params(messages, temperature=0.5)

        assert params["model"] == "gpt-3.5-turbo"
        assert params["messages"] == messages
        assert params["temperature"] == 0.5  # 覆盖默认值
        assert params["max_tokens"] == 1000
        assert params["stream"] is False

    @pytest.mark.asyncio
    async def test_generate_response_success(self, valid_config, test_messages):
        """测试成功生成响应"""
        client = OpenAILLMClient(valid_config)
        client._initialized = True

        # Mock OpenAI客户端和响应
        mock_client = AsyncMock()
        client._client = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello! I'm doing well, thank you."
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-3.5-turbo"
        mock_response.id = "test-id"
        mock_response.created = 1234567890
        mock_response.object = "chat.completion"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 15
        mock_response.usage.total_tokens = 25

        mock_client.chat.completions.create.return_value = mock_response

        # Mock速率限制处理器
        client._rate_limit_handler = AsyncMock()
        client._rate_limit_handler.wait_if_needed = AsyncMock()

        response = await client.generate_response(test_messages)

        assert response.content == "Hello! I'm doing well, thank you."
        assert response.model == "gpt-3.5-turbo"
        assert response.finish_reason == "stop"
        assert response.usage["total_tokens"] == 25
        assert client._request_count == 1
        assert client._total_tokens == 25

    @pytest.mark.asyncio
    async def test_generate_response_not_initialized(self, valid_config, test_messages):
        """测试未初始化时生成响应"""
        client = OpenAILLMClient(valid_config)

        with pytest.raises(RuntimeError) as exc_info:
            await client.generate_response(test_messages)

        assert "OpenAI客户端未初始化" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stream_response_success(self, valid_config, test_messages):
        """测试成功流式响应"""
        client = OpenAILLMClient(valid_config)
        client._initialized = True

        # Mock OpenAI客户端和流式响应
        mock_client = AsyncMock()
        client._client = mock_client

        # 创建模拟的流式响应
        class MockChunk:
            def __init__(self, content):
                self.choices = [MockChoice(content)]

        class MockChoice:
            def __init__(self, content):
                self.delta = MockDelta(content)

        class MockDelta:
            def __init__(self, content):
                self.content = content

        async def mock_stream():
            chunks = [MockChunk("Hello"), MockChunk(" there"), MockChunk("!")]
            for chunk in chunks:
                yield chunk

        mock_client.chat.completions.create.return_value = mock_stream()

        # Mock速率限制处理器
        client._rate_limit_handler = AsyncMock()
        client._rate_limit_handler.wait_if_needed = AsyncMock()

        # 收集流式响应
        chunks = []
        async for chunk in client.stream_response(test_messages):
            chunks.append(chunk)

        assert chunks == ["Hello", " there", "!"]
        assert client._request_count == 1

    @pytest.mark.asyncio
    async def test_stream_response_not_initialized(self, valid_config, test_messages):
        """测试未初始化时流式响应"""
        client = OpenAILLMClient(valid_config)

        with pytest.raises(RuntimeError) as exc_info:
            async for _ in client.stream_response(test_messages):
                pass

        assert "OpenAI客户端未初始化" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_close_client(self, valid_config):
        """测试关闭客户端"""
        client = OpenAILLMClient(valid_config)
        client._initialized = True
        client._request_count = 5
        client._total_tokens = 100

        # Mock客户端
        mock_client = AsyncMock()
        client._client = mock_client

        await client.close()

        assert client._client is None
        assert client._initialized is False
        mock_client.close.assert_called_once()

    def test_get_stats(self, valid_config):
        """测试获取统计信息"""
        client = OpenAILLMClient(valid_config)
        client._initialized = True
        client._request_count = 10
        client._total_tokens = 500

        stats = client.get_stats()

        assert stats["provider"] == "openai"
        assert stats["model"] == "gpt-3.5-turbo"
        assert stats["initialized"] is True
        assert stats["request_count"] == 10
        assert stats["total_tokens"] == 500
        assert "uptime_seconds" in stats
        assert "requests_per_minute" in stats
        assert "tokens_per_request" in stats


@pytest.mark.integration
class TestOpenAILLMClientIntegration:
    """OpenAI客户端集成测试"""

    @pytest.fixture
    def real_config(self):
        """真实的配置（需要环境变量）"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("需要设置OPENAI_API_KEY环境变量进行集成测试")

        return LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key=api_key,
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=50,  # 限制tokens以节省成本
            timeout=30.0,
        )

    @pytest.mark.asyncio
    async def test_real_openai_integration(self, real_config):
        """测试真实的OpenAI集成（需要API密钥）"""
        client = OpenAILLMClient(real_config)

        try:
            # 初始化客户端
            success = await client.initialize()
            assert success is True

            # 发送测试消息
            messages = [
                LLMMessage(role="user", content="Say 'Hello, World!' and nothing else.")
            ]

            response = await client.generate_response(messages)

            assert response.content
            assert response.model
            assert response.usage.get("total_tokens", 0) > 0

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_real_openai_streaming(self, real_config):
        """测试真实的OpenAI流式响应（需要API密钥）"""
        client = OpenAILLMClient(real_config)

        try:
            # 初始化客户端
            success = await client.initialize()
            assert success is True

            # 发送测试消息
            messages = [
                LLMMessage(
                    role="user", content="Count from 1 to 3, one number per line."
                )
            ]

            chunks = []
            async for chunk in client.stream_response(messages):
                chunks.append(chunk)

            assert len(chunks) > 0
            full_response = "".join(chunks)
            assert full_response

        finally:
            await client.close()
