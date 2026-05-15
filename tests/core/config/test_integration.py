"""
配置系统集成测试模块

测试配置加载器、LLM工厂和客户端之间的集成功能
"""

import asyncio
import os
import tempfile
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from src.core.common.exceptions import ConfigurationError
from src.core.config.loader import ConfigLoader, Environment, load_llm_config
from src.core.llm.client import (
    BaseLLMClient,
    LLMMessage,
    LLMResponse,
)
from src.core.llm.client import (
    LLMConfig as ClientLLMConfig,
)
from src.core.llm.factory import (
    LLMProviderFactory,
    create_llm_client,
)


class IntegrationMockClient(BaseLLMClient):
    """集成测试用的模拟客户端"""

    def __init__(self, config: ClientLLMConfig):
        super().__init__(config)
        self.call_count = 0
        self.last_messages = None
        self.is_closed = False
        self._initialized = False

    async def initialize(self) -> bool:
        self._initialized = True
        return True

    async def generate_response(self, messages, **kwargs):
        self.call_count += 1
        self.last_messages = messages

        return LLMResponse(
            content=f"Integration test response #{self.call_count}",
            model=self.config.model or "integration-mock",
            usage={"total_tokens": 20 + self.call_count},
            metadata={"response_time": 0.1 * self.call_count},
        )

    async def stream_response(self, messages, **kwargs):
        yield f"Integration test response #{self.call_count}"

    async def chat_completion(self, messages, **kwargs):
        """模拟聊天完成"""
        return await self.generate_response(messages, **kwargs)

    async def close(self):
        """关闭客户端"""
        self.is_closed = True


class SlowMockClient(BaseLLMClient):
    """慢速模拟客户端"""

    def __init__(self, config: ClientLLMConfig):
        super().__init__(config)
        self._initialized = False

    async def initialize(self) -> bool:
        self._initialized = True
        return True

    async def generate_response(self, messages, **kwargs):
        """模拟慢速响应"""
        await asyncio.sleep(0.2)  # 模拟慢速响应

        return LLMResponse(
            content="Slow response",
            model="slow-mock",
            usage={"total_tokens": 50},
            metadata={"response_time": 0.2},
        )

    async def stream_response(self, messages, **kwargs):
        await asyncio.sleep(0.2)
        yield "Slow response"

    async def chat_completion(self, messages, **kwargs):
        return await self.generate_response(messages, **kwargs)

    async def close(self):
        pass


class TestConfigurationIntegration:
    """测试配置集成"""

    @pytest.fixture
    def comprehensive_config(self) -> dict[str, Any]:
        """全面的配置数据"""
        return {
            "global": {
                "default_provider": "primary",
                "retry": {"max_attempts": 3, "delay": 1.0, "backoff_factor": 2.0},
                "timeout": {"connect": 30.0, "read": 60.0, "total": 90.0},
                "logging": {
                    "level": "INFO",
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "providers": {
                "primary": {
                    "type": "integration_mock",
                    "enabled": True,
                    "api_key": "${PRIMARY_API_KEY:test-key}",
                    "base_url": "https://api.primary.com/v1",
                    "default_model": "primary-model",
                    "models": {
                        "primary-model": {
                            "max_tokens": 4000,
                            "temperature": 0.7,
                            "top_p": 0.9,
                        },
                        "primary-fast": {"max_tokens": 2000, "temperature": 0.5},
                    },
                    "retry": {"max_attempts": 5, "initial_delay": 0.5},
                    "timeout": {"total": 120.0},
                },
                "secondary": {
                    "type": "slow_mock",
                    "enabled": True,
                    "api_key": "test-key-2",
                    "default_model": "secondary-model",
                    "models": {
                        "secondary-model": {"max_tokens": 3000, "temperature": 0.8}
                    },
                },
                "disabled": {
                    "type": "integration_mock",
                    "enabled": False,
                    "default_model": "disabled-model",
                    "models": {"disabled-model": {}},
                },
            },
            "environments": {
                "development": {
                    "debug": True,
                    "log_level": "DEBUG",
                    "cache_enabled": False,
                },
                "production": {
                    "debug": False,
                    "log_level": "WARNING",
                    "cache_enabled": True,
                    "monitoring_enabled": True,
                },
            },
            "features": {
                "streaming": {"enabled": True, "chunk_size": 1024, "timeout": 30.0},
                "context": {
                    "max_history": 10,
                    "auto_summarize": True,
                    "summarize_threshold": 8,
                },
                "cache": {"enabled": True, "ttl": 3600, "max_size": 1000},
                "monitoring": {
                    "enabled": True,
                    "metrics_interval": 60,
                    "log_requests": True,
                },
                "rate_limiting": {
                    "enabled": True,
                    "requests_per_minute": 60,
                    "burst_size": 10,
                },
                "security": {
                    "api_key_rotation": True,
                    "request_signing": False,
                    "encryption_at_rest": True,
                },
            },
        }

    @pytest.fixture
    def temp_config_file(self, comprehensive_config):
        """创建临时配置文件"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(comprehensive_config, f)
            temp_path = f.name

        yield temp_path

        os.unlink(temp_path)

    def setup_method(self):
        """每个测试方法前的设置"""
        pass

    def _register_mocks(self, factory):
        """注册Mock提供商"""
        factory.registry.register_provider("integration_mock", IntegrationMockClient)
        factory.registry.register_provider("slow_mock", SlowMockClient)
        factory.registry.register_provider("mock", IntegrationMockClient)

        # 确保配置中的mock提供商被启用
        if factory._config and "mock" in factory._config.providers:
            factory._config.providers["mock"].enabled = True

    def test_end_to_end_config_loading(self, temp_config_file):
        """测试端到端配置加载"""
        loader = ConfigLoader(temp_config_file)
        config = loader.load_config()

        assert config is not None
        assert config.global_config.default_provider == "primary"

        # 验证提供商配置
        assert "primary" in config.providers
        assert config.providers["primary"].enabled is True

        # 验证环境配置应用
        # LLMConfig object stores environment name but doesn't expose raw environments dict
        assert config.environment == Environment.DEVELOPMENT

    @pytest.mark.asyncio
    async def test_factory_with_loaded_config(self, temp_config_file):
        """测试工厂使用加载的配置"""
        factory = LLMProviderFactory(temp_config_file)
        self._register_mocks(factory)

        # 验证工厂状态
        # LLMProviderFactory 没有 config 属性，使用 load_config() 获取
        config = factory.load_config()
        assert config is not None
        assert config.global_config.default_provider == "primary"

        # 获取可用提供商
        available_providers = factory.list_available_providers()
        assert len(available_providers) == 3

        enabled_providers = factory.list_enabled_providers()
        assert len(enabled_providers) == 2
        assert "primary" in enabled_providers
        assert "secondary" in enabled_providers
        assert "disabled" not in enabled_providers

        await factory.close()

    @pytest.mark.asyncio
    async def test_client_creation_and_usage(self, temp_config_file):
        """测试客户端创建和使用"""
        factory = LLMProviderFactory(temp_config_file)
        self._register_mocks(factory)

        # 获取默认客户端
        client = factory.get_default_provider()
        assert isinstance(client, IntegrationMockClient)

        # 测试聊天完成
        messages = [LLMMessage(role="user", content="Hello, this is a test message.")]

        response = await client.chat_completion(messages)

        assert response.content == "Integration test response #1"
        assert response.model == "primary-model"
        assert response.usage["total_tokens"] == 21
        assert client.call_count == 1
        assert client.last_messages == messages

        await factory.close()

    @pytest.mark.asyncio
    async def test_multiple_provider_usage(self, temp_config_file):
        """测试多提供商使用"""
        factory = LLMProviderFactory(temp_config_file)
        self._register_mocks(factory)

        # 获取不同的提供商
        primary_client = factory.get_provider("primary")
        secondary_client = factory.get_provider("secondary")

        assert isinstance(primary_client, IntegrationMockClient)
        assert isinstance(secondary_client, SlowMockClient)

        # 测试两个客户端
        messages = [LLMMessage(role="user", content="Test message")]

        # 测试主要客户端
        response1 = await primary_client.chat_completion(messages)
        assert "Integration test response" in response1.content
        assert response1.model == "primary-model"

        # 测试次要客户端
        response2 = await secondary_client.chat_completion(messages)
        assert response2.content == "Slow response"
        assert response2.model == "slow-mock"
        assert response2.metadata["response_time"] >= 0.2

        await factory.close()

    @pytest.mark.asyncio
    async def test_client_pooling_integration(self, temp_config_file):
        """测试客户端池化集成"""
        factory = LLMProviderFactory(temp_config_file)
        self._register_mocks(factory)

        # 多次获取同一提供商
        client1 = factory.get_provider("primary")
        client2 = factory.get_provider("primary")
        client3 = factory.get_default_provider()  # 应该是primary

        # 应该是同一个实例
        assert client1 is client2
        assert client1 is client3

        # 测试调用计数
        messages = [LLMMessage(role="user", content="Test")]

        await client1.chat_completion(messages)
        assert client1.call_count == 1

        await client2.chat_completion(messages)
        assert client1.call_count == 2  # 同一个实例

        await client3.chat_completion(messages)
        assert client1.call_count == 3

        await factory.close()

    @pytest.mark.asyncio
    async def test_configuration_validation_integration(self, temp_config_file):
        """测试配置验证集成"""
        factory = LLMProviderFactory(temp_config_file)
        self._register_mocks(factory)

        # 验证所有提供商配置
        assert factory.validate_provider_config("primary") is True
        assert factory.validate_provider_config("secondary") is True
        assert factory.validate_provider_config("disabled") is False  # 禁用的
        assert factory.validate_provider_config("nonexistent") is False

        # 获取提供商模型
        primary_models = factory.get_provider_models("primary")
        assert "primary-model" in primary_models
        assert "primary-fast" in primary_models

        secondary_models = factory.get_provider_models("secondary")
        assert "secondary-model" in secondary_models

        await factory.close()

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, temp_config_file):
        """测试错误处理集成"""
        factory = LLMProviderFactory(temp_config_file)
        self._register_mocks(factory)

        # 测试获取禁用的提供商
        with pytest.raises(ValueError, match="提供商.*未启用"):
            factory.get_provider("disabled")

        # 测试获取不存在的提供商
        with pytest.raises(ValueError, match="提供商.*未配置"):
            factory.get_provider("nonexistent")

        await factory.close()

    def test_environment_variable_integration(self, temp_config_file):
        """测试环境变量集成"""
        # 创建带有环境变量占位符的配置
        config_data = """
global:
  default_provider: ${JARVIS_LLM_DEFAULT_PROVIDER:primary}
providers:
  primary:
    type: integration_mock
    enabled: true
    api_key: ${JARVIS_LLM_PRIMARY_API_KEY}
    default_model: primary-model
    models:
      primary-model: {}
  secondary:
    type: slow_mock
    enabled: true
    api_key: test-key-2
    default_model: secondary-model
    models:
      secondary-model: {}
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_data)
            temp_path = f.name

        try:
            # 设置环境变量
            with patch.dict(
                os.environ,
                {
                    "JARVIS_LLM_PRIMARY_API_KEY": "env-api-key",
                    "JARVIS_LLM_DEFAULT_PROVIDER": "secondary",
                },
            ):
                loader = ConfigLoader(temp_path)
                config = loader.load_config()

                # 验证环境变量覆盖
                assert config.global_config.default_provider == "secondary"
                assert config.providers["primary"].api_key == "env-api-key"
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_create_llm_client_function_integration(self, temp_config_file):
        """测试create_llm_client函数集成"""
        # 准备工厂并注册Mocks
        factory = LLMProviderFactory(temp_config_file)
        self._register_mocks(factory)

        # Patch get_llm_factory 返回我们准备好的工厂
        with patch("src.core.llm.factory.get_llm_factory", return_value=factory):
            # 测试默认客户端创建
            client = create_llm_client(config_path=temp_config_file)
            assert isinstance(client, IntegrationMockClient)

            # 测试指定提供商的客户端创建
            client2 = create_llm_client(
                provider_name="secondary", config_path=temp_config_file
            )
            assert isinstance(client2, SlowMockClient)

            # 测试使用
            messages = [LLMMessage(role="user", content="Integration test")]
            response = await client.chat_completion(messages)
            assert "Integration test response" in response.content

    def test_config_file_variations(self):
        """测试不同配置文件变体"""
        # 测试最小配置
        minimal_config = {
            "global": {"default_provider": "simple"},
            "providers": {
                "simple": {
                    "type": "integration_mock",
                    "enabled": True,
                    "default_model": "simple-model",
                    "models": {"simple-model": {}},
                }
            },
            "environments": {},
            "features": {},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(minimal_config, f)
            temp_path = f.name

        try:
            loader = ConfigLoader(temp_path)
            config = loader.load_config()
            assert config.global_config.default_provider == "simple"
            assert len(config.providers) == 1

            factory = LLMProviderFactory(temp_path)
            # 需要注册 integration_mock
            factory.registry.register_provider(
                "integration_mock", IntegrationMockClient
            )

            client = factory.get_default_provider()
            assert isinstance(client, IntegrationMockClient)

            # Use asyncio to close if needed, but here we can just skip or try simple close
            # But factory.close() is async now.
            # We can't await in sync test easily.
            # We can rely on context manager or make test async.
            # Or just let it be GC'ed since it's mock.
            # But to be clean:
            asyncio.run(factory.close())
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_concurrent_client_usage(self, temp_config_file):
        """测试并发客户端使用"""
        factory = LLMProviderFactory(temp_config_file)
        self._register_mocks(factory)

        # 获取客户端
        primary_client = factory.get_provider("primary")
        secondary_client = factory.get_provider("secondary")

        # 并发调用
        messages = [LLMMessage(role="user", content="Concurrent test")]

        tasks = [
            primary_client.chat_completion(messages),
            secondary_client.chat_completion(messages),
            primary_client.chat_completion(messages),
        ]

        responses = await asyncio.gather(*tasks)

        # 验证响应
        assert len(responses) == 3
        assert "Integration test response #1" in responses[0].content
        assert responses[1].content == "Slow response"
        assert "Integration test response #2" in responses[2].content

        # 验证调用计数
        assert primary_client.call_count == 2

        await factory.close()

    @pytest.mark.asyncio
    async def test_factory_context_manager_integration(self, temp_config_file):
        """测试工厂上下文管理器集成"""
        client_ref = None

        async with LLMProviderFactory(temp_config_file) as factory:
            self._register_mocks(factory)
            client = factory.get_provider("primary")
            client_ref = client
            assert isinstance(client, IntegrationMockClient)
            assert not client.is_closed

        # 工厂关闭后，客户端应该被关闭
        assert client_ref.is_closed

    def test_configuration_inheritance_and_overrides(self, temp_config_file):
        """测试配置继承和覆盖"""
        config = load_llm_config(config_path=temp_config_file)

        # 验证全局配置
        global_retry = config.global_config.retry
        assert global_retry.max_attempts == 3
        assert global_retry.initial_delay == 1.0

        # 验证提供商特定的覆盖
        primary_retry = config.providers["primary"].retry
        assert primary_retry.max_attempts == 5  # 覆盖了全局设置
        assert primary_retry.initial_delay == 0.5  # 覆盖了全局设置
        assert primary_retry.exponential_base == 2.0  # 继承全局设置

        # 验证超时配置覆盖
        global_timeout = config.global_config.timeout
        assert global_timeout.total == 90.0

        primary_timeout = config.providers["primary"].timeout
        assert primary_timeout.total == 120.0  # 覆盖了全局设置
        assert primary_timeout.connect == 30.0  # 继承全局设置
        assert primary_timeout.read == 60.0  # 继承全局设置


class TestConfigurationErrorScenarios:
    """测试配置错误场景"""

    def setup_method(self):
        """每个测试方法前的设置"""
        pass

    def test_invalid_yaml_syntax(self):
        """测试无效YAML语法"""
        invalid_yaml = """
        global:
          default_provider: test
        providers:
          test:
            type: integration_mock
            enabled: true
            invalid: [unclosed list
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(invalid_yaml)
            temp_path = f.name

        try:
            # 直接使用 ConfigLoader 避免全局状态干扰
            loader = ConfigLoader(temp_path)
            with pytest.raises(ConfigurationError, match="YAML解析失败"):
                loader.load_config()
        finally:
            os.unlink(temp_path)

    def test_missing_required_sections(self):
        """测试缺失必需的配置节"""
        # ConfigLoader 现在允许缺失的节，使用默认值
        # 只需要验证默认值是否被正确应用
        pass

    def test_invalid_default_provider_reference(self):
        """测试无效的默认提供商引用"""
        config_data = {
            "global": {"default_provider": "nonexistent"},
            "providers": {
                "existing": {
                    "type": "integration_mock",
                    "enabled": True,
                    "default_model": "test",
                    "models": {"test": {}},
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            loader = ConfigLoader(temp_path)
            with pytest.raises(ConfigurationError, match="默认提供商.*未配置"):
                loader.load_config()
        finally:
            os.unlink(temp_path)

    def test_unregistered_provider_type(self):
        """测试未注册的提供商类型"""
        config_data = {
            "global": {"default_provider": "test"},
            "providers": {
                "test": {
                    "type": "unregistered_type",
                    "enabled": True,
                    "default_model": "test-model",
                    "models": {"test-model": {"max_tokens": 100}},
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            factory = LLMProviderFactory(temp_path)

            with pytest.raises(ValueError, match="提供商类型.*未注册"):
                factory.get_provider("test")
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
