"""
LLM工厂测试模块

测试LLM提供商注册、工厂创建、客户端管理等功能
"""

import pytest
import tempfile
import yaml
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional

from src.core.llm.factory import (
    LLMProviderRegistry, LLMProviderFactory, ProviderInfo, create_llm_client, ProviderType
)
from src.core.config.loader import (
    LLMConfig, GlobalConfig, ProviderConfig, FeatureConfig
)
from src.core.llm.client import BaseLLMClient, LLMProvider, LLMConfig as ClientLLMConfig
from src.core.common.exceptions import ConfigurationError, LLMError


class MockLLMClient(BaseLLMClient):
    """模拟LLM客户端"""
    
    def __init__(self, config: ClientLLMConfig):
        super().__init__(config)
        self.is_closed = False
        self._initialized = False
    
    async def initialize(self) -> bool:
        self._initialized = True
        return True

    async def generate_response(self, messages, **kwargs):
        from src.core.llm.client import LLMResponse
        return LLMResponse(
            content="Mock response",
            model="mock-model",
            usage={"total_tokens": 10},
            response_time=0.1
        )
        
    async def stream_response(self, messages, **kwargs):
        yield "Mock response"

    async def chat_completion(self, messages, **kwargs):
        """模拟聊天完成"""
        return await self.generate_response(messages, **kwargs)
    
    async def close(self):
        """关闭客户端"""
        self.is_closed = True


class AnotherMockClient(BaseLLMClient):
    """另一个模拟客户端"""
    
    def __init__(self, config: ClientLLMConfig):
        super().__init__(config)
        self._initialized = False
    
    async def initialize(self) -> bool:
        self._initialized = True
        return True

    async def generate_response(self, messages, **kwargs):
        from src.core.llm.client import LLMResponse
        return LLMResponse(
            content="Another mock response",
            model="another-mock-model",
            usage={"total_tokens": 15},
            response_time=0.2
        )
        
    async def stream_response(self, messages, **kwargs):
        yield "Another mock response"
    
    async def chat_completion(self, messages, **kwargs):
        return await self.generate_response(messages, **kwargs)
    
    async def close(self):
        pass


class TestLLMProviderRegistry:
    """测试LLM提供商注册表"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.registry = LLMProviderRegistry()
    
    def test_register_provider(self):
        """测试注册提供商"""
        self.registry.register_provider("mock", MockLLMClient)
        
        assert "mock" in self.registry._providers
        assert self.registry._providers["mock"] == MockLLMClient
    
    def test_register_duplicate_provider(self):
        """测试注册重复提供商"""
        self.registry.register_provider("mock", MockLLMClient)
        
        # 重复注册应该覆盖
        self.registry.register_provider("mock", AnotherMockClient)
        assert self.registry._providers["mock"] == AnotherMockClient
    
    def test_get_provider_existing(self):
        """测试获取存在的提供商"""
        self.registry.register_provider("mock", MockLLMClient)
        
        provider_class = self.registry.get_provider_class("mock")
        assert provider_class == MockLLMClient
    
    def test_get_provider_nonexistent(self):
        """测试获取不存在的提供商"""
        provider_class = self.registry.get_provider_class("nonexistent")
        assert provider_class is None
    
    def test_list_providers_empty(self):
        """测试列出空提供商列表"""
        # 默认会注册内置提供商，所以先清空
        self.registry._providers.clear()
        providers = self.registry.list_providers()
        assert providers == []
    
    def test_list_providers_with_data(self):
        """测试列出有数据的提供商列表"""
        self.registry._providers.clear()
        self.registry.register_provider("mock1", MockLLMClient)
        self.registry.register_provider("mock2", AnotherMockClient)
        
        providers = self.registry.list_providers()
        assert len(providers) == 2
        assert "mock1" in providers
        assert "mock2" in providers
    
    def test_is_registered_true(self):
        """测试提供商已注册"""
        self.registry.register_provider("mock", MockLLMClient)
        
        assert self.registry.is_provider_registered("mock") is True
    
    def test_is_registered_false(self):
        """测试提供商未注册"""
        assert self.registry.is_provider_registered("nonexistent") is False


class TestLLMProviderFactory:
    """测试LLM提供商工厂"""
    
    @pytest.fixture
    def sample_config_data(self) -> Dict[str, Any]:
        """示例配置数据"""
        return {
            "global": {
                "default_provider": "mock",
                "retry": {"max_attempts": 3},
                "timeout": {"total": 90.0},
                "logging": {"level": "INFO"}
            },
            "providers": {
                "mock": {
                    "type": "mock",
                    "enabled": True,
                    "default_model": "mock-model",
                    "models": {"mock-model": {}}
                },
                "another_mock": {
                    "type": "another_mock",
                    "enabled": True,
                    "default_model": "another-mock-model",
                    "models": {"another-mock-model": {}},
                    "api_key": "test-key"
                },
                "disabled_mock": {
                    "type": "mock",
                    "enabled": False,
                    "default_model": "disabled-model",
                    "models": {"disabled-model": {}}
                }
            },
            "environments": {},
            "features": {
                "streaming": {"enabled": True},
                "context": {"max_history": 10}
            }
        }
    
    @pytest.fixture
    def temp_config_file(self, sample_config_data):
        """创建临时配置文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config_data, f)
            temp_path = f.name
        
        yield temp_path
        
        import os
        os.unlink(temp_path)
    
    def setup_method(self):
        """每个测试方法前的设置"""
        # 这里的setup并不直接影响Factory实例，因为每次测试都会创建新的Factory
        pass
    
    def test_factory_init_with_config_path(self, temp_config_file):
        """测试使用配置文件路径初始化工厂"""
        factory = LLMProviderFactory(temp_config_file)
        
        assert factory.config_path == temp_config_file
        assert factory._config is None
        
        config = factory.load_config()
        assert config is not None
        assert config.global_config.default_provider == "mock"
    
    def test_factory_init_without_config_path(self):
        """测试不使用配置文件路径初始化工厂"""
        with patch('src.core.llm.factory.load_llm_config') as mock_load:
            mock_config = Mock()
            mock_config.global_config.default_provider = "mock"
            mock_load.return_value = mock_config
            
            factory = LLMProviderFactory()
            factory.load_config()
            
            assert factory.config_path is None
            assert factory._config == mock_config
            mock_load.assert_called_once()
    
    def test_load_config(self, temp_config_file):
        """测试加载配置"""
        factory = LLMProviderFactory(temp_config_file)
        config = factory.load_config()
        
        assert isinstance(config, LLMConfig)
        assert config.global_config.default_provider == "mock"
        assert "mock" in config.providers
    
    def test_get_default_provider(self, temp_config_file):
        """测试获取默认提供商"""
        factory = LLMProviderFactory(temp_config_file)
        # Register local MockLLMClient
        factory.registry.register_provider("mock", MockLLMClient)
        client = factory.get_default_provider()
        
        assert isinstance(client, MockLLMClient)
    
    def test_get_provider_by_name(self, temp_config_file):
        """测试按名称获取提供商"""
        factory = LLMProviderFactory(temp_config_file)
        # Register local MockLLMClient
        factory.registry.register_provider("mock", MockLLMClient)
        factory.registry.register_provider("another_mock", AnotherMockClient)
        
        # 获取启用的提供商
        client = factory.get_provider("mock")
        assert isinstance(client, MockLLMClient)
        
        client2 = factory.get_provider("another_mock")
        assert isinstance(client2, AnotherMockClient)
    
    def test_get_provider_disabled(self, temp_config_file):
        """测试获取禁用的提供商"""
        factory = LLMProviderFactory(temp_config_file)
        
        with pytest.raises(ValueError, match="提供商.*未启用"):
            factory.get_provider("disabled_mock")
    
    def test_get_provider_nonexistent(self, temp_config_file):
        """测试获取不存在的提供商"""
        factory = LLMProviderFactory(temp_config_file)
        
        with pytest.raises(ValueError, match="提供商.*未配置"):
            factory.get_provider("nonexistent")
    
    def test_get_provider_unregistered(self, temp_config_file, sample_config_data):
        """测试获取未注册的提供商"""
        # 添加未注册的提供商到配置
        sample_config_data["providers"]["unregistered"] = {
            "type": "unregistered",
            "enabled": True,
            "default_model": "test",
            "models": {"test": {}}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config_data, f)
            temp_path = f.name
        
        try:
            factory = LLMProviderFactory(temp_path)
            
            with pytest.raises(ValueError, match="提供商类型.*未注册"):
                factory.get_provider("unregistered")
        finally:
            import os
            os.unlink(temp_path)
    
    def test_list_available_providers(self, temp_config_file):
        """测试列出可用提供商"""
        factory = LLMProviderFactory(temp_config_file)
        # Register local MockLLMClient to match assertion
        factory.registry.register_provider("mock", MockLLMClient)
        factory.registry.register_provider("another_mock", AnotherMockClient)
        providers = factory.list_available_providers()
        
        assert len(providers) == 3  # mock, another_mock, disabled_mock
        
        provider_names = [p.name for p in providers]
        assert "mock" in provider_names
        assert "another_mock" in provider_names
        assert "disabled_mock" in provider_names
        
        # 检查提供商信息
        mock_provider = next(p for p in providers if p.name == "mock")
        assert mock_provider.type == ProviderType.MOCK
        assert mock_provider.enabled is True
        assert mock_provider.client_class == MockLLMClient
    
    def test_list_enabled_providers(self, temp_config_file):
        """测试列出启用的提供商"""
        factory = LLMProviderFactory(temp_config_file)
        factory.registry.register_provider("another_mock", AnotherMockClient)
        enabled_providers = factory.list_enabled_providers()
        
        assert len(enabled_providers) == 2  # mock, another_mock
        assert "mock" in enabled_providers
        assert "another_mock" in enabled_providers
        assert "disabled_mock" not in enabled_providers
    
    def test_get_provider_models(self, temp_config_file):
        """测试获取提供商模型"""
        factory = LLMProviderFactory(temp_config_file)
        factory.registry.register_provider("another_mock", AnotherMockClient)
        
        models = factory.get_provider_models("mock")
        assert "mock-model" in models
        
        models2 = factory.get_provider_models("another_mock")
        assert "another-mock-model" in models2
    
    def test_get_provider_models_nonexistent(self, temp_config_file):
        """测试获取不存在提供商的模型"""
        factory = LLMProviderFactory(temp_config_file)
        
        with pytest.raises(ValueError, match="提供商.*未配置"):
            factory.get_provider_models("nonexistent")
    
    def test_validate_provider_config_valid(self, temp_config_file):
        """测试验证有效的提供商配置"""
        factory = LLMProviderFactory(temp_config_file)
        factory.registry.register_provider("another_mock", AnotherMockClient)
        
        assert factory.validate_provider_config("mock") is True
        assert factory.validate_provider_config("another_mock") is True
    
    def test_validate_provider_config_invalid(self, temp_config_file):
        """测试验证无效的提供商配置"""
        factory = LLMProviderFactory(temp_config_file)
        
        assert factory.validate_provider_config("nonexistent") is False
        assert factory.validate_provider_config("disabled_mock") is False
    
    def test_client_pooling(self, temp_config_file):
        """测试客户端池化"""
        factory = LLMProviderFactory(temp_config_file)
        factory.registry.register_provider("mock", MockLLMClient)
        
        # 第一次获取
        client1 = factory.get_provider("mock")
        
        # 第二次获取应该是同一个实例
        client2 = factory.get_provider("mock")
        
        assert client1 is client2
    
    @pytest.mark.asyncio
    async def test_close_factory(self, temp_config_file):
        """测试关闭工厂"""
        factory = LLMProviderFactory(temp_config_file)
        factory.registry.register_provider("mock", MockLLMClient)
        factory.registry.register_provider("another_mock", AnotherMockClient)
        
        # 创建一些客户端
        client1 = factory.get_provider("mock")
        client2 = factory.get_provider("another_mock")
        
        # 关闭工厂
        await factory.close()
        
        # 验证客户端被关闭
        assert client1.is_closed is True
        # AnotherMockClient没有is_closed属性，但close方法应该被调用
    
    @pytest.mark.asyncio
    async def test_context_manager(self, temp_config_file):
        """测试上下文管理器"""
        async with LLMProviderFactory(temp_config_file) as factory:
            factory.registry.register_provider("mock", MockLLMClient)
            client = factory.get_provider("mock")
            assert isinstance(client, MockLLMClient)
        
        # 工厂应该自动关闭
        assert client.is_closed is True


class TestCreateLLMClient:
    """测试创建LLM客户端函数"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        import src.core.llm.factory
        src.core.llm.factory._factory = None
    
    @patch('src.core.llm.factory.LLMProviderFactory')
    def test_create_llm_client_default(self, mock_factory_class):
        """测试创建默认LLM客户端"""
        mock_factory = Mock()
        mock_client = Mock()
        mock_factory.get_default_provider.return_value = mock_client
        mock_factory_class.return_value = mock_factory
        
        result = create_llm_client()
        
        mock_factory_class.assert_called_once_with(None)
        mock_factory.get_default_provider.assert_called_once()
        assert result == mock_client
    
    @patch('src.core.llm.factory.LLMProviderFactory')
    def test_create_llm_client_with_provider(self, mock_factory_class):
        """测试创建指定提供商的LLM客户端"""
        mock_factory = Mock()
        mock_client = Mock()
        mock_factory.get_provider.return_value = mock_client
        mock_factory_class.return_value = mock_factory
        
        result = create_llm_client(provider_name="mock")
        
        mock_factory_class.assert_called_once_with(None)
        # 允许更多参数或仅检查第一个参数
        assert mock_factory.get_provider.call_args[0][0] == "mock"
        assert result == mock_client
    
    @patch('src.core.llm.factory.LLMProviderFactory')
    def test_create_llm_client_with_config_path(self, mock_factory_class):
        """测试使用配置路径创建LLM客户端"""
        mock_factory = Mock()
        mock_client = Mock()
        mock_factory.get_default_provider.return_value = mock_client
        mock_factory_class.return_value = mock_factory
        
        result = create_llm_client(config_path="test.yaml")
        
        mock_factory_class.assert_called_once_with("test.yaml")
        mock_factory.get_default_provider.assert_called_once()
        assert result == mock_client


class TestProviderInfo:
    """测试提供商信息类"""
    
    def test_provider_info_creation(self):
        """测试提供商信息创建"""
        config = ProviderConfig(enabled=True)
        
        info = ProviderInfo(
            name="test_provider",
            type=ProviderType.MOCK,
            client_class=MockLLMClient,
            enabled=True,
            config=config
        )
        
        assert info.name == "test_provider"
        assert info.type == ProviderType.MOCK
        assert info.enabled is True
        assert info.client_class == MockLLMClient
        assert info.config == config


class TestLLMFactoryEdgeCases:
    """测试LLM工厂边缘情况"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        pass
    
    def test_factory_with_empty_config(self):
        """测试空配置的工厂"""
        config_data = {
            "global": {"default_provider": "mock"},
            "providers": {},
            "environments": {},
            "features": {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            factory = LLMProviderFactory(temp_path)
            # Need to register mock because factory is new
            factory.registry.register_provider("mock", MockLLMClient)
            
            with pytest.raises(ConfigurationError):
                factory.get_default_provider()
        finally:
            import os
            os.unlink(temp_path)
    
    def test_factory_with_invalid_default_provider(self):
        """测试无效默认提供商的工厂"""
        config_data = {
            "global": {"default_provider": "nonexistent"},
            "providers": {
                "mock": {
                    "type": "mock",
                    "enabled": True
                }
            },
            "environments": {},
            "features": {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            factory = LLMProviderFactory(temp_path)
            # Register mock
            factory.registry.register_provider("mock", MockLLMClient)
            
            with pytest.raises(ConfigurationError):
                factory.get_default_provider()
        finally:
            import os
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_multiple_factory_instances(self):
        """测试多个工厂实例"""
        config_data = {
            "global": {"default_provider": "mock"},
            "providers": {
                "mock": {
                    "type": "mock",
                    "enabled": True,
                    "default_model": "mock-model",
                    "models": {"mock-model": {}}
                }
            },
            "environments": {},
            "features": {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            factory1 = LLMProviderFactory(temp_path)
            factory2 = LLMProviderFactory(temp_path)
            
            factory1.registry.register_provider("mock", MockLLMClient)
            factory2.registry.register_provider("mock", MockLLMClient)
            
            client1 = factory1.get_provider("mock")
            client2 = factory2.get_provider("mock")
            
            # 不同工厂实例应该创建不同的客户端
            assert client1 is not client2
            
            await factory1.close()
            await factory2.close()
        finally:
            import os
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
