"""
LLM提供商工厂模式

该模块实现了基于配置的LLM提供商工厂，支持：
- 配置驱动的提供商选择
- 动态提供商创建和管理
- 提供商池管理和复用
- 配置热重载支持
- 多环境支持
"""

import logging
import asyncio
from typing import Dict, Optional, Type, Any, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from .config_loader import LLMConfig, ProviderConfig, load_llm_config
from .llm_client import BaseLLMClient, LLMProvider
from .llm_providers.openai_client import OpenAILLMClient
from .llm_providers.mock_client import MockLLMClient

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """支持的提供商类型"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    OLLAMA = "ollama"
    MOCK = "mock"


@dataclass
class ProviderInfo:
    """提供商信息"""
    name: str
    type: ProviderType
    client_class: Type[BaseLLMClient]
    enabled: bool
    config: ProviderConfig


class LLMProviderRegistry:
    """LLM提供商注册表"""
    
    def __init__(self):
        """初始化提供商注册表"""
        self._providers: Dict[str, Type[BaseLLMClient]] = {}
        self._register_builtin_providers()
    
    def _register_builtin_providers(self) -> None:
        """注册内置提供商"""
        self.register_provider("openai", OpenAILLMClient)
        self.register_provider("mock", MockLLMClient)
        
        if LiteLLMClient:
            self.register_provider("litellm", LiteLLMClient)
            
        # TODO: 添加其他提供商
        # self.register_provider("anthropic", AnthropicLLMClient)
        # self.register_provider("azure_openai", AzureOpenAILLMClient)
        # self.register_provider("ollama", OllamaLLMClient)
        
        logger.info(f"已注册 {len(self._providers)} 个内置提供商")
    
    def register_provider(self, name: str, client_class: Type[BaseLLMClient]) -> None:
        """
        注册LLM提供商
        
        Args:
            name: 提供商名称
            client_class: 客户端类
        """
        if not issubclass(client_class, BaseLLMClient):
            raise ValueError(f"客户端类必须继承自BaseLLMClient: {client_class}")
        
        self._providers[name] = client_class
        logger.debug(f"注册提供商: {name} -> {client_class.__name__}")
    
    def unregister_provider(self, name: str) -> None:
        """
        注销LLM提供商
        
        Args:
            name: 提供商名称
        """
        if name in self._providers:
            del self._providers[name]
            logger.debug(f"注销提供商: {name}")
    
    def get_provider_class(self, name: str) -> Optional[Type[BaseLLMClient]]:
        """
        获取提供商客户端类
        
        Args:
            name: 提供商名称
            
        Returns:
            客户端类或None
        """
        return self._providers.get(name)
    
    def list_providers(self) -> List[str]:
        """获取所有已注册的提供商名称"""
        return list(self._providers.keys())
    
    def is_provider_registered(self, name: str) -> bool:
        """检查提供商是否已注册"""
        return name in self._providers


class LLMProviderFactory:
    """LLM提供商工厂"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化LLM提供商工厂
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.registry = LLMProviderRegistry()
        self._config: Optional[LLMConfig] = None
        self._client_pool: Dict[str, BaseLLMClient] = {}
        self._initialized_providers: Dict[str, bool] = {}
        
        logger.info("LLM提供商工厂初始化完成")
    
    def load_config(self, environment: Optional[str] = None, 
                   reload: bool = False) -> LLMConfig:
        """
        加载配置
        
        Args:
            environment: 环境名称
            reload: 是否强制重新加载
            
        Returns:
            LLMConfig: 配置对象
        """
        if reload or self._config is None:
            self._config = load_llm_config(
                environment=environment,
                config_path=self.config_path,
                reload=reload
            )
            
            # 清理客户端池（配置变更时）
            if reload:
                self._cleanup_client_pool()
            
            logger.info(f"配置加载完成，环境: {self._config.environment.value}")
        
        return self._config
    
    def get_default_provider(self, environment: Optional[str] = None) -> BaseLLMClient:
        """
        获取默认LLM提供商客户端
        
        Args:
            environment: 环境名称
            
        Returns:
            BaseLLMClient: LLM客户端实例
        """
        config = self.load_config(environment)
        default_provider_name = config.global_config.default_provider
        
        return self.get_provider(default_provider_name, environment)
    
    def get_provider(self, provider_name: str, 
                    environment: Optional[str] = None) -> BaseLLMClient:
        """
        获取指定的LLM提供商客户端
        
        Args:
            provider_name: 提供商名称
            environment: 环境名称
            
        Returns:
            BaseLLMClient: LLM客户端实例
            
        Raises:
            ValueError: 提供商不存在或未启用
            RuntimeError: 提供商初始化失败
        """
        config = self.load_config(environment)
        
        # 检查提供商是否存在
        if provider_name not in config.providers:
            raise ValueError(f"提供商 '{provider_name}' 未配置")
        
        provider_config = config.providers[provider_name]
        
        # 检查提供商是否启用
        if not provider_config.enabled:
            raise ValueError(f"提供商 '{provider_name}' 未启用")
        
        # 从客户端池获取或创建客户端
        cache_key = f"{provider_name}_{config.environment.value}"
        if cache_key in self._client_pool:
            return self._client_pool[cache_key]
        
        # 创建新的客户端实例
        client = self._create_provider_client(provider_name, provider_config, config)
        self._client_pool[cache_key] = client
        
        logger.info(f"创建提供商客户端: {provider_name}")
        return client
    
    def _create_provider_client(self, provider_name: str, 
                               provider_config: ProviderConfig,
                               global_config: LLMConfig) -> BaseLLMClient:
        """
        创建提供商客户端实例
        
        Args:
            provider_name: 提供商名称
            provider_config: 提供商配置
            global_config: 全局配置
            
        Returns:
            BaseLLMClient: 客户端实例
            
        Raises:
            ValueError: 提供商未注册
            RuntimeError: 客户端创建失败
        """
        # 确定提供商类型
        provider_type = provider_config.type or provider_name
        
        # 获取客户端类
        client_class = self.registry.get_provider_class(provider_type)
        if client_class is None:
            # 尝试直接使用名称
            client_class = self.registry.get_provider_class(provider_name)
            
        if client_class is None:
            raise ValueError(f"提供商类型 '{provider_type}' (名称: '{provider_name}') 未注册")
        
        try:
            # 准备配置参数
            config_dict = self._prepare_provider_config(
                provider_name, provider_config, global_config
            )
            
            # 转换为LLMConfig对象
            from .llm_client import LLMConfig as ClientLLMConfig
            client_config = ClientLLMConfig(**config_dict)
            
            # 创建客户端实例
            client = client_class(client_config)
            
            # 初始化客户端
            # 注意: BaseLLMClient.initialize() 通常不接受参数
            # 如果需要传递额外初始化参数，应该通过构造函数或专门的configure方法
            if asyncio.iscoroutinefunction(client.initialize):
                # 如果是异步初始化，此时无法await (因为这是同步方法)
                # 这可能是一个问题，但目前的架构似乎假设initialize是异步的
                # 工厂方法是同步的，所以无法await
                # 我们只能依靠客户端在首次调用时自动初始化，或者更改工厂方法为异步
                # 现在的实现中，OpenAILLMClient.initialize 是异步的
                pass
            else:
                client.initialize()
            
            # 标记为已初始化
            self._initialized_providers[provider_name] = True
            
            return client
            
        except Exception as e:
            logger.error(f"创建提供商客户端失败 '{provider_name}': {e}")
            raise RuntimeError(f"创建提供商客户端失败 '{provider_name}': {e}") from e
    
    def _prepare_provider_config(self, provider_name: str,
                                provider_config: ProviderConfig,
                                global_config: LLMConfig) -> Dict[str, Any]:
        """
        准备提供商配置参数
        
        Args:
            provider_name: 提供商名称
            provider_config: 提供商配置
            global_config: 全局配置
            
        Returns:
            Dict[str, Any]: 配置字典
        """
        config_dict = {
            "provider": provider_name,
            "api_key": provider_config.api_key,
            "base_url": provider_config.base_url,
            "default_model": provider_config.default_model,
            "models": {},
            "timeout": {
                "connect": provider_config.timeout.connect,
                "read": provider_config.timeout.read,
                "total": provider_config.timeout.total
            },
            "retry": {
                "max_attempts": provider_config.retry.max_attempts,
                "initial_delay": provider_config.retry.initial_delay,
                "max_delay": provider_config.retry.max_delay,
                "exponential_base": provider_config.retry.exponential_base
            }
        }
        
        # 添加模型配置
        for model_name, model_config in provider_config.models.items():
            config_dict["models"][model_name] = {
                "max_tokens": model_config.max_tokens,
                "temperature": model_config.temperature,
                "top_p": model_config.top_p,
                "frequency_penalty": model_config.frequency_penalty,
                "presence_penalty": model_config.presence_penalty
            }
            
            # 添加特定提供商的配置
            if model_config.deployment_name:  # Azure OpenAI
                config_dict["models"][model_name]["deployment_name"] = model_config.deployment_name
            
            if model_config.response_delay > 0:  # Mock
                config_dict["models"][model_name]["response_delay"] = model_config.response_delay
        
        # 添加特定提供商的配置
        if provider_config.azure_endpoint:
            config_dict["azure_endpoint"] = provider_config.azure_endpoint
        
        if provider_config.api_version:
            config_dict["api_version"] = provider_config.api_version
        
        if provider_config.organization:
            config_dict["organization"] = provider_config.organization
        
        if provider_config.rate_limit:
            config_dict["rate_limit"] = {
                "requests_per_minute": provider_config.rate_limit.requests_per_minute,
                "tokens_per_minute": provider_config.rate_limit.tokens_per_minute
            }
        
        return config_dict
    
    def list_available_providers(self, environment: Optional[str] = None) -> List[ProviderInfo]:
        """
        获取所有可用的提供商信息
        
        Args:
            environment: 环境名称
            
        Returns:
            List[ProviderInfo]: 提供商信息列表
        """
        config = self.load_config(environment)
        providers = []
        
        for provider_name, provider_config in config.providers.items():
            provider_type_name = provider_config.type or provider_name
            client_class = self.registry.get_provider_class(provider_type_name)
            
            if client_class:
                try:
                    provider_type = ProviderType(provider_type_name)
                except ValueError:
                    # 自定义提供商，使用名称作为类型
                    provider_type = provider_type_name
                
                providers.append(ProviderInfo(
                    name=provider_name,
                    type=provider_type,
                    client_class=client_class,
                    enabled=provider_config.enabled,
                    config=provider_config
                ))
        
        return providers
    
    def list_enabled_providers(self, environment: Optional[str] = None) -> List[str]:
        """
        获取所有启用的提供商名称
        
        Args:
            environment: 环境名称
            
        Returns:
            List[str]: 启用的提供商名称列表
        """
        config = self.load_config(environment)
        return [
            name for name, provider_config in config.providers.items()
            if provider_config.enabled and self.registry.is_provider_registered(provider_config.type or name)
        ]
    
    def get_provider_models(self, provider_name: str, 
                           environment: Optional[str] = None) -> List[str]:
        """
        获取提供商支持的模型列表
        
        Args:
            provider_name: 提供商名称
            environment: 环境名称
            
        Returns:
            List[str]: 模型名称列表
        """
        config = self.load_config(environment)
        
        if provider_name not in config.providers:
            raise ValueError(f"提供商 '{provider_name}' 未配置")
        
        provider_config = config.providers[provider_name]
        return list(provider_config.models.keys())
    
    def validate_provider_config(self, provider_name: str,
                                environment: Optional[str] = None) -> bool:
        """
        验证提供商配置
        
        Args:
            provider_name: 提供商名称
            environment: 环境名称
            
        Returns:
            bool: 配置是否有效
        """
        try:
            config = self.load_config(environment)
            
            if provider_name not in config.providers:
                logger.error(f"提供商 '{provider_name}' 未配置")
                return False
            
            provider_config = config.providers[provider_name]
            
            if not provider_config.enabled:
                logger.warning(f"提供商 '{provider_name}' 未启用")
                return False
            
            provider_type = provider_config.type or provider_name
            if not self.registry.is_provider_registered(provider_type):
                logger.error(f"提供商类型 '{provider_type}' (用于 '{provider_name}') 未注册")
                return False
            
            # 验证必要的配置项
            if provider_type != "mock" and not provider_config.api_key:
                logger.error(f"提供商 '{provider_name}' 缺少API密钥")
                return False
            
            if not provider_config.models:
                logger.error(f"提供商 '{provider_name}' 未配置模型")
                return False
            
            if provider_config.default_model not in provider_config.models:
                logger.error(f"提供商 '{provider_name}' 的默认模型未配置")
                return False
            
            logger.info(f"提供商 '{provider_name}' 配置验证通过")
            return True
            
        except Exception as e:
            logger.error(f"验证提供商配置失败 '{provider_name}': {e}")
            return False
    
    def reload_config(self, environment: Optional[str] = None) -> None:
        """
        重新加载配置
        
        Args:
            environment: 环境名称
        """
        logger.info("重新加载配置...")
        self.load_config(environment, reload=True)
        logger.info("配置重新加载完成")
    
    def _cleanup_client_pool(self) -> None:
        """清理客户端池"""
        for client in self._client_pool.values():
            try:
                if hasattr(client, 'close'):
                    client.close()
            except Exception as e:
                logger.warning(f"关闭客户端时出错: {e}")
        
        self._client_pool.clear()
        self._initialized_providers.clear()
        logger.debug("客户端池已清理")
    
    def close(self) -> None:
        """关闭工厂并清理资源"""
        logger.info("关闭LLM提供商工厂...")
        self._cleanup_client_pool()
        logger.info("LLM提供商工厂已关闭")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


# 全局工厂实例
_factory: Optional[LLMProviderFactory] = None


def get_llm_factory(config_path: Optional[str] = None) -> LLMProviderFactory:
    """
    获取全局LLM提供商工厂实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        LLMProviderFactory: 工厂实例
    """
    global _factory
    if _factory is None or config_path is not None:
        _factory = LLMProviderFactory(config_path)
    return _factory


def create_llm_client(provider_name: Optional[str] = None,
                     environment: Optional[str] = None,
                     config_path: Optional[str] = None) -> BaseLLMClient:
    """
    便捷函数：创建LLM客户端
    
    Args:
        provider_name: 提供商名称，None表示使用默认提供商
        environment: 环境名称
        config_path: 配置文件路径
        
    Returns:
        BaseLLMClient: LLM客户端实例
    """
    factory = get_llm_factory(config_path)
    
    if provider_name is None:
        return factory.get_default_provider(environment)
    else:
        return factory.get_provider(provider_name, environment)
