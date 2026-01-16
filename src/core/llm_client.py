"""
LLM客户端模块

提供统一的LLM接口，支持多种LLM提供商。
"""

import logging
import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from abc import ABC, abstractmethod

from .llm_types import LLMProvider, LLMMessage, LLMResponse, LLMConfig
from .llm_utils.cache_manager import CacheManager
from .llm_utils.metrics_collector import global_metrics


class BaseLLMClient(ABC):
    """LLM客户端基类"""
    
    def __init__(self, config: LLMConfig):
        """
        初始化LLM客户端
        
        Args:
            config: LLM配置
        """
        self.config = config
        provider_name = config.provider.value if isinstance(config.provider, LLMProvider) else str(config.provider)
        self.logger = logging.getLogger(f"llm.{provider_name}")
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        初始化客户端
        
        Returns:
            是否初始化成功
        """
        pass
    
    @abstractmethod
    async def generate_response(
        self, 
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """
        生成响应
        
        Args:
            messages: 消息列表
            **kwargs: 额外参数
            
        Returns:
            LLM响应
        """
        pass
    
    @abstractmethod
    async def stream_response(
        self, 
        messages: List[LLMMessage],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        流式生成响应
        
        Args:
            messages: 消息列表
            **kwargs: 额外参数
            
        Yields:
            响应片段
        """
        pass
    
    async def close(self) -> None:
        """关闭客户端"""
        self._initialized = False
        self.logger.info("LLM客户端已关闭")


# MockLLMClient 现已移动到 src/core/llm_providers/mock_client.py
try:
    from .llm_providers.mock_client import MockLLMClient
except ImportError:
    MockLLMClient = None

# 导入真实的OpenAI客户端实现
try:
    from .llm_providers.openai_client import OpenAILLMClient
except ImportError:
    # 如果导入失败，使用占位符实现
    class OpenAILLMClient(BaseLLMClient):
        """OpenAI LLM客户端（占位符实现）"""
        
        def __init__(self, config: LLMConfig):
            super().__init__(config)
            self.logger.warning("OpenAI客户端实现未找到，使用Mock客户端")
        
        async def initialize(self) -> bool:
            """初始化OpenAI客户端"""
            self.logger.warning("OpenAI客户端实现未找到，使用Mock客户端")
            from .llm_providers.mock_client import MockLLMClient
            mock_client = MockLLMClient(self.config)
            await mock_client.initialize()
            self._mock_client = mock_client
            self._initialized = True
            return True
        
        async def generate_response(
            self, 
            messages: List[LLMMessage],
            **kwargs
        ) -> LLMResponse:
            """生成响应（使用Mock客户端）"""
            if not self._initialized:
                raise RuntimeError("OpenAI客户端未初始化")
            
            from .llm_providers.mock_client import MockLLMClient
            mock_client = getattr(self, '_mock_client', MockLLMClient(self.config))
            return await mock_client.generate_response(messages, **kwargs)
        
        async def stream_response(
            self, 
            messages: List[LLMMessage],
            **kwargs
        ) -> AsyncGenerator[str, None]:
            """流式生成响应（使用Mock客户端）"""
            if not self._initialized:
                raise RuntimeError("OpenAI客户端未初始化")
            
            from .llm_providers.mock_client import MockLLMClient
            mock_client = getattr(self, '_mock_client', MockLLMClient(self.config))
            async for chunk in mock_client.stream_response(messages, **kwargs):
                yield chunk


class LLMClientWrapper(BaseLLMClient):
    """LLM客户端包装器"""
    def __init__(self, client: BaseLLMClient):
        self.client = client
        self.config = client.config
        self.logger = client.logger
        self._initialized = getattr(client, '_initialized', False)
        
    async def initialize(self) -> bool:
        result = await self.client.initialize()
        self._initialized = self.client._initialized
        return result
        
    async def generate_response(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        return await self.client.generate_response(messages, **kwargs)
        
    async def stream_response(self, messages: List[LLMMessage], **kwargs) -> AsyncGenerator[str, None]:
        async for chunk in self.client.stream_response(messages, **kwargs):
            yield chunk
            
    async def close(self) -> None:
        await self.client.close()

class CachedLLMClient(LLMClientWrapper):
    """带缓存的LLM客户端包装器"""
    def __init__(self, client: BaseLLMClient, cache_manager: Optional[CacheManager] = None):
        super().__init__(client)
        self._cache_manager = cache_manager or CacheManager()
        
    async def generate_response(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        # 检查缓存
        cached = self._cache_manager.get(messages, **kwargs)
        if cached:
            return cached
            
        # 生成响应
        response = await self.client.generate_response(messages, **kwargs)
        
        # 设置缓存
        self._cache_manager.set(messages, response, **kwargs)
        return response

class MonitoredLLMClient(LLMClientWrapper):
    """带监控的LLM客户端包装器"""
    def __init__(self, client: BaseLLMClient):
        super().__init__(client)
        
    async def generate_response(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        start_time = time.time()
        success = False
        tokens = {}
        try:
            response = await self.client.generate_response(messages, **kwargs)
            success = True
            tokens = response.usage
            return response
        finally:
            latency = time.time() - start_time
            global_metrics.record_request(success, latency, tokens)


class LLMClientFactory:
    """LLM客户端工厂"""
    
    @staticmethod
    def create_client(config: LLMConfig) -> BaseLLMClient:
        """
        创建LLM客户端
        
        Args:
            config: LLM配置
            
        Returns:
            LLM客户端实例
        """
        # Create client
        client = None
        if config.provider == LLMProvider.MOCK:
            client = MockLLMClient(config)
        elif config.provider == LLMProvider.OPENAI:
            client = OpenAILLMClient(config)
        elif config.provider == LLMProvider.LITELLM:
            try:
                from .llm_providers.litellm_client import LiteLLMClient
                client = LiteLLMClient(config)
            except ImportError:
                raise ImportError("LiteLLMClient不可用，请确保已安装依赖")
        elif config.provider == LLMProvider.ANTHROPIC:
            try:
                from .llm_providers.litellm_client import LiteLLMClient
                client = LiteLLMClient(config)
            except ImportError:
                raise ImportError("LiteLLMClient不可用，请确保已安装依赖")
        elif config.provider == LLMProvider.AZURE_OPENAI:
            # 可以在这里添加Azure OpenAI客户端
            raise NotImplementedError("Azure OpenAI客户端暂未实现")
        elif config.provider == LLMProvider.OLLAMA:
            # 可以在这里添加Ollama客户端
            raise NotImplementedError("Ollama客户端暂未实现")
        else:
            raise ValueError(f"不支持的LLM提供商: {config.provider}")
            
        # Apply wrappers based on configuration
        # Assuming config has flags for cache and monitoring
        # Since LLMConfig might not have these flags directly, we can check extra_params or assume defaults
        # For now, let's enable monitoring by default and cache if configured
        
        # Enable monitoring
        client = MonitoredLLMClient(client)
        
        # Enable caching if configured (e.g. via extra_params or if we add fields to LLMConfig)
        # Using a simple heuristic for now
        if config.extra_params.get("cache_enabled", True):
            client = CachedLLMClient(client)
            
        return client


class LLMManager:
    """LLM管理器，负责管理多个LLM客户端"""
    
    def __init__(self, default_config: Optional[LLMConfig] = None):
        """
        初始化LLM管理器
        
        Args:
            default_config: 可选的默认LLM配置，如果提供将自动创建默认客户端
        """
        self.clients: Dict[str, BaseLLMClient] = {}
        self.default_client: Optional[str] = None
        self.fallback_providers: List[str] = []
        self.logger = logging.getLogger("llm.manager")
        self._default_config = default_config
    
    def set_fallback_providers(self, providers: List[str]) -> None:
        """
        设置降级提供商列表
        
        Args:
            providers: 提供商名称列表
        """
        self.fallback_providers = providers
        self.logger.info(f"已设置降级提供商: {providers}")
    
    async def add_client(self, name: str, config: LLMConfig) -> bool:
        """
        添加LLM客户端
        
        Args:
            name: 客户端名称
            config: LLM配置
            
        Returns:
            是否添加成功
        """
        try:
            client = LLMClientFactory.create_client(config)
            success = await client.initialize()
            
            if success:
                self.clients[name] = client
                if self.default_client is None:
                    self.default_client = name
                self.logger.info(f"LLM客户端 '{name}' 添加成功")
                return True
            else:
                self.logger.error(f"LLM客户端 '{name}' 初始化失败")
                return False
                
        except Exception as e:
            self.logger.error(f"添加LLM客户端 '{name}' 失败: {e}")
            return False
    
    async def initialize_default_client(self) -> bool:
        """
        初始化默认客户端
        
        Returns:
            是否初始化成功
        """
        if self._default_config is None:
            self.logger.warning("未提供默认配置，无法初始化默认客户端")
            return False
            
        return await self.add_client("default", self._default_config)
    
    async def generate_response(
        self, 
        messages: List[LLMMessage],
        client_name: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        生成响应（支持自动降级）
        
        Args:
            messages: 消息列表
            client_name: 客户端名称，如果为None则使用默认客户端
            **kwargs: 额外参数
            
        Returns:
            LLM响应
        """
        client_name = client_name or self.default_client
        
        if not client_name or client_name not in self.clients:
            raise ValueError(f"LLM客户端 '{client_name}' 不存在")
        
        try:
            client = self.clients[client_name]
            return await client.generate_response(messages, **kwargs)
        except Exception as e:
            # 如果是默认客户端且配置了降级提供商，尝试降级
            if client_name == self.default_client and self.fallback_providers:
                self.logger.warning(f"主提供商 '{client_name}' 失败: {e}，尝试降级...")
                
                for fallback_name in self.fallback_providers:
                    if fallback_name in self.clients:
                        try:
                            self.logger.info(f"尝试降级提供商: '{fallback_name}'")
                            fallback_client = self.clients[fallback_name]
                            return await fallback_client.generate_response(messages, **kwargs)
                        except Exception as fallback_e:
                            self.logger.warning(f"降级提供商 '{fallback_name}' 失败: {fallback_e}")
                            continue
                    else:
                        self.logger.warning(f"降级提供商 '{fallback_name}' 未初始化")
            
            # 如果所有尝试都失败，抛出原始异常
            raise e
    
    async def stream_response(
        self, 
        messages: List[LLMMessage],
        client_name: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        流式生成响应
        
        Args:
            messages: 消息列表
            client_name: 客户端名称，如果为None则使用默认客户端
            **kwargs: 额外参数
            
        Yields:
            响应片段
        """
        client_name = client_name or self.default_client
        
        if not client_name or client_name not in self.clients:
            raise ValueError(f"LLM客户端 '{client_name}' 不存在")
        
        client = self.clients[client_name]
        async for chunk in client.stream_response(messages, **kwargs):
            yield chunk
    
    def get_available_clients(self) -> List[str]:
        """
        获取可用的客户端列表
        
        Returns:
            客户端名称列表
        """
        return list(self.clients.keys())
    
    async def close_all(self) -> None:
        """关闭所有客户端"""
        for name, client in self.clients.items():
            try:
                await client.close()
                self.logger.info(f"LLM客户端 '{name}' 已关闭")
            except Exception as e:
                self.logger.error(f"关闭LLM客户端 '{name}' 失败: {e}")
        
        self.clients.clear()
        self.default_client = None
        self.logger.info("所有LLM客户端已关闭")