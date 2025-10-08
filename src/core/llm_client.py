"""
LLM客户端模块

提供统一的LLM接口，支持多种LLM提供商。
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class LLMProvider(Enum):
    """支持的LLM提供商"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    OLLAMA = "ollama"
    MOCK = "mock"  # 用于测试


@dataclass
class LLMMessage:
    """LLM消息格式"""
    role: str  # "system", "user", "assistant"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """LLM响应格式"""
    content: str
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = ""
    finish_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfig:
    """LLM配置"""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30
    retry_attempts: int = 3
    stream: bool = False
    extra_params: Dict[str, Any] = field(default_factory=dict)


class BaseLLMClient(ABC):
    """LLM客户端基类"""
    
    def __init__(self, config: LLMConfig):
        """
        初始化LLM客户端
        
        Args:
            config: LLM配置
        """
        self.config = config
        self.logger = logging.getLogger(f"llm.{config.provider.value}")
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


class MockLLMClient(BaseLLMClient):
    """模拟LLM客户端，用于测试"""
    
    async def initialize(self) -> bool:
        """初始化模拟客户端"""
        self._initialized = True
        self.logger.info("模拟LLM客户端初始化完成")
        return True
    
    async def generate_response(
        self, 
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """生成模拟响应"""
        if not self._initialized:
            raise RuntimeError("LLM客户端未初始化")
        
        # 模拟处理延迟
        await asyncio.sleep(0.1)
        
        # 根据最后一条用户消息生成响应
        last_user_message = ""
        for msg in reversed(messages):
            if msg.role == "user":
                last_user_message = msg.content
                break
        
        # 生成智能的模拟响应
        response_content = self._generate_mock_response(last_user_message, messages)
        
        return LLMResponse(
            content=response_content,
            usage={"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
            model="mock-gpt-4",
            finish_reason="stop",
            metadata={"mock": True}
        )
    
    async def stream_response(
        self, 
        messages: List[LLMMessage],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """流式生成模拟响应"""
        if not self._initialized:
            raise RuntimeError("LLM客户端未初始化")
        
        # 获取完整响应
        response = await self.generate_response(messages, **kwargs)
        
        # 模拟流式输出
        words = response.content.split()
        for word in words:
            await asyncio.sleep(0.05)  # 模拟网络延迟
            yield word + " "
    
    def _generate_mock_response(self, user_input: str, messages: List[LLMMessage]) -> str:
        """
        生成智能的模拟响应
        
        Args:
            user_input: 用户输入
            messages: 消息历史
            
        Returns:
            模拟响应内容
        """
        user_input_lower = user_input.lower()
        
        # 问候语
        if any(greeting in user_input_lower for greeting in ["hello", "hi", "你好", "嗨"]):
            return "Hello! I'm Jarvis, your AI assistant. How can I help you today?"
        
        # 告别语
        elif any(goodbye in user_input_lower for goodbye in ["bye", "goodbye", "再见", "拜拜"]):
            return "Goodbye! It was nice talking with you. Have a great day!"
        
        # 询问能力
        elif any(capability in user_input_lower for capability in ["what can you do", "capabilities", "你能做什么", "功能"]):
            return ("I'm an AI assistant powered by ARK engine. I can help you with various tasks including "
                   "answering questions, providing information, using tools, and having conversations. "
                   "What would you like me to help you with?")
        
        # 询问名字
        elif any(name_q in user_input_lower for name_q in ["what's your name", "who are you", "你是谁", "你叫什么"]):
            return "I'm Jarvis, an AI assistant built with the ARK (Adaptive Reasoning Kernel) engine. Nice to meet you!"
        
        # 感谢
        elif any(thanks in user_input_lower for thanks in ["thank", "thanks", "谢谢", "感谢"]):
            return "You're welcome! I'm happy to help. Is there anything else you'd like to know?"
        
        # 询问时间
        elif any(time_q in user_input_lower for time_q in ["time", "what time", "几点", "时间"]):
            from datetime import datetime
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return f"The current time is {current_time}."
        
        # 询问天气
        elif any(weather in user_input_lower for weather in ["weather", "天气"]):
            return ("I don't have access to real-time weather data right now, but I can help you find weather "
                   "information if you provide me with the right tools or APIs.")
        
        # 数学问题
        elif any(math_word in user_input_lower for math_word in ["calculate", "math", "计算", "数学"]):
            return ("I can help with mathematical calculations! Please provide me with the specific calculation "
                   "you'd like me to perform.")
        
        # 编程相关
        elif any(code_word in user_input_lower for code_word in ["code", "programming", "编程", "代码"]):
            return ("I can assist with programming tasks! I can help explain code, debug issues, suggest "
                   "improvements, or help you write new code. What programming task are you working on?")
        
        # 默认智能响应
        else:
            # 分析消息历史长度
            conversation_length = len([msg for msg in messages if msg.role in ["user", "assistant"]])
            
            if conversation_length <= 2:
                return (f"I understand you're asking about '{user_input}'. That's an interesting topic! "
                       f"Could you provide more details about what specifically you'd like to know?")
            else:
                return (f"Based on our conversation, I can see you're interested in '{user_input}'. "
                       f"Let me provide some helpful information about that topic. "
                       f"What specific aspect would you like me to focus on?")


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
            
            mock_client = getattr(self, '_mock_client', MockLLMClient(self.config))
            async for chunk in mock_client.stream_response(messages, **kwargs):
                yield chunk


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
        if config.provider == LLMProvider.MOCK:
            return MockLLMClient(config)
        elif config.provider == LLMProvider.OPENAI:
            return OpenAILLMClient(config)
        elif config.provider == LLMProvider.ANTHROPIC:
            # 可以在这里添加Anthropic客户端
            raise NotImplementedError("Anthropic客户端暂未实现")
        elif config.provider == LLMProvider.AZURE_OPENAI:
            # 可以在这里添加Azure OpenAI客户端
            raise NotImplementedError("Azure OpenAI客户端暂未实现")
        elif config.provider == LLMProvider.OLLAMA:
            # 可以在这里添加Ollama客户端
            raise NotImplementedError("Ollama客户端暂未实现")
        else:
            raise ValueError(f"不支持的LLM提供商: {config.provider}")


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
        self.logger = logging.getLogger("llm.manager")
        self._default_config = default_config
    
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
        生成响应
        
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
        
        client = self.clients[client_name]
        return await client.generate_response(messages, **kwargs)
    
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