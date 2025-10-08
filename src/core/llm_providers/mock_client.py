"""
Mock LLM客户端实现

提供模拟的LLM API集成，用于测试和开发环境。
"""

import asyncio
import logging
import time
import random
from typing import List, Dict, Any, Optional, AsyncGenerator

from ..llm_client import BaseLLMClient, LLMConfig, LLMMessage, LLMResponse
from ..llm_utils.error_handler import LLMConfigurationError


class MockLLMClient(BaseLLMClient):
    """
    Mock LLM客户端
    
    提供模拟的LLM API响应，用于：
    - 开发和测试环境
    - 功能验证
    - 性能测试
    - 演示目的
    """
    
    def __init__(self, config: LLMConfig):
        """
        初始化Mock客户端
        
        Args:
            config: LLM配置
        """
        super().__init__(config)
        self._request_count = 0
        self._total_tokens = 0
        self._start_time = time.time()
        
        # Mock响应模板
        self._response_templates = [
            "Hello! I'm a mock AI assistant. How can I help you today?",
            "That's an interesting question. Let me provide you with a helpful response.",
            "I understand what you're asking. Here's my mock response to your query.",
            "Thank you for your message. I'm here to assist you with mock responses.",
            "Based on your input, here's a simulated AI response for testing purposes."
        ]
    
    async def initialize(self) -> bool:
        """
        初始化Mock客户端
        
        Returns:
            初始化是否成功
        """
        try:
            # 验证配置
            self._validate_config()
            
            # 模拟初始化延迟
            await asyncio.sleep(0.1)
            
            self._initialized = True
            self.logger.info(f"Mock LLM客户端初始化成功，模型: {self.config.model}")
            return True
            
        except Exception as e:
            self.logger.error(f"Mock LLM客户端初始化失败: {e}")
            self._initialized = False
            return False
    
    def _validate_config(self) -> None:
        """
        验证配置参数
        
        Raises:
            LLMConfigurationError: 配置错误
        """
        if not self.config.model:
            raise LLMConfigurationError("模型名称不能为空")
        
        if self.config.max_tokens is not None and self.config.max_tokens <= 0:
            raise LLMConfigurationError("max_tokens必须大于0")
        
        if self.config.temperature is not None and (
            self.config.temperature < 0 or self.config.temperature > 2
        ):
            raise LLMConfigurationError("temperature必须在0-2之间")
    
    async def generate_response(
        self, 
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """
        生成Mock响应
        
        Args:
            messages: 消息列表
            **kwargs: 额外参数
            
        Returns:
            LLM响应
            
        Raises:
            RuntimeError: 客户端未初始化
        """
        if not self._initialized:
            raise RuntimeError("Mock客户端未初始化")
        
        # 模拟处理延迟
        processing_time = random.uniform(0.5, 2.0)
        await asyncio.sleep(processing_time)
        
        # 记录请求
        self._request_count += 1
        
        # 生成Mock响应
        last_message = messages[-1] if messages else None
        response_content = self._generate_mock_content(last_message)
        
        # 计算Token使用量
        input_tokens = sum(len(msg.content.split()) for msg in messages)
        output_tokens = len(response_content.split())
        total_tokens = input_tokens + output_tokens
        self._total_tokens += total_tokens
        
        # 创建响应对象
        response = LLMResponse(
            content=response_content,
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": total_tokens
            },
            model=self.config.model,
            finish_reason="stop"
        )
        
        self.logger.debug(f"Mock响应生成完成，耗时: {processing_time:.2f}s")
        return response
    
    async def stream_response(
        self, 
        messages: List[LLMMessage],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        流式生成Mock响应
        
        Args:
            messages: 消息列表
            **kwargs: 额外参数
            
        Yields:
            响应内容片段
            
        Raises:
            RuntimeError: 客户端未初始化
        """
        if not self._initialized:
            raise RuntimeError("Mock客户端未初始化")
        
        # 记录请求
        self._request_count += 1
        
        # 生成Mock响应内容
        last_message = messages[-1] if messages else None
        full_content = self._generate_mock_content(last_message)
        
        # 将内容分块流式返回
        words = full_content.split()
        for i, word in enumerate(words):
            # 模拟流式延迟
            await asyncio.sleep(random.uniform(0.05, 0.2))
            
            # 添加空格（除了最后一个词）
            chunk = word + (" " if i < len(words) - 1 else "")
            yield chunk
        
        self.logger.debug("Mock流式响应完成")
    
    def _generate_mock_content(self, last_message: Optional[LLMMessage]) -> str:
        """
        生成Mock响应内容
        
        Args:
            last_message: 最后一条消息
            
        Returns:
            Mock响应内容
        """
        if last_message and last_message.content:
            # 基于用户输入生成相关响应
            user_content = last_message.content.lower()
            
            if "hello" in user_content or "hi" in user_content:
                return "Hello! Nice to meet you. I'm a mock AI assistant ready to help."
            elif "help" in user_content:
                return "I'm here to help! This is a mock response for testing purposes."
            elif "python" in user_content:
                return "Python is a great programming language! Here's some mock information about Python development."
            elif "thank" in user_content:
                return "You're welcome! I'm glad I could help with this mock response."
            elif "?" in user_content:
                return "That's a good question! Here's a mock answer to demonstrate the response generation."
            else:
                # 随机选择一个模板响应
                return random.choice(self._response_templates)
        else:
            return random.choice(self._response_templates)
    
    async def close(self) -> None:
        """
        关闭Mock客户端
        """
        if self._initialized:
            self.logger.info("Mock LLM客户端已关闭")
            self._initialized = False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取客户端统计信息
        
        Returns:
            统计信息字典
        """
        uptime = time.time() - self._start_time
        
        return {
            "provider": "mock",
            "model": self.config.model,
            "initialized": self._initialized,
            "request_count": self._request_count,
            "total_tokens": self._total_tokens,
            "uptime_seconds": uptime,
            "avg_tokens_per_request": (
                self._total_tokens / self._request_count 
                if self._request_count > 0 else 0
            ),
            "requests_per_minute": (
                self._request_count / (uptime / 60) 
                if uptime > 0 else 0
            )
        }