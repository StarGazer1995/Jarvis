"""
OpenAI LLM客户端实现

提供真实的OpenAI API集成，支持完整的功能和错误处理。
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, AsyncGenerator

from ..client import BaseLLMClient, LLMMessage, LLMResponse
from ..types import LLMConfig
from ..utils.error_handler import (
    LLMError, LLMAPIError, LLMAuthenticationError, 
    LLMTimeoutError, LLMConfigurationError,
    handle_openai_error, log_llm_error
)
from ..utils.retry_handler import llm_retry, RateLimitHandler


class OpenAILLMClient(BaseLLMClient):
    """
    OpenAI LLM客户端
    
    提供与OpenAI API的完整集成，包括：
    - 真实的API调用
    - 错误处理和重试机制
    - 流式响应支持
    - 性能监控
    """
    
    def __init__(self, config: LLMConfig):
        """
        初始化OpenAI客户端
        
        Args:
            config: LLM配置
        """
        super().__init__(config)
        self._client = None
        self._rate_limit_handler = RateLimitHandler(logger=self.logger)
        self._request_count = 0
        self._total_tokens = 0
        self._start_time = time.time()
    
    async def initialize(self) -> bool:
        """
        初始化OpenAI客户端
        
        Returns:
            初始化是否成功
            
        Raises:
            LLMConfigurationError: 配置错误
            LLMAuthenticationError: 认证错误
        """
        try:
            # 验证配置
            self._validate_config()
            
            # 导入OpenAI库和httpx
            try:
                import openai
                import httpx
            except ImportError:
                raise LLMConfigurationError(
                    "OpenAI或httpx库未安装，请运行: uv add openai httpx"
                )
            
            # 创建自定义 httpx 客户端，配置连接池
            http_client = httpx.AsyncClient(
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
                timeout=httpx.Timeout(60.0)
            )

            # 创建OpenAI客户端
            self._client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=0,  # 我们使用自己的重试机制
                http_client=http_client,
            )
            
            # 验证连接
            await self._validate_connection()
            
            self._initialized = True
            self.logger.info(f"OpenAI客户端初始化成功，模型: {self.config.model}")
            return True
            
        except Exception as e:
            error = handle_openai_error(e)
            log_llm_error(error, self.logger)
            self._initialized = False
            return False
    
    def _validate_config(self) -> None:
        """
        验证配置参数
        
        Raises:
            LLMConfigurationError: 配置错误
        """
        provider_display = self.config.provider_name or "OpenAI"
        if not self.config.api_key:
            raise LLMConfigurationError(
                f"{provider_display} API密钥未设置，请在配置中提供api_key或设置相应的环境变量",
                config_field="api_key"
            )
        
        if not self.config.model:
            raise LLMConfigurationError(
                "模型名称未设置",
                config_field="model"
            )
        
        if self.config.temperature < 0 or self.config.temperature > 2:
            raise LLMConfigurationError(
                "temperature必须在0-2之间",
                config_field="temperature"
            )
        
        if self.config.max_tokens <= 0:
            raise LLMConfigurationError(
                "max_tokens必须大于0",
                config_field="max_tokens"
            )
    
    async def _validate_connection(self) -> None:
        """
        验证与OpenAI API的连接
        
        Raises:
            LLMAuthenticationError: 认证失败
            LLMAPIError: API错误
        """
        try:
            # 发送一个简单的测试请求
            test_messages = [
                {"role": "user", "content": "Hello"}
            ]
            
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=test_messages,
                max_tokens=1,
                temperature=0
            )
            
            self.logger.debug("OpenAI连接验证成功")
            
        except Exception as e:
            error = handle_openai_error(e)
            raise error
    
    @llm_retry
    async def generate_response(
        self, 
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """
        生成OpenAI响应
        
        Args:
            messages: 消息列表
            **kwargs: 额外参数
            
        Returns:
            LLM响应
            
        Raises:
            RuntimeError: 客户端未初始化
            LLMError: 各种LLM错误
        """
        if not self._initialized:
            raise RuntimeError("OpenAI客户端未初始化")
        
        try:
            # 等待速率限制
            await self._rate_limit_handler.wait_if_needed()
            
            # 转换消息格式
            openai_messages = self._convert_messages_to_openai(messages)
            
            # 准备请求参数
            request_params = self._prepare_request_params(openai_messages, **kwargs)
            
            # 记录请求开始
            start_time = time.time()
            self.logger.debug(f"发送OpenAI请求: {request_params['model']}")
            
            # 发送请求
            response = await self._client.chat.completions.create(**request_params)
            
            # 记录请求完成
            duration = time.time() - start_time
            self._request_count += 1
            
            # 转换响应
            llm_response = self._convert_openai_response(response)
            
            # 更新统计信息
            self._total_tokens += llm_response.usage.get('total_tokens', 0)
            
            self.logger.debug(
                f"OpenAI请求完成，耗时: {duration:.2f}s, "
                f"tokens: {llm_response.usage.get('total_tokens', 0)}"
            )
            
            return llm_response
            
        except Exception as e:
            error = handle_openai_error(e)
            log_llm_error(error, self.logger)
            raise error
    
    @llm_retry
    async def stream_response(
        self, 
        messages: List[LLMMessage],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        流式生成OpenAI响应
        
        Args:
            messages: 消息列表
            **kwargs: 额外参数
            
        Yields:
            响应内容片段
            
        Raises:
            RuntimeError: 客户端未初始化
            LLMError: 各种LLM错误
        """
        if not self._initialized:
            raise RuntimeError("OpenAI客户端未初始化")
        
        try:
            # 等待速率限制
            await self._rate_limit_handler.wait_if_needed()
            
            # 转换消息格式
            openai_messages = self._convert_messages_to_openai(messages)
            
            # 准备请求参数（启用流式）
            request_params = self._prepare_request_params(
                openai_messages, 
                stream=True, 
                **kwargs
            )
            
            # 记录请求开始
            start_time = time.time()
            self._request_count += 1
            self.logger.debug(f"发送OpenAI流式请求: {request_params['model']}")
            
            # 发送流式请求
            stream = await self._client.chat.completions.create(**request_params)
            
            # 处理流式响应
            in_reasoning = False
            async for chunk in stream:
                if not chunk.choices:
                    continue
                    
                delta = chunk.choices[0].delta
                
                # 尝试获取 reasoning_content (DeepSeek R1 等模型)
                reasoning = getattr(delta, 'reasoning_content', None)
                if reasoning is None and hasattr(delta, 'model_extra') and delta.model_extra:
                    reasoning = delta.model_extra.get('reasoning_content')
                
                # 如果有 reasoning，我们暂时忽略它或者将其合并到 thought 中
                # 在 JSON Mode 下，模型通常会将思考过程放入 "thought" 字段中
                # 如果模型通过 reasoning_content 字段返回思考过程，这通常是非 JSON 的 raw text
                # 这会破坏我们的 JSON 解析器。
                # 策略：忽略 reasoning_content，因为我们在 System Prompt 中要求模型在 JSON 的 "thought" 字段中输出思考。
                # if reasoning:
                #    pass 
                
                if delta.content:
                    yield delta.content
            
            # 记录请求完成
            duration = time.time() - start_time
            self.logger.debug(f"OpenAI流式请求完成，耗时: {duration:.2f}s")
            
        except Exception as e:
            error = handle_openai_error(e)
            log_llm_error(error, self.logger)
            raise error
    
    def _convert_messages_to_openai(self, messages: List[LLMMessage]) -> List[Dict[str, str]]:
        """
        将LLMMessage转换为OpenAI格式
        
        Args:
            messages: LLM消息列表
            
        Returns:
            OpenAI格式的消息列表
        """
        openai_messages = []
        for msg in messages:
            openai_msg = {
                "role": msg.role,
                "content": msg.content
            }
            openai_messages.append(openai_msg)
        return openai_messages
    
    def _prepare_request_params(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        准备OpenAI请求参数
        
        Args:
            messages: OpenAI格式的消息列表
            **kwargs: 额外参数
            
        Returns:
            请求参数字典
        """
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": kwargs.get("stream", False)
        }
        
        # 添加额外参数
        for key, value in self.config.extra_params.items():
            if key not in params:
                params[key] = value
        
        # 添加运行时参数
        for key, value in kwargs.items():
            if key not in ["temperature", "max_tokens", "stream"]:
                params[key] = value
        
        return params
    
    def _convert_openai_response(self, response) -> LLMResponse:
        """
        将OpenAI响应转换为LLMResponse
        
        Args:
            response: OpenAI响应对象
            
        Returns:
            LLM响应
        """
        choice = response.choices[0]
        content = choice.message.content or ""
        
        usage = {}
        if hasattr(response, 'usage') and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        
        return LLMResponse(
            content=content,
            usage=usage,
            model=response.model,
            finish_reason=choice.finish_reason or "",
            metadata={
                "id": response.id,
                "created": response.created,
                "object": response.object
            }
        )
    
    async def close(self) -> None:
        """
        关闭客户端连接
        """
        if self._client:
            await self._client.close()
            self._client = None
        
        self._initialized = False
        
        # 记录统计信息
        duration = time.time() - self._start_time
        self.logger.info(
            f"OpenAI客户端已关闭，统计信息: "
            f"请求数: {self._request_count}, "
            f"总tokens: {self._total_tokens}, "
            f"运行时间: {duration:.2f}s"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取客户端统计信息
        
        Returns:
            统计信息字典
        """
        duration = time.time() - self._start_time
        
        return {
            "provider": "openai",
            "model": self.config.model,
            "initialized": self._initialized,
            "request_count": self._request_count,
            "total_tokens": self._total_tokens,
            "uptime_seconds": duration,
            "requests_per_minute": (self._request_count / duration * 60) if duration > 0 else 0,
            "tokens_per_request": (self._total_tokens / self._request_count) if self._request_count > 0 else 0
        }