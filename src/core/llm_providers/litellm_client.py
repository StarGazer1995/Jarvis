"""
LiteLLM客户端实现

提供基于LiteLLM的多提供商支持。
"""

import logging
import time
from typing import List, Dict, Any, AsyncGenerator

from ..llm_client import BaseLLMClient, LLMConfig, LLMMessage, LLMResponse
from ..llm_utils.error_handler import (
    LLMError, LLMAPIError, LLMAuthenticationError, 
    LLMTimeoutError, LLMConfigurationError,
    log_llm_error
)
from ..llm_utils.retry_handler import llm_retry

class LiteLLMClient(BaseLLMClient):
    """
    LiteLLM统一客户端
    
    使用LiteLLM库支持多种LLM提供商（OpenAI, Anthropic, Gemini等）。
    """
    
    def __init__(self, config: LLMConfig):
        """
        初始化LiteLLM客户端
        
        Args:
            config: LLM配置
        """
        super().__init__(config)
        self._request_count = 0
        self._total_tokens = 0
        self._start_time = time.time()

    async def initialize(self) -> bool:
        """
        初始化客户端
        """
        try:
            import litellm
            # 可以在这里配置litellm的全局设置
            litellm.suppress_instrumentation = True  # 禁用自动仪表化以减少噪音
            
            self._initialized = True
            self.logger.info(f"LiteLLM客户端初始化成功，模型: {self.config.model}")
            return True
        except ImportError:
            raise LLMConfigurationError("LiteLLM库未安装，请运行: pip install litellm")
        except Exception as e:
            self.logger.error(f"LiteLLM初始化失败: {e}")
            return False

    @llm_retry
    async def generate_response(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """生成响应"""
        if not self._initialized:
            raise RuntimeError("LiteLLM客户端未初始化")

        try:
            import litellm
            
            # 转换消息格式
            formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            
            # 准备参数
            params = {
                "model": self.config.model,
                "messages": formatted_messages,
                "api_key": self.config.api_key,
                "base_url": self.config.base_url,
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "stream": False
            }
            
            # 处理超时
            if self.config.timeout:
                # LiteLLM通常接受timeout参数
                if isinstance(self.config.timeout, (int, float)):
                    params["timeout"] = float(self.config.timeout)
                elif isinstance(self.config.timeout, dict):
                     params["timeout"] = self.config.timeout.get("total", 30.0)

            # 添加额外参数
            params.update(self.config.extra_params)
            for k, v in kwargs.items():
                if k not in ["temperature", "max_tokens", "stream"]:
                    params[k] = v

            # 移除None值的参数
            params = {k: v for k, v in params.items() if v is not None}

            start_time = time.time()
            self.logger.debug(f"发送LiteLLM请求: {params['model']}")

            response = await litellm.acompletion(**params)
            
            duration = time.time() - start_time
            self._request_count += 1
            
            # 转换响应
            content = response.choices[0].message.content or ""
            usage = {}
            if hasattr(response, 'usage'):
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
                self._total_tokens += usage.get("total_tokens", 0)

            return LLMResponse(
                content=content,
                usage=usage,
                model=response.model,
                finish_reason=response.choices[0].finish_reason or "",
                metadata={
                    "id": response.id,
                    "created": response.created,
                    "object": response.object,
                    "provider": "litellm"
                }
            )

        except Exception as e:
            # 这里可以进一步细化LiteLLM的异常处理
            log_llm_error(LLMAPIError(f"LiteLLM调用失败: {str(e)}"), self.logger)
            raise LLMAPIError(f"LiteLLM调用失败: {str(e)}")

    @llm_retry
    async def stream_response(self, messages: List[LLMMessage], **kwargs) -> AsyncGenerator[str, None]:
        """流式生成响应"""
        if not self._initialized:
            raise RuntimeError("LiteLLM客户端未初始化")
            
        try:
            import litellm
            
            formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            
            params = {
                "model": self.config.model,
                "messages": formatted_messages,
                "api_key": self.config.api_key,
                "base_url": self.config.base_url,
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "stream": True
            }
            
            # 处理超时
            if self.config.timeout:
                if isinstance(self.config.timeout, (int, float)):
                    params["timeout"] = float(self.config.timeout)
                elif isinstance(self.config.timeout, dict):
                     params["timeout"] = self.config.timeout.get("total", 30.0)

             # 添加额外参数
            params.update(self.config.extra_params)
            for k, v in kwargs.items():
                if k not in ["temperature", "max_tokens", "stream"]:
                    params[k] = v
            
            params = {k: v for k, v in params.items() if v is not None}

            self.logger.debug(f"发送LiteLLM流式请求: {params['model']}")
            
            response = await litellm.acompletion(**params)
            
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            log_llm_error(LLMAPIError(f"LiteLLM流式调用失败: {str(e)}"), self.logger)
            raise LLMAPIError(f"LiteLLM流式调用失败: {str(e)}")
