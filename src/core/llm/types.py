"""
LLM类型定义模块

包含LLM相关的基础数据结构和枚举。
"""

from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

class LLMProvider(Enum):
    """支持的LLM提供商"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    OLLAMA = "ollama"
    LITELLM = "litellm"  # LiteLLM统一接口
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
    provider: Union[LLMProvider, str]
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: Union[float, Dict[str, float]] = 30.0
    retry: Dict[str, Any] = field(default_factory=dict)
    stream: bool = False
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    # 兼容性字段
    retry_attempts: int = 3

    def __post_init__(self):
        # 如果retry为空但提供了retry_attempts，构造retry字典
        if not self.retry and self.retry_attempts:
            self.retry = {"max_attempts": self.retry_attempts}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LLMConfig':
        """Create LLMConfig from dictionary."""
        provider_str = config_dict.get('provider', 'mock')
        try:
            provider = LLMProvider(provider_str)
        except ValueError:
            # Try lowercase
            try:
                provider = LLMProvider(provider_str.lower())
            except ValueError:
                provider = LLMProvider.MOCK # Default to mock if unknown
            
        return cls(
            provider=provider,
            api_key=config_dict.get('api_key'),
            model=config_dict.get('model', 'gpt-3.5-turbo'),
            base_url=config_dict.get('base_url'),
            max_tokens=config_dict.get('max_tokens', 1000),
            temperature=config_dict.get('temperature', 0.7),
            timeout=config_dict.get('timeout', 30),
            retry_attempts=config_dict.get('retry_attempts', 3),
            stream=config_dict.get('stream', False),
            extra_params=config_dict.get('extra_params', {})
        )
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Create LLMConfig from environment variables."""
        import os
        provider_str = os.getenv('LLM_PROVIDER', 'mock')
        try:
            provider = LLMProvider(provider_str)
        except ValueError:
            try:
                provider = LLMProvider(provider_str.lower())
            except ValueError:
                provider = LLMProvider.MOCK
            
        return cls(
            provider=provider,
            api_key=os.getenv('OPENAI_API_KEY'),
            model=os.getenv('LLM_MODEL', 'gpt-3.5-turbo'),
            base_url=os.getenv('LLM_BASE_URL'),
            max_tokens=int(os.getenv('LLM_MAX_TOKENS', '1000')),
            temperature=float(os.getenv('LLM_TEMPERATURE', '0.7')),
            timeout=int(os.getenv('LLM_TIMEOUT', '30')),
            retry_attempts=int(os.getenv('LLM_RETRY_ATTEMPTS', '3')),
            stream=os.getenv('LLM_STREAM', 'false').lower() == 'true'
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert LLMConfig to dictionary."""
        provider_val = self.provider.value if isinstance(self.provider, LLMProvider) else str(self.provider)
        return {
            'provider': provider_val,
            'api_key': self.api_key,
            'model': self.model,
            'base_url': self.base_url,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'timeout': self.timeout,
            'retry_attempts': self.retry_attempts,
            'stream': self.stream,
            'extra_params': self.extra_params
        }
