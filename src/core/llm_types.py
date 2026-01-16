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
