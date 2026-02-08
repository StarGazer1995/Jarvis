"""
LLM错误处理模块

定义LLM相关的异常类和错误处理逻辑。
"""

import logging
from typing import Optional, Dict, Any

try:
    import openai
except ImportError:
    openai = None

class LLMError(Exception):
    """
    LLM相关错误基类
    
    所有LLM相关的异常都应该继承自这个基类。
    """
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        初始化LLM错误
        
        Args:
            message: 错误消息
            error_code: 错误代码
            details: 错误详情
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class LLMAPIError(LLMError):
    """
    LLM API调用错误
    
    当API调用失败时抛出此异常。
    """
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict[str, Any]] = None):
        """
        初始化API错误
        
        Args:
            message: 错误消息
            status_code: HTTP状态码
            response_data: 响应数据
        """
        super().__init__(message, error_code="API_ERROR")
        self.status_code = status_code
        self.response_data = response_data or {}


class LLMRateLimitError(LLMError):
    """
    LLM速率限制错误
    
    当API调用超过速率限制时抛出此异常。
    """
    
    def __init__(self, message: str, retry_after: Optional[int] = None):
        """
        初始化速率限制错误
        
        Args:
            message: 错误消息
            retry_after: 建议重试等待时间（秒）
        """
        super().__init__(message, error_code="RATE_LIMIT_ERROR")
        self.retry_after = retry_after


class LLMAuthenticationError(LLMError):
    """
    LLM认证错误
    
    当API密钥无效或认证失败时抛出此异常。
    """
    
    def __init__(self, message: str = "认证失败，请检查API密钥"):
        """
        初始化认证错误
        
        Args:
            message: 错误消息
        """
        super().__init__(message, error_code="AUTHENTICATION_ERROR")


class LLMTimeoutError(LLMError):
    """
    LLM超时错误
    
    当API调用超时时抛出此异常。
    """
    
    def __init__(self, message: str = "请求超时", timeout_duration: Optional[float] = None):
        """
        初始化超时错误
        
        Args:
            message: 错误消息
            timeout_duration: 超时时长（秒）
        """
        super().__init__(message, error_code="TIMEOUT_ERROR")
        self.timeout_duration = timeout_duration


class LLMConfigurationError(LLMError):
    """
    LLM配置错误
    
    当配置无效或缺失时抛出此异常。
    """
    
    def __init__(self, message: str, config_field: Optional[str] = None):
        """
        初始化配置错误
        
        Args:
            message: 错误消息
            config_field: 配置字段名
        """
        super().__init__(message, error_code="CONFIGURATION_ERROR")
        self.config_field = config_field


def handle_openai_error(error: Exception) -> LLMError:
    """
    处理OpenAI特定的错误，转换为标准LLM错误
    
    Args:
        error: OpenAI异常
        
    Returns:
        标准化的LLM错误
    """
    logger = logging.getLogger(__name__)
    
    if openai is None:
        logger.warning("OpenAI库未安装，无法处理特定错误类型")
        return LLMError(f"LLM错误: {str(error)}")
        
    if isinstance(error, openai.AuthenticationError):
        return LLMAuthenticationError("OpenAI API密钥无效或认证失败")
    elif isinstance(error, openai.RateLimitError):
        return LLMRateLimitError("OpenAI API速率限制", retry_after=getattr(error, 'retry_after', None))
    elif isinstance(error, openai.APITimeoutError):
        return LLMTimeoutError("OpenAI API请求超时")
    elif isinstance(error, openai.APIError):
        status_code = getattr(error, 'status_code', None)
        return LLMAPIError(f"OpenAI API错误: {str(error)}", status_code=status_code)
    else:
        return LLMError(f"OpenAI未知错误: {str(error)}")


def log_llm_error(error: LLMError, logger: Optional[logging.Logger] = None) -> None:
    """
    记录LLM错误日志
    
    Args:
        error: LLM错误
        logger: 日志记录器，如果为None则使用默认记录器
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    error_info = {
        "error_type": type(error).__name__,
        "error_code": error.error_code,
        "message": error.message,
        "details": error.details
    }
    
    # 添加特定错误类型的额外信息
    if isinstance(error, LLMAPIError):
        error_info["status_code"] = error.status_code
        error_info["response_data"] = error.response_data
    elif isinstance(error, LLMRateLimitError):
        error_info["retry_after"] = error.retry_after
    elif isinstance(error, LLMTimeoutError):
        error_info["timeout_duration"] = error.timeout_duration
    elif isinstance(error, LLMConfigurationError):
        error_info["config_field"] = error.config_field
    
    logger.error(f"LLM错误: {error_info}")