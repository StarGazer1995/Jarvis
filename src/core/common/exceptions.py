"""
核心异常模块

定义项目中使用的自定义异常类
"""

from typing import Optional, Any, Dict


class JarvisError(Exception):
    """Jarvis项目的基础异常类"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        初始化异常
        
        Args:
            message: 错误消息
            details: 错误详情字典
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        """返回异常的字符串表示"""
        if self.details:
            return f"{self.message} (详情: {self.details})"
        return self.message


class ConfigurationError(JarvisError):
    """配置相关的错误"""
    
    def __init__(self, message: str, config_path: Optional[str] = None, 
                 config_section: Optional[str] = None, **kwargs):
        """
        初始化配置错误
        
        Args:
            message: 错误消息
            config_path: 配置文件路径
            config_section: 配置节名称
            **kwargs: 其他详情
        """
        details = kwargs
        if config_path:
            details["config_path"] = config_path
        if config_section:
            details["config_section"] = config_section
        
        super().__init__(message, details)
        self.config_path = config_path
        self.config_section = config_section


class LLMError(JarvisError):
    """LLM相关的错误"""
    
    def __init__(self, message: str, provider: Optional[str] = None, 
                 model: Optional[str] = None, **kwargs):
        """
        初始化LLM错误
        
        Args:
            message: 错误消息
            provider: 提供商名称
            model: 模型名称
            **kwargs: 其他详情
        """
        details = kwargs
        if provider:
            details["provider"] = provider
        if model:
            details["model"] = model
        
        super().__init__(message, details)
        self.provider = provider
        self.model = model


class ProviderError(LLMError):
    """提供商相关的错误"""
    
    def __init__(self, message: str, provider: str, error_code: Optional[str] = None, **kwargs):
        """
        初始化提供商错误
        
        Args:
            message: 错误消息
            provider: 提供商名称
            error_code: 错误代码
            **kwargs: 其他详情
        """
        details = kwargs
        if error_code:
            details["error_code"] = error_code
        
        super().__init__(message, provider=provider, **details)
        self.error_code = error_code


class AuthenticationError(ProviderError):
    """认证相关的错误"""
    pass


class RateLimitError(ProviderError):
    """速率限制错误"""
    
    def __init__(self, message: str, provider: str, retry_after: Optional[int] = None, **kwargs):
        """
        初始化速率限制错误
        
        Args:
            message: 错误消息
            provider: 提供商名称
            retry_after: 重试等待时间（秒）
            **kwargs: 其他详情
        """
        details = kwargs
        if retry_after:
            details["retry_after"] = retry_after
        
        super().__init__(message, provider, **details)
        self.retry_after = retry_after


class ModelNotFoundError(LLMError):
    """模型未找到错误"""
    pass


class ValidationError(JarvisError):
    """验证错误"""
    
    def __init__(self, message: str, field: Optional[str] = None, 
                 value: Optional[Any] = None, **kwargs):
        """
        初始化验证错误
        
        Args:
            message: 错误消息
            field: 字段名称
            value: 字段值
            **kwargs: 其他详情
        """
        details = kwargs
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = value
        
        super().__init__(message, details)
        self.field = field
        self.value = value


class TimeoutError(JarvisError):
    """超时错误"""
    
    def __init__(self, message: str, timeout_duration: Optional[float] = None, **kwargs):
        """
        初始化超时错误
        
        Args:
            message: 错误消息
            timeout_duration: 超时时长（秒）
            **kwargs: 其他详情
        """
        details = kwargs
        if timeout_duration:
            details["timeout_duration"] = timeout_duration
        
        super().__init__(message, details)
        self.timeout_duration = timeout_duration


class RetryExhaustedError(JarvisError):
    """重试次数耗尽错误"""
    
    def __init__(self, message: str, max_attempts: Optional[int] = None, 
                 last_error: Optional[Exception] = None, **kwargs):
        """
        初始化重试耗尽错误
        
        Args:
            message: 错误消息
            max_attempts: 最大重试次数
            last_error: 最后一次的错误
            **kwargs: 其他详情
        """
        details = kwargs
        if max_attempts:
            details["max_attempts"] = max_attempts
        if last_error:
            details["last_error"] = str(last_error)
        
        super().__init__(message, details)
        self.max_attempts = max_attempts
        self.last_error = last_error


class SecurityError(JarvisError):
    """安全相关的错误"""
    pass


class CapabilityError(JarvisError):
    """能力相关的错误"""
    
    def __init__(self, message: str, capability: Optional[str] = None, **kwargs):
        """
        初始化能力错误
        
        Args:
            message: 错误消息
            capability: 能力名称
            **kwargs: 其他详情
        """
        details = kwargs
        if capability:
            details["capability"] = capability
        
        super().__init__(message, details)
        self.capability = capability


class ToolError(JarvisError):
    """工具相关的错误"""
    
    def __init__(self, message: str, tool_name: Optional[str] = None, **kwargs):
        """
        初始化工具错误
        
        Args:
            message: 错误消息
            tool_name: 工具名称
            **kwargs: 其他详情
        """
        details = kwargs
        if tool_name:
            details["tool_name"] = tool_name
        
        super().__init__(message, details)
        self.tool_name = tool_name


class MCPError(JarvisError):
    """MCP相关的错误"""
    
    def __init__(self, message: str, server_name: Optional[str] = None, **kwargs):
        """
        初始化MCP错误
        
        Args:
            message: 错误消息
            server_name: 服务器名称
            **kwargs: 其他详情
        """
        details = kwargs
        if server_name:
            details["server_name"] = server_name
        
        super().__init__(message, details)
        self.server_name = server_name