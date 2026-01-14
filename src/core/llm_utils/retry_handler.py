"""
重试处理模块

提供LLM API调用的重试机制。
"""

import asyncio
import logging
import time
import inspect
from typing import Callable, Any, Optional, Type, Union, List
from functools import wraps

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)

from .error_handler import LLMError, LLMAPIError, LLMRateLimitError, LLMTimeoutError


class RetryHandler:
    """
    重试处理器
    
    提供灵活的重试配置和策略。
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        min_wait: float = 1.0,
        max_wait: float = 60.0,
        multiplier: float = 2.0,
        retry_on_exceptions: Optional[List[Type[Exception]]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化重试处理器
        
        Args:
            max_attempts: 最大重试次数
            min_wait: 最小等待时间（秒）
            max_wait: 最大等待时间（秒）
            multiplier: 等待时间倍数
            retry_on_exceptions: 需要重试的异常类型列表
            logger: 日志记录器
        """
        self.max_attempts = max_attempts
        self.min_wait = min_wait
        self.max_wait = max_wait
        self.multiplier = multiplier
        self.logger = logger or logging.getLogger(__name__)
        
        # 默认重试的异常类型
        if retry_on_exceptions is None:
            retry_on_exceptions = [
                LLMAPIError,
                LLMRateLimitError,
                LLMTimeoutError,
                ConnectionError,
                TimeoutError
            ]
        self.retry_on_exceptions = retry_on_exceptions
    
    def create_retry_decorator(self):
        """
        创建重试装饰器
        
        Returns:
            重试装饰器
        """
        return retry(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(
                multiplier=self.multiplier,
                min=self.min_wait,
                max=self.max_wait
            ),
            retry=retry_if_exception_type(tuple(self.retry_on_exceptions)),
            before_sleep=before_sleep_log(self.logger, logging.WARNING),
            after=after_log(self.logger, logging.INFO)
        )
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        执行函数并在失败时重试
        
        Args:
            func: 要执行的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            函数执行结果
            
        Raises:
            最后一次执行的异常
        """
        retry_decorator = self.create_retry_decorator()
        
        if asyncio.iscoroutinefunction(func):
            @retry_decorator
            async def async_wrapper():
                return await func(*args, **kwargs)
            return await async_wrapper()
        else:
            @retry_decorator
            def sync_wrapper():
                return func(*args, **kwargs)
            return sync_wrapper()


def with_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    multiplier: float = 2.0,
    retry_on_exceptions: Optional[List[Type[Exception]]] = None,
    logger: Optional[logging.Logger] = None
):
    """
    重试装饰器工厂函数
    
    Args:
        max_attempts: 最大重试次数
        min_wait: 最小等待时间（秒）
        max_wait: 最大等待时间（秒）
        multiplier: 等待时间倍数
        retry_on_exceptions: 需要重试的异常类型列表
        logger: 日志记录器
        
    Returns:
        重试装饰器
    """
    handler = RetryHandler(
        max_attempts=max_attempts,
        min_wait=min_wait,
        max_wait=max_wait,
        multiplier=multiplier,
        retry_on_exceptions=retry_on_exceptions,
        logger=logger
    )
    
    def decorator(func):
        if inspect.isasyncgenfunction(func):
            @wraps(func)
            async def async_gen_wrapper(*args, **kwargs):
                # 对于异步生成器，暂时不提供自动重试机制
                # 或者可以尝试仅对生成器的创建进行重试（如果创建时就抛出异常）
                # 但这里简单起见，直接透传
                async for item in func(*args, **kwargs):
                    yield item
            return async_gen_wrapper
        elif asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await handler.execute_with_retry(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return handler.execute_with_retry(func, *args, **kwargs)
            return sync_wrapper
    
    return decorator


class RateLimitHandler:
    """
    速率限制处理器
    
    处理API速率限制，实现智能等待和重试。
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化速率限制处理器
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        self._last_request_time = 0.0
        self._min_interval = 0.1  # 最小请求间隔（秒）
    
    async def wait_if_needed(self, retry_after: Optional[Union[int, float]] = None) -> None:
        """
        根据速率限制等待
        
        Args:
            retry_after: 建议等待时间（秒）
        """
        current_time = time.time()
        
        if retry_after:
            # 如果有明确的等待时间，使用它
            wait_time = float(retry_after)
            self.logger.info(f"速率限制，等待 {wait_time} 秒")
            await asyncio.sleep(wait_time)
        else:
            # 否则使用最小间隔
            elapsed = current_time - self._last_request_time
            if elapsed < self._min_interval:
                wait_time = self._min_interval - elapsed
                await asyncio.sleep(wait_time)
        
        self._last_request_time = time.time()
    
    def handle_rate_limit_error(self, error: LLMRateLimitError) -> float:
        """
        处理速率限制错误，返回建议等待时间
        
        Args:
            error: 速率限制错误
            
        Returns:
            建议等待时间（秒）
        """
        if error.retry_after:
            return float(error.retry_after)
        
        # 如果没有明确的等待时间，使用指数退避
        return min(60.0, 2.0 ** (3 - 1))  # 最多等待60秒


# 预定义的重试装饰器
llm_retry = with_retry(
    max_attempts=3,
    min_wait=1.0,
    max_wait=60.0,
    multiplier=2.0,
    retry_on_exceptions=[
        LLMAPIError,
        LLMRateLimitError,
        LLMTimeoutError,
        ConnectionError,
        TimeoutError
    ]
)

# 用于速率限制敏感操作的重试装饰器
rate_limit_retry = with_retry(
    max_attempts=5,
    min_wait=2.0,
    max_wait=120.0,
    multiplier=2.0,
    retry_on_exceptions=[LLMRateLimitError]
)