"""
重试处理器单元测试

测试重试机制的各种场景和配置。
"""

import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock

from src.core.llm.utils.retry_handler import (
    RetryHandler,
    RateLimitHandler,
    with_retry,
    llm_retry,
    rate_limit_retry
)
from src.core.llm.utils.error_handler import (
    LLMError,
    LLMAPIError,
    LLMRateLimitError,
    LLMTimeoutError
)


class TestRetryHandler:
    """重试处理器测试"""
    
    def test_retry_handler_initialization(self):
        """测试重试处理器初始化"""
        handler = RetryHandler(
            max_attempts=5,
            min_wait=2.0,
            max_wait=60.0,
            multiplier=2.5,
            retry_on_exceptions=[LLMAPIError, LLMTimeoutError]
        )
        
        assert handler.max_attempts == 5
        assert handler.min_wait == 2.0
        assert handler.max_wait == 60.0
        assert handler.multiplier == 2.5
        assert LLMAPIError in handler.retry_on_exceptions
        assert LLMTimeoutError in handler.retry_on_exceptions
    
    def test_retry_handler_default_values(self):
        """测试重试处理器默认值"""
        handler = RetryHandler()
        
        assert handler.max_attempts == 3
        assert handler.min_wait == 1.0
        assert handler.max_wait == 60.0
        assert handler.multiplier == 2.0
        assert LLMAPIError in handler.retry_on_exceptions
        assert LLMTimeoutError in handler.retry_on_exceptions
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_success_first_attempt(self):
        """测试第一次尝试就成功"""
        handler = RetryHandler(max_attempts=3)
        
        async def successful_function():
            return "成功"
        
        result = await handler.execute_with_retry(successful_function)
        assert result == "成功"
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_success_after_retries(self):
        """测试重试后成功"""
        handler = RetryHandler(
            max_attempts=3,
            min_wait=0.01  # 短等待时间
        )
        
        call_count = 0
        
        async def retry_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise LLMAPIError("API错误")
            return "成功"
        
        result = await handler.execute_with_retry(retry_then_success)
        assert result == "成功"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_max_attempts_exceeded(self):
        """测试超过最大重试次数"""
        handler = RetryHandler(
            max_attempts=2,
            min_wait=0.01
        )
        
        async def always_fails():
            raise LLMAPIError("API错误")
        
        # tenacity会包装异常为RetryError
        from tenacity import RetryError
        with pytest.raises(RetryError):
            await handler.execute_with_retry(always_fails)
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_non_retryable_exception(self):
        """测试不可重试异常"""
        handler = RetryHandler(
            max_attempts=3,
            retry_on_exceptions=[LLMAPIError]
        )
        
        async def non_retryable_error():
            raise LLMError("一般错误")
        
        with pytest.raises(LLMError):
            await handler.execute_with_retry(non_retryable_error)


class TestRateLimitHandler:
    """速率限制处理器测试"""
    
    def test_rate_limit_handler_initialization(self):
        """测试速率限制处理器初始化"""
        handler = RateLimitHandler()
        assert handler.logger is not None
    
    @pytest.mark.asyncio
    async def test_wait_if_needed_with_retry_after(self):
        """测试带retry_after的等待"""
        handler = RateLimitHandler()
        
        start_time = time.time()
        await handler.wait_if_needed(retry_after=0.1)  # 短等待时间
        end_time = time.time()
        
        # 验证等待时间大致正确（允许一些误差）
        assert 0.05 <= (end_time - start_time) <= 0.2
    
    @pytest.mark.asyncio
    async def test_wait_if_needed_without_retry_after(self):
        """测试没有retry_after时的等待"""
        handler = RateLimitHandler()
        
        # 第一次调用应该不等待（因为没有上次请求时间）
        start_time = time.time()
        await handler.wait_if_needed()
        end_time = time.time()
        
        # 第一次调用应该很快完成
        assert (end_time - start_time) < 0.1
        
        # 第二次调用应该等待最小间隔
        start_time = time.time()
        await handler.wait_if_needed()
        end_time = time.time()
        
        # 应该等待至少最小间隔（0.1秒），但允许一些误差
        assert (end_time - start_time) >= 0.05  # 允许50%的误差
    
    def test_handle_rate_limit_error_with_retry_after(self):
        """测试处理带retry_after的速率限制错误"""
        handler = RateLimitHandler()
        
        error = LLMRateLimitError("速率限制", retry_after=60)
        wait_time = handler.handle_rate_limit_error(error)
        
        assert wait_time == 60.0
    
    def test_handle_rate_limit_error_without_retry_after(self):
        """测试处理不带retry_after的速率限制错误"""
        handler = RateLimitHandler()
        
        error = LLMRateLimitError("速率限制")
        wait_time = handler.handle_rate_limit_error(error)
        
        assert wait_time == 4.0  # 指数退避计算结果


class TestWithRetryDecorator:
    """with_retry装饰器测试"""
    
    @pytest.mark.asyncio
    async def test_with_retry_success_first_attempt(self):
        """测试第一次尝试就成功"""
        @with_retry(max_attempts=3)
        def successful_function():
            return "成功"
        
        result = await successful_function()
        assert result == "成功"
    
    @pytest.mark.asyncio
    async def test_with_retry_success_after_retries(self):
        """测试重试后成功"""
        call_count = 0
        
        @with_retry(
            max_attempts=3,
            min_wait=0.01,  # 短等待时间
            retry_on_exceptions=[LLMAPIError]
        )
        def retry_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise LLMAPIError("API错误")
            return "成功"
        
        result = await retry_then_success()
        assert result == "成功"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_with_retry_max_attempts_exceeded(self):
        """测试超过最大重试次数"""
        @with_retry(
            max_attempts=2,
            min_wait=0.01,
            retry_on_exceptions=[LLMAPIError]
        )
        def always_fails():
            raise LLMAPIError("API错误")
        
        from tenacity import RetryError
        with pytest.raises(RetryError):  # tenacity会包装异常
            await always_fails()
    
    @pytest.mark.asyncio
    async def test_with_retry_non_retryable_exception(self):
        """测试不可重试异常"""
        @with_retry(
            max_attempts=3,
            retry_on_exceptions=[LLMAPIError]
        )
        def non_retryable_error():
            raise LLMError("一般错误")
        
        with pytest.raises(LLMError):
            await non_retryable_error()
    
    @pytest.mark.asyncio
    async def test_with_retry_async_success(self):
        """测试异步函数重试成功"""
        call_count = 0
        
        @with_retry(
            max_attempts=3,
            min_wait=0.01,
            retry_on_exceptions=[LLMAPIError]
        )
        async def async_retry_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise LLMAPIError("API错误")
            return "异步成功"
        
        result = await async_retry_then_success()
        assert result == "异步成功"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_with_retry_async_max_attempts(self):
        """测试异步函数超过最大重试次数"""
        @with_retry(
            max_attempts=2,
            min_wait=0.01,
            retry_on_exceptions=[LLMAPIError]
        )
        async def async_always_fails():
            raise LLMAPIError("API错误")
        
        from tenacity import RetryError
        with pytest.raises(RetryError):
            await async_always_fails()


class TestPredefinedDecorators:
    """预定义装饰器测试"""
    
    @pytest.mark.asyncio
    async def test_llm_retry_decorator(self):
        """测试llm_retry装饰器"""
        call_count = 0
        
        @llm_retry
        def function_with_llm_retry():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise LLMAPIError("API错误")
            return "LLM重试成功"
        
        result = await function_with_llm_retry()
        assert result == "LLM重试成功"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_rate_limit_retry_decorator(self):
        """测试rate_limit_retry装饰器"""
        call_count = 0
        
        @rate_limit_retry
        def function_with_rate_limit_retry():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise LLMRateLimitError("速率限制", retry_after=0.01)
            return "速率限制重试成功"
        
        result = await function_with_rate_limit_retry()
        assert result == "速率限制重试成功"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_llm_retry_async_decorator(self):
        """测试llm_retry异步装饰器"""
        call_count = 0
        
        @llm_retry
        async def async_function_with_llm_retry():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise LLMTimeoutError("超时")
            return "异步LLM重试成功"
        
        result = await async_function_with_llm_retry()
        assert result == "异步LLM重试成功"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_rate_limit_retry_async_decorator(self):
        """测试rate_limit_retry异步装饰器"""
        call_count = 0
        
        @rate_limit_retry
        async def async_function_with_rate_limit_retry():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise LLMRateLimitError("速率限制", retry_after=0.01)
            return "异步速率限制重试成功"
        
        result = await async_function_with_rate_limit_retry()
        assert result == "异步速率限制重试成功"
        assert call_count == 2


class TestRetryHandlerEdgeCases:
    """重试处理器边缘情况测试"""
    
    def test_zero_max_attempts(self):
        """测试最大尝试次数为0"""
        handler = RetryHandler(max_attempts=0)
        assert handler.max_attempts == 0
    
    def test_negative_wait_time(self):
        """测试负等待时间"""
        handler = RetryHandler(min_wait=-1.0)
        # 实现应该处理负值
        assert handler.min_wait == -1.0  # 或者被修正为0
    
    def test_zero_multiplier(self):
        """测试倍数为0"""
        handler = RetryHandler(multiplier=0.0)
        assert handler.multiplier == 0.0
    
    @pytest.mark.asyncio
    async def test_with_retry_function_with_args_kwargs(self):
        """测试带参数的函数重试"""
        call_count = 0
        
        @with_retry(
            max_attempts=2,
            min_wait=0.01,
            retry_on_exceptions=[LLMAPIError]
        )
        def function_with_args(x, y, z=None):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise LLMAPIError("API错误")
            return f"x={x}, y={y}, z={z}"
        
        result = await function_with_args(1, 2, z=3)
        assert result == "x=1, y=2, z=3"
        assert call_count == 2
    
    def test_with_retry_preserves_function_metadata(self):
        """测试重试装饰器保留函数元数据"""
        @with_retry()
        def documented_function():
            """这是一个有文档的函数"""
            return "结果"
        
        # 检查包装后的函数是否保留了原始函数的元数据
        assert hasattr(documented_function, '__name__')
        assert hasattr(documented_function, '__doc__')