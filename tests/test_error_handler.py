"""
错误处理器单元测试

测试LLM错误处理的各种异常类型和处理机制。
"""

import pytest
import logging
from unittest.mock import patch, MagicMock

from src.core.llm.utils.error_handler import (
    LLMError,
    LLMAPIError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMTimeoutError,
    LLMConfigurationError,
    handle_openai_error,
    log_llm_error,
)


class TestLLMErrorClasses:
    """LLM错误类测试"""

    def test_llm_error_base(self):
        """测试基础LLM错误"""
        error = LLMError("测试错误")

        assert str(error) == "测试错误"
        assert error.message == "测试错误"
        assert error.error_code is None
        assert error.details == {}

    def test_llm_error_with_details(self):
        """测试带详细信息的LLM错误"""
        details = {"request_id": "123", "model": "gpt-4"}
        error = LLMError("测试错误", error_code="TEST_ERROR", details=details)

        assert error.error_code == "TEST_ERROR"
        assert error.details == details

    def test_llm_api_error(self):
        """测试API错误"""
        error = LLMAPIError("API调用失败", status_code=500)

        assert error.status_code == 500
        assert error.error_code == "API_ERROR"
        assert isinstance(error, LLMError)

    def test_llm_rate_limit_error(self):
        """测试速率限制错误"""
        error = LLMRateLimitError("速率限制", retry_after=60)

        assert error.retry_after == 60
        assert error.error_code == "RATE_LIMIT_ERROR"
        assert isinstance(error, LLMError)

    def test_llm_authentication_error(self):
        """测试认证错误"""
        error = LLMAuthenticationError("认证失败")

        assert error.error_code == "AUTHENTICATION_ERROR"
        assert isinstance(error, LLMError)

    def test_llm_timeout_error(self):
        """测试超时错误"""
        error = LLMTimeoutError("请求超时", timeout_duration=30.0)

        assert error.timeout_duration == 30.0
        assert error.error_code == "TIMEOUT_ERROR"
        assert isinstance(error, LLMError)

    def test_llm_configuration_error(self):
        """测试配置错误"""
        error = LLMConfigurationError("配置错误", config_field="api_key")

        assert error.config_field == "api_key"
        assert error.error_code == "CONFIGURATION_ERROR"
        assert isinstance(error, LLMError)


class TestOpenAIErrorHandling:
    """OpenAI错误处理测试"""

    @patch("src.core.llm.utils.error_handler.openai", create=True)
    def test_handle_openai_authentication_error(self, mock_openai_module):
        """测试处理OpenAI认证错误"""

        # 创建模拟的认证错误类
        class MockAuthenticationError(Exception):
            pass

        mock_openai_module.AuthenticationError = MockAuthenticationError
        mock_error = MockAuthenticationError("Invalid API key")

        result = handle_openai_error(mock_error)

        assert isinstance(result, LLMAuthenticationError)
        assert "API密钥" in str(result)

    @patch("src.core.llm.utils.error_handler.openai", create=True)
    def test_handle_openai_rate_limit_error(self, mock_openai_module):
        """测试处理OpenAI速率限制错误"""

        # 定义所有相关的异常类
        class MockAuthenticationError(Exception):
            pass

        class MockRateLimitError(Exception):
            def __init__(self, message, retry_after=None):
                super().__init__(message)
                self.retry_after = retry_after

        class MockAPITimeoutError(Exception):
            pass

        class MockAPIError(Exception):
            pass

        # 赋值给mock模块
        mock_openai_module.AuthenticationError = MockAuthenticationError
        mock_openai_module.RateLimitError = MockRateLimitError
        mock_openai_module.APITimeoutError = MockAPITimeoutError
        mock_openai_module.APIError = MockAPIError

        mock_error = MockRateLimitError("Rate limit exceeded", retry_after=60)

        result = handle_openai_error(mock_error)

        assert isinstance(result, LLMRateLimitError)
        assert "速率限制" in str(result)
        assert result.retry_after == 60

    @patch("src.core.llm.utils.error_handler.openai", create=True)
    def test_handle_openai_api_error(self, mock_openai_module):
        """测试处理OpenAI API错误"""

        # 定义所有相关的异常类
        class MockAuthenticationError(Exception):
            pass

        class MockRateLimitError(Exception):
            pass

        class MockAPITimeoutError(Exception):
            pass

        class MockAPIError(Exception):
            def __init__(self, message, status_code=None):
                super().__init__(message)
                self.status_code = status_code

        # 赋值给mock模块
        mock_openai_module.AuthenticationError = MockAuthenticationError
        mock_openai_module.RateLimitError = MockRateLimitError
        mock_openai_module.APITimeoutError = MockAPITimeoutError
        mock_openai_module.APIError = MockAPIError

        mock_error = MockAPIError("API error", status_code=400)

        result = handle_openai_error(mock_error)

        assert isinstance(result, LLMAPIError)
        assert "API错误" in str(result)
        assert result.status_code == 400

    @patch("src.core.llm.utils.error_handler.openai", create=True)
    def test_handle_openai_timeout_error(self, mock_openai_module):
        """测试处理OpenAI超时错误"""

        # 定义所有相关的异常类
        class MockAuthenticationError(Exception):
            pass

        class MockRateLimitError(Exception):
            pass

        class MockAPITimeoutError(Exception):
            pass

        class MockAPIError(Exception):
            pass

        # 赋值给mock模块
        mock_openai_module.AuthenticationError = MockAuthenticationError
        mock_openai_module.RateLimitError = MockRateLimitError
        mock_openai_module.APITimeoutError = MockAPITimeoutError
        mock_openai_module.APIError = MockAPIError

        mock_error = MockAPITimeoutError("Request timeout")

        result = handle_openai_error(mock_error)

        assert isinstance(result, LLMTimeoutError)
        assert "超时" in str(result)

    def test_handle_openai_connection_error(self):
        """测试处理OpenAI连接错误"""
        # 模拟连接错误（不是OpenAI特定错误）
        connection_error = ConnectionError("Connection failed")

        result = handle_openai_error(connection_error)

        assert isinstance(result, LLMError)
        assert "Connection failed" in str(result)

    def test_handle_unknown_openai_error(self):
        """测试处理未知OpenAI错误"""
        unknown_error = ValueError("Unknown error")

        result = handle_openai_error(unknown_error)

        assert isinstance(result, LLMError)
        assert "Unknown error" in str(result)

    def test_handle_openai_import_error(self):
        """测试OpenAI库未安装时的处理"""
        error = Exception("Test error")

        # 模拟openai模块导入失败
        with patch("builtins.__import__") as mock_import:
            mock_import.side_effect = ImportError("No module named 'openai'")

            result = handle_openai_error(error)

            assert isinstance(result, LLMError)
            assert "Test error" in str(result)


class TestErrorLogging:
    """错误日志记录测试"""

    @patch("src.core.llm.utils.error_handler.logging.getLogger")
    def test_log_llm_error_basic(self, mock_get_logger):
        """测试基本错误日志记录"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        error = LLMAPIError("API错误")
        log_llm_error(error)

        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0][0]
        assert "API错误" in call_args

    @patch("src.core.llm.utils.error_handler.logging.getLogger")
    def test_log_llm_error_with_custom_logger(self, mock_get_logger):
        """测试使用自定义logger记录错误"""
        custom_logger = MagicMock()

        error = LLMAPIError("API错误")
        log_llm_error(error, custom_logger)

        custom_logger.error.assert_called_once()
        # 不应该调用getLogger
        mock_get_logger.assert_not_called()

    @patch("src.core.llm.utils.error_handler.logging.getLogger")
    def test_log_llm_error_with_details(self, mock_get_logger):
        """测试记录带详细信息的错误"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        error = LLMAPIError("API错误", status_code=500)
        log_llm_error(error)

        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0][0]
        assert "API错误" in call_args


class TestErrorHandlerEdgeCases:
    """错误处理器边缘情况测试"""

    def test_handle_none_error(self):
        """测试处理None错误"""
        result = handle_openai_error(None)

        assert isinstance(result, LLMError)
        assert "None" in str(result)

    def test_handle_error_without_message(self):
        """测试处理没有消息的错误"""
        error = Exception()

        result = handle_openai_error(error)

        assert isinstance(result, LLMError)
        # 应该有默认消息
        assert str(result) != ""

    def test_log_error_with_none_logger(self):
        """测试使用None logger记录错误"""
        error = LLMAPIError("API错误")

        # 应该使用默认logger而不抛出异常
        log_llm_error(error, None)

        # 验证函数正常执行（没有抛出异常）
        assert True

    def test_error_repr(self):
        """测试错误的字符串表示"""
        error = LLMAPIError("API错误", status_code=500)

        repr_str = repr(error)
        assert "LLMAPIError" in repr_str
        assert "API错误" in repr_str
