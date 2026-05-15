"""
覆盖补全测试 - 异常类和相关边缘情况

使用已验证的 API 进行测试。
"""

import pytest


class TestExceptionsCoverage:
    """Jarvis 异常类的全面测试"""

    def test_jarvis_error_basic(self):
        from src.core.common.exceptions import JarvisError

        e = JarvisError("test error")
        assert str(e) == "test error"
        assert e.details == {}

    def test_jarvis_error_with_details(self):
        from src.core.common.exceptions import JarvisError

        e = JarvisError("test error", details={"key": "val"})
        assert "test error" in str(e)
        assert "key" in str(e)
        assert e.details["key"] == "val"

    def test_configuration_error_with_path(self):
        from src.core.common.exceptions import ConfigurationError

        e = ConfigurationError("bad config", config_path="/path/to/config.yaml")
        assert "bad config" in str(e)
        assert e.config_path == "/path/to/config.yaml"

    def test_configuration_error_with_section(self):
        from src.core.common.exceptions import ConfigurationError

        e = ConfigurationError("bad config", config_section="providers")
        assert e.config_section == "providers"

    def test_configuration_error_no_path(self):
        from src.core.common.exceptions import ConfigurationError

        e = ConfigurationError("bad config")
        assert e.config_path is None

    def test_llm_error_basic(self):
        from src.core.common.exceptions import LLMError

        e = LLMError("llm error")
        assert "llm error" in str(e)

    def test_provider_error(self):
        from src.core.common.exceptions import ProviderError

        e = ProviderError("openai", "provider failed")
        assert isinstance(e, ProviderError)

    def test_authentication_error(self):
        from src.core.common.exceptions import AuthenticationError

        e = AuthenticationError("openai", "auth failed")
        assert "auth failed" in str(e)

    def test_rate_limit_error(self):
        from src.core.common.exceptions import RateLimitError

        e = RateLimitError("openai", "rate limited")
        assert "rate limited" in str(e)

    def test_timeout_error(self):
        from src.core.common.exceptions import TimeoutError

        e = TimeoutError("timeout")
        assert "timeout" in str(e)

    def test_validation_error(self):
        from src.core.common.exceptions import ValidationError

        e = ValidationError("invalid input")
        assert "invalid input" in str(e)

    def test_security_error(self):
        from src.core.common.exceptions import SecurityError

        e = SecurityError("security violation")
        assert "security" in str(e).lower()

    def test_tool_error(self):
        from src.core.common.exceptions import ToolError

        e = ToolError("tool execution failed")
        assert "tool" in str(e).lower()
