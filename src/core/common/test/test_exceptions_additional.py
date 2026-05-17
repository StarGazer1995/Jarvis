"""Additional exception tests migrated from legacy `src/test` files."""


class TestExceptionDetailedCoverage:
    """异常类完整边界测试"""

    def test_jarvis_error_str_with_details(self):
        from src.core.common.exceptions import JarvisError

        e = JarvisError("msg", details={"k": "v"})
        assert "k" in str(e)

    def test_jarvis_error_str_without_details(self):
        from src.core.common.exceptions import JarvisError

        e = JarvisError("msg")
        assert str(e) == "msg"

    def test_provider_error_with_provider(self):
        from src.core.common.exceptions import ProviderError

        e = ProviderError("openai", "failed")
        assert "openai" in str(e)

    def test_rate_limit_error(self):
        from src.core.common.exceptions import RateLimitError

        e = RateLimitError("openai", "too fast")
        assert "too fast" in str(e)

    def test_timeout_error_custom(self):
        from src.core.common.exceptions import TimeoutError

        e = TimeoutError("timeout", timeout=30.0)
        assert "timeout" in str(e)

    def test_retry_exhausted_error(self):
        from src.core.common.exceptions import RetryExhaustedError

        e = RetryExhaustedError("failed after 3 retries", last_error="timeout")
        assert "retries" in str(e)


class TestExceptionsMoreCoverage:
    """更多异常类测试"""

    def test_model_not_found_error(self):
        from src.core.common.exceptions import ModelNotFoundError

        e = ModelNotFoundError("model gpt-5 not found")
        assert "model" in str(e).lower()

    def test_security_error_str(self):
        from src.core.common.exceptions import SecurityError

        e = SecurityError("access denied")
        assert "denied" in str(e)

    def test_mcp_error(self):
        from src.core.common.exceptions import MCPError

        e = MCPError("MCP connection failed")
        assert "MCP" in str(e)

    def test_capability_error(self):
        from src.core.common.exceptions import CapabilityError

        e = CapabilityError("capability not implemented")
        assert "capability" in str(e).lower()

    def test_validation_error_custom(self):
        from src.core.common.exceptions import ValidationError

        e = ValidationError("invalid input")
        assert "invalid" in str(e)


class TestExceptionsRemaining:
    """Exception 剩余测试"""

    def test_mcp_error(self):
        from src.core.common.exceptions import MCPError

        e = MCPError("MCP failed")
        assert "MCP" in str(e)

    def test_capability_error(self):
        from src.core.common.exceptions import CapabilityError

        e = CapabilityError("cap failed")
        assert "cap" in str(e).lower()

    def test_model_not_found(self):
        from src.core.common.exceptions import ModelNotFoundError

        e = ModelNotFoundError("model not found")
        assert "model" in str(e).lower()

    def test_retry_exhausted(self):
        from src.core.common.exceptions import RetryExhaustedError

        e = RetryExhaustedError("exhausted", last_error="timeout")
        assert "exhausted" in str(e)

    def test_validation_error(self):
        from src.core.common.exceptions import ValidationError

        e = ValidationError("invalid")
        assert "invalid" in str(e)

    def test_security_error(self):
        from src.core.common.exceptions import SecurityError

        e = SecurityError("security breach")
        assert "security" in str(e).lower()
