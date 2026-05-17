"""Additional security manager tests migrated from legacy `src/test` files."""

import pytest


class TestSecurityDetailedCoverage:
    """安全管理器边界测试"""

    def test_security_context_to_dict_permissions(self):
        from src.core.security.manager import SecurityContext

        ctx = SecurityContext(
            user_id="u1",
            session_id="s1",
            tool_name="tool1",
            tool_version="1.0",
            execution_id="e1",
            timestamp=1234567890.0,
            permissions_granted=set(),
        )
        d = ctx.to_dict()
        assert d["user_id"] == "u1"
        assert d["permissions_granted"] == []

    def test_security_policy_roundtrip(self):
        from src.core.security.manager import SecurityLevel, SecurityPolicy

        policy = SecurityPolicy(
            name="custom",
            description="Custom policy",
            security_level=SecurityLevel.MEDIUM,
            allowed_permissions=set(),
        )
        d = policy.to_dict()
        assert d["name"] == "custom"
        restored = SecurityPolicy.from_dict(d)
        assert restored.name == "custom"
        assert restored.security_level == SecurityLevel.MEDIUM

    def test_input_validator_size_bytes(self):
        from src.core.security.manager import InputValidator

        assert InputValidator.validate_size(b"small", max_size=100) is True
        assert InputValidator.validate_size(b"x" * 200, max_size=100) is False

    def test_input_validator_size_number(self):
        from src.core.security.manager import InputValidator

        assert InputValidator.validate_size(12345, max_size=100) is True

    def test_validate_file_path_blocked(self):
        from src.core.security.manager import InputValidator

        result = InputValidator.validate_file_path(
            "/etc/passwd",
            allowed_patterns=["/home/*"],
            blocked_patterns=["/etc/*"],
        )
        assert result is False

    def test_validate_file_path_allowed(self):
        from src.core.security.manager import InputValidator

        result = InputValidator.validate_file_path(
            "/home/user/file.txt",
            allowed_patterns=["/home/*"],
            blocked_patterns=["/etc/*"],
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_rate_limiter_check(self):
        from src.core.security.manager import RateLimiter

        limiter = RateLimiter()
        result = await limiter.check_rate_limit(
            "user1", "tool1", per_minute_limit=60, per_hour_limit=1000
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_rate_limiter_rate_exceeded(self):
        from src.core.security.manager import RateLimiter

        limiter = RateLimiter()
        # Exceed per-minute limit
        for _ in range(5):
            await limiter.check_rate_limit(
                "user1", "tool1", per_minute_limit=3, per_hour_limit=100
            )
        # Should be False now
        result = await limiter.check_rate_limit(
            "user1", "tool1", per_minute_limit=3, per_hour_limit=100
        )
        assert result is False

    def test_audit_logger_redact_sensitive_dict(self):
        from src.core.security.manager import AuditLogger

        logger = AuditLogger()
        # _redact only redacts values for sensitive keys in dicts
        result = logger._redact({"api_key": "sk-secret123"})
        assert "***" in str(result)
        # Non-sensitive dict should not be redacted
        result2 = logger._redact({"name": "test"})
        assert result2["name"] == "test"
        # Number should pass through
        assert logger._redact(42) == 42

    def test_audit_logger_init_with_file(self):
        from src.core.security.manager import AuditLogger

        logger = AuditLogger(log_file="/tmp/test_audit.log")
        assert logger.log_file == "/tmp/test_audit.log"

    def test_security_manager_invalid_policy(self):
        from src.core.security.manager import ARKSecurityManager

        mgr = ARKSecurityManager()
        # Policies should be loaded by default
        assert len(mgr.policies) > 0
        assert "low" in mgr.policies
        assert "medium" in mgr.policies


class TestSecurityManagerMoreCoverage:
    """Security Manager 更多边界测试"""

    def test_security_policy_from_dict_full(self):
        from src.core.security.manager import SecurityLevel, SecurityPolicy

        # Test full roundtrip with all fields
        policy = SecurityPolicy(
            name="full",
            description="Full policy",
            security_level=SecurityLevel.HIGH,
            allowed_permissions=set(),
            denied_permissions=set(),
            rate_limit_per_minute=30,
            allowed_domains={"example.com"},
            blocked_domains={"bad.com"},
            allowed_file_patterns=["/safe/*"],
            blocked_file_patterns=["/danger/*"],
            max_input_size=512,
        )
        d = policy.to_dict()
        assert d["rate_limit_per_minute"] == 30
        assert "example.com" in d["allowed_domains"]
        assert "/safe/*" in d["allowed_file_patterns"]

    def test_security_manager_get_policy_nonexistent(self):
        from src.core.security.manager import ARKSecurityManager

        mgr = ARKSecurityManager()
        policy = mgr.get_policy("nonexistent_policy")
        assert policy is None

    def test_security_manager_list_policies(self):
        from src.core.security.manager import ARKSecurityManager

        mgr = ARKSecurityManager()
        names = mgr.list_policies()
        assert isinstance(names, list)
        assert "low" in names
        assert "medium" in names
        from src.core.security.manager import ARKSecurityManager

        mgr = ARKSecurityManager()
        names = mgr.list_policies()
        assert isinstance(names, list)
        assert "low" in names
        assert "medium" in names


class TestSecurityManagerRemaining:
    """Security Manager 剩余测试"""

    def test_add_policy(self):
        from src.core.security.manager import (
            ARKSecurityManager,
            SecurityLevel,
            SecurityPolicy,
        )

        mgr = ARKSecurityManager()
        policy = SecurityPolicy(
            name="custom",
            description="Custom",
            security_level=SecurityLevel.LOW,
            allowed_permissions=set(),
        )
        result = mgr.add_policy(policy)
        assert result is True
        assert mgr.get_policy("custom") is not None

    def test_remove_policy(self):
        from src.core.security.manager import ARKSecurityManager

        mgr = ARKSecurityManager()
        result = mgr.remove_policy("low")
        assert result is True
        assert mgr.get_policy("low") is None

    def test_register_custom_validator(self):
        from src.core.security.manager import ARKSecurityManager

        mgr = ARKSecurityManager()
        result = mgr.register_custom_validator("test", lambda x: True)
        assert result is True


class TestSecurityLastBatch:
    """Security 最后边界"""

    def test_check_rate_limit_with_exceed(self):
        import asyncio

        from src.core.security.manager import RateLimiter

        limiter = RateLimiter()
        # Exceed per-minute limit
        for i in range(5):
            asyncio.run(limiter.check_rate_limit("u1", "t1", 3, 1000))
        result = asyncio.run(limiter.check_rate_limit("u1", "t1", 3, 1000))
        assert result is False

    def test_check_rate_limit_per_hour(self):
        import asyncio

        from src.core.security.manager import RateLimiter

        limiter = RateLimiter()
        # Exceed per-hour limit by filling minute windows
        # Just test basic functionality
        asyncio.run(limiter.check_rate_limit("u2", "t1", 60, 3))
        asyncio.run(limiter.check_rate_limit("u2", "t1", 60, 3))
        asyncio.run(limiter.check_rate_limit("u2", "t1", 60, 3))
        # 3 is the per-hour limit, so 3rd should still work (3 <= 3), 4th fails
        result4 = asyncio.run(limiter.check_rate_limit("u2", "t1", 60, 3))
        assert result4 is False
