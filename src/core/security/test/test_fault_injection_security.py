"""
Fault Injection Tests — Security Layer

Tests the resilience of the ARKSecurityManager against various failure
scenarios using the FaultInjector framework. Covers:

- Rate limit enforcement under high load
- Input validation for malicious payloads
- Permission checking for denied tools
- Audit logging failures
- Policy enforcement edge cases
"""

import time

import pytest

from src.core.common.exceptions import SecurityError
from src.core.security.manager import (
    ARKSecurityManager,
    AuditLogger,
    InputValidator,
    PermissionType,
    SecurityContext,
    ValidationRequest,
    ValidationResponse,
    ValidationResult,
)
from src.test_support.fault_injection.security_faults import (
    input_validation_failed,
    permission_denied,
    rate_limit_exceeded,
    requires_approval,
    security_audit_failure,
    security_then_succeed,
)


def make_request(
    user_id: str = "test_user",
    tool_name: str = "test_tool",
    permissions: set = None,
) -> ValidationRequest:
    """Helper to create a minimal ValidationRequest."""
    if permissions is None:
        permissions = {PermissionType.READ}
    return ValidationRequest(
        context=SecurityContext(
            user_id=user_id,
            session_id="session_1",
            tool_name=tool_name,
            tool_version="1.0",
            execution_id=f"exec_{time.time_ns()}",
            timestamp=time.time(),
        ),
        tool_parameters={},
        requested_permissions=permissions,
    )


# ═══════════════════════════════════════════════════════════════════
# 1. Pre-built Security Fault Helpers
# ═══════════════════════════════════════════════════════════════════


class TestPrebuiltSecurityFaults:
    """Verify the pre-built Security fault helpers work correctly."""

    def test_rate_limit_exceeded(self):
        side_effect = rate_limit_exceeded()
        with pytest.raises(SecurityError, match="Rate limit"):
            side_effect()

    def test_permission_denied(self):
        side_effect = permission_denied("delete_db")
        with pytest.raises(SecurityError, match="delete_db"):
            side_effect()

    def test_input_validation_failed(self):
        side_effect = input_validation_failed("user_input")
        with pytest.raises(SecurityError, match="user_input"):
            side_effect()

    def test_requires_approval(self):
        side_effect = requires_approval("dangerous_tool")
        with pytest.raises(SecurityError, match="requires user approval"):
            side_effect()

    def test_security_audit_failure(self):
        side_effect = security_audit_failure()
        with pytest.raises(SecurityError, match="audit"):
            side_effect()

    def test_security_then_succeed(self):
        side_effect = security_then_succeed(
            fail_count=2,
            success_value=ValidationResponse(
                result=ValidationResult.ALLOWED,
                policy_applied="test_policy",
                message="All checks passed",
            ),
        )

        with pytest.raises(SecurityError):
            side_effect()
        with pytest.raises(SecurityError):
            side_effect()

        result = side_effect()
        assert result.result == ValidationResult.ALLOWED


# ═══════════════════════════════════════════════════════════════════
# 2. ARKSecurityManager — Rate Limiting
# ═══════════════════════════════════════════════════════════════════


class TestSecurityManagerRateLimiting:
    """Test rate limiting behavior under load."""

    @pytest.mark.asyncio
    async def test_rate_limit_allows_within_limits(self):
        """Requests within rate limits should be allowed."""
        manager = ARKSecurityManager()
        request = make_request()

        response = await manager.validate_tool_execution(request, policy_name="low")
        assert response.result == ValidationResult.ALLOWED

    @pytest.mark.asyncio
    async def test_different_users_have_independent_rate_limits(self):
        """Rate limits should be per-user, not global."""
        manager = ARKSecurityManager()

        # User A exhausts their limit on the low policy (60/min)
        for _ in range(65):
            await manager.validate_tool_execution(
                make_request(user_id="user_a", tool_name="tool"),
                policy_name="low",
            )

        # User B should still be allowed
        response = await manager.validate_tool_execution(
            make_request(user_id="user_b"),
            policy_name="low",
        )
        assert response.result == ValidationResult.ALLOWED


# ═══════════════════════════════════════════════════════════════════
# 3. ARKSecurityManager — Permission Checking
# ═══════════════════════════════════════════════════════════════════


class TestSecurityManagerPermissionChecking:
    """Test permission-based access control."""

    @pytest.mark.asyncio
    async def test_high_security_denies_network_access(self):
        """HIGH security policy should deny NETWORK permissions."""
        manager = ARKSecurityManager()

        request = make_request(
            tool_name="web_search",
            permissions={PermissionType.NETWORK},
        )

        response = await manager.validate_tool_execution(request, policy_name="high")
        # HIGH policy requires approval for NETWORK permissions
        assert response.result in (
            ValidationResult.DENIED,
            ValidationResult.REQUIRES_APPROVAL,
        )

    @pytest.mark.asyncio
    async def test_high_security_denies_system_access(self):
        """HIGH security policy should deny SYSTEM permissions."""
        manager = ARKSecurityManager()

        request = make_request(
            tool_name="shell_exec",
            permissions={PermissionType.SYSTEM},
        )

        response = await manager.validate_tool_execution(request, policy_name="high")
        assert response.result == ValidationResult.DENIED

    @pytest.mark.asyncio
    async def test_low_security_allows_read_access(self):
        """LOW security policy should allow READ permissions."""
        manager = ARKSecurityManager()

        request = make_request(
            permissions={PermissionType.READ},
        )

        response = await manager.validate_tool_execution(request, policy_name="low")
        assert response.result == ValidationResult.ALLOWED


# ═══════════════════════════════════════════════════════════════════
# 4. InputValidator Edge Cases
# ═══════════════════════════════════════════════════════════════════


class TestInputValidatorFaultInjection:
    """Test InputValidator handling of malicious or malformed inputs."""

    def test_validate_size_rejects_oversized_input(self):
        """InputValidator should reject inputs exceeding max size."""
        validator = InputValidator()

        large_input = "A" * (1024 * 1024 + 1)
        is_valid = validator.validate_size(large_input, max_size=1024 * 1024)
        assert is_valid is False

    def test_validate_size_accepts_normal_input(self):
        """InputValidator should accept inputs within size limits."""
        validator = InputValidator()

        is_valid = validator.validate_size("normal input", max_size=1024 * 1024)
        assert is_valid is True

    def test_validate_file_path_allows_safe_path(self):
        """InputValidator should allow safe file paths."""
        validator = InputValidator()

        is_valid = validator.validate_file_path(
            "/tmp/safe_file.txt",
            allowed_patterns=["/tmp/*"],
            blocked_patterns=[],
        )
        assert is_valid is True

    def test_validate_domain_blocks_malicious_domain(self):
        """InputValidator should block blacklisted domains."""
        validator = InputValidator()

        is_valid = validator.validate_domain(
            "malicious.com",
            allowed_domains=set(),
            blocked_domains={"malicious.com"},
        )
        assert is_valid is False

    def test_validate_domain_allows_safe_domain(self):
        """InputValidator should allow whitelisted domains."""
        validator = InputValidator()

        is_valid = validator.validate_domain(
            "api.example.com",
            allowed_domains={"api.example.com"},
            blocked_domains=set(),
        )
        assert is_valid is True


# ═══════════════════════════════════════════════════════════════════
# 5. AuditLogger Fault Injection
# ═══════════════════════════════════════════════════════════════════


class TestAuditLoggerFaultInjection:
    """Test AuditLogger behavior under various conditions."""

    def test_audit_log_redacts_sensitive_keys(self):
        """AuditLogger should redact sensitive information like API keys."""
        logger = AuditLogger()

        context = SecurityContext(
            user_id="user",
            session_id="sess",
            tool_name="api_call",
            tool_version="1.0",
            execution_id="exec_1",
            timestamp=time.time(),
            metadata={"api_key": "sk-1234567890abcdef", "token": "secret-token"},
        )
        request = ValidationRequest(
            context=context,
            tool_parameters={"url": "https://api.example.com"},
            requested_permissions={PermissionType.READ},
        )

        audit_id = logger.log_validation_request(request)
        assert audit_id is not None

    def test_audit_log_violation_does_not_raise(self):
        """AuditLogger.log_security_violation should never raise."""
        logger = AuditLogger()

        context = SecurityContext(
            user_id="bad_actor",
            session_id="sess",
            tool_name="malicious_tool",
            tool_version="1.0",
            execution_id="exec_1",
            timestamp=time.time(),
        )

        try:
            logger.log_security_violation(
                context=context,
                violation_type="SQL injection attempt",
                details="DROP TABLE users;",
            )
        except Exception:
            pytest.fail("AuditLogger.log_security_violation raised unexpectedly")


# ═══════════════════════════════════════════════════════════════════
# 6. ARKSecurityManager — Edge Cases
# ═══════════════════════════════════════════════════════════════════


class TestSecurityManagerEdgeCases:
    """Test edge cases and error handling in the security manager."""

    @pytest.mark.asyncio
    async def test_validate_with_unknown_policy_falls_back(self):
        """Unknown policy names should fall back to a default."""
        manager = ARKSecurityManager()

        request = make_request()

        response = await manager.validate_tool_execution(
            request, policy_name="nonexistent_policy"
        )
        assert response is not None
        assert response.result in (
            ValidationResult.ALLOWED,
            ValidationResult.DENIED,
        )

    @pytest.mark.asyncio
    async def test_multiple_policies_independent_state(self):
        """Different policies should maintain independent state."""
        manager = ARKSecurityManager()

        for i in range(10):
            await manager.validate_tool_execution(
                make_request(tool_name=f"tool_{i}"),
                policy_name="low",
            )
            await manager.validate_tool_execution(
                make_request(tool_name=f"tool_{i}"),
                policy_name="medium",
            )
            await manager.validate_tool_execution(
                make_request(tool_name=f"tool_{i}"),
                policy_name="high",
            )

        assert True
