"""
Tests for the ARK Security Manager.

This module contains comprehensive tests for the security validation
and management system, including policy enforcement, rate limiting,
input validation, and audit logging.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.security.manager import (
    ARKSecurityManager,
    AuditLogger,
    InputValidator,
    PermissionType,
    RateLimiter,
    SecurityContext,
    SecurityLevel,
    SecurityPolicy,
    ValidationRequest,
    ValidationResponse,
    ValidationResult,
)


class TestSecurityPolicy:
    """Test cases for SecurityPolicy class."""

    def test_security_policy_creation(self):
        """Test creating a security policy."""
        policy = SecurityPolicy(
            name="test_policy",
            description="Test policy",
            security_level=SecurityLevel.MEDIUM,
            allowed_permissions={PermissionType.READ, PermissionType.WRITE},
        )

        assert policy.name == "test_policy"
        assert policy.description == "Test policy"
        assert policy.security_level == SecurityLevel.MEDIUM
        assert PermissionType.READ in policy.allowed_permissions
        assert PermissionType.WRITE in policy.allowed_permissions
        assert policy.max_execution_time == 30.0
        assert policy.rate_limit_per_minute == 60
        assert not policy.requires_approval

    def test_security_policy_to_dict(self):
        """Test converting security policy to dictionary."""
        policy = SecurityPolicy(
            name="test_policy",
            description="Test policy",
            security_level=SecurityLevel.HIGH,
            allowed_permissions={PermissionType.READ},
            denied_permissions={PermissionType.SYSTEM},
            requires_approval=True,
        )

        policy_dict = policy.to_dict()

        assert policy_dict["name"] == "test_policy"
        assert policy_dict["security_level"] == "high"
        assert "read" in policy_dict["allowed_permissions"]
        assert "system" in policy_dict["denied_permissions"]
        assert policy_dict["requires_approval"] is True

    def test_security_policy_from_dict(self):
        """Test creating security policy from dictionary."""
        policy_data = {
            "name": "test_policy",
            "description": "Test policy",
            "security_level": "low",
            "allowed_permissions": ["read", "write"],
            "denied_permissions": ["system"],
            "max_execution_time": 45.0,
            "requires_approval": True,
        }

        policy = SecurityPolicy.from_dict(policy_data)

        assert policy.name == "test_policy"
        assert policy.security_level == SecurityLevel.LOW
        assert PermissionType.READ in policy.allowed_permissions
        assert PermissionType.WRITE in policy.allowed_permissions
        assert PermissionType.SYSTEM in policy.denied_permissions
        assert policy.max_execution_time == 45.0
        assert policy.requires_approval is True


class TestSecurityContext:
    """Test cases for SecurityContext class."""

    def test_security_context_creation(self):
        """Test creating a security context."""
        context = SecurityContext(
            user_id="user123",
            session_id="session456",
            tool_name="test_tool",
            tool_version="1.0.0",
            execution_id="exec789",
            timestamp=time.time(),
        )

        assert context.user_id == "user123"
        assert context.session_id == "session456"
        assert context.tool_name == "test_tool"
        assert context.tool_version == "1.0.0"
        assert context.execution_id == "exec789"
        assert context.timestamp > 0

    def test_security_context_to_dict(self):
        """Test converting security context to dictionary."""
        context = SecurityContext(
            user_id="user123",
            session_id="session456",
            tool_name="test_tool",
            tool_version="1.0.0",
            execution_id="exec789",
            timestamp=1234567890.0,
            permissions_granted={PermissionType.READ, PermissionType.WRITE},
        )

        context_dict = context.to_dict()

        assert context_dict["user_id"] == "user123"
        assert context_dict["tool_name"] == "test_tool"
        assert "read" in context_dict["permissions_granted"]
        assert "write" in context_dict["permissions_granted"]


class TestValidationResponse:
    """Test cases for ValidationResponse class."""

    def test_validation_response_creation(self):
        """Test creating a validation response."""
        response = ValidationResponse(
            result=ValidationResult.ALLOWED,
            policy_applied="medium",
            message="Tool execution allowed",
            allowed_permissions={PermissionType.READ},
        )

        assert response.result == ValidationResult.ALLOWED
        assert response.policy_applied == "medium"
        assert response.message == "Tool execution allowed"
        assert PermissionType.READ in response.allowed_permissions

    def test_validation_response_to_dict(self):
        """Test converting validation response to dictionary."""
        response = ValidationResponse(
            result=ValidationResult.DENIED,
            policy_applied="high",
            message="Permission denied",
            denied_permissions={PermissionType.SYSTEM},
        )

        response_dict = response.to_dict()

        assert response_dict["result"] == "denied"
        assert response_dict["policy_applied"] == "high"
        assert response_dict["message"] == "Permission denied"
        assert "system" in response_dict["denied_permissions"]


class TestRateLimiter:
    """Test cases for RateLimiter class."""

    @pytest.fixture
    def rate_limiter(self):
        """Create a rate limiter for testing."""
        return RateLimiter()

    @pytest.mark.asyncio
    async def test_rate_limit_within_limits(self, rate_limiter):
        """Test rate limiting when within limits."""
        result = await rate_limiter.check_rate_limit(
            user_id="user123",
            tool_name="test_tool",
            per_minute_limit=10,
            per_hour_limit=100,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_rate_limit_per_minute_exceeded(self, rate_limiter):
        """Test rate limiting when per-minute limit is exceeded."""
        user_id = "user123"
        tool_name = "test_tool"

        # Execute up to the limit
        for _ in range(5):
            result = await rate_limiter.check_rate_limit(
                user_id=user_id,
                tool_name=tool_name,
                per_minute_limit=5,
                per_hour_limit=100,
            )
            assert result is True

        # Next execution should be rate limited
        result = await rate_limiter.check_rate_limit(
            user_id=user_id, tool_name=tool_name, per_minute_limit=5, per_hour_limit=100
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_rate_limit_status(self, rate_limiter):
        """Test getting rate limit status."""
        user_id = "user123"
        tool_name = "test_tool"

        # Execute a few times
        for _ in range(3):
            await rate_limiter.check_rate_limit(
                user_id=user_id,
                tool_name=tool_name,
                per_minute_limit=10,
                per_hour_limit=100,
            )

        status = await rate_limiter.get_rate_limit_status(user_id, tool_name)

        assert status["executions_last_minute"] == 3
        assert status["executions_last_hour"] == 3
        assert "next_reset_minute" in status
        assert "next_reset_hour" in status

    @pytest.mark.asyncio
    async def test_rate_limit_different_users(self, rate_limiter):
        """Test rate limiting for different users."""
        # User 1 hits the limit
        for _ in range(5):
            result = await rate_limiter.check_rate_limit(
                user_id="user1",
                tool_name="test_tool",
                per_minute_limit=5,
                per_hour_limit=100,
            )
            assert result is True

        # User 1 is now rate limited
        result = await rate_limiter.check_rate_limit(
            user_id="user1",
            tool_name="test_tool",
            per_minute_limit=5,
            per_hour_limit=100,
        )
        assert result is False

        # User 2 should still be allowed
        result = await rate_limiter.check_rate_limit(
            user_id="user2",
            tool_name="test_tool",
            per_minute_limit=5,
            per_hour_limit=100,
        )
        assert result is True


class TestInputValidator:
    """Test cases for InputValidator class."""

    def test_validate_size_string(self):
        """Test validating string size."""
        small_string = "hello"
        large_string = "x" * 1000

        assert InputValidator.validate_size(small_string, 100) is True
        assert InputValidator.validate_size(large_string, 100) is False

    def test_validate_size_dict(self):
        """Test validating dictionary size."""
        small_dict = {"key": "value"}
        large_dict = {"key": "x" * 1000}

        assert InputValidator.validate_size(small_dict, 100) is True
        assert InputValidator.validate_size(large_dict, 100) is False

    def test_validate_file_path_allowed(self):
        """Test validating file paths with allowed patterns."""
        allowed_patterns = [r"/tmp/.*", r"/home/user/.*"]
        blocked_patterns = [r"/etc/.*"]

        assert (
            InputValidator.validate_file_path(
                "/tmp/test.txt", allowed_patterns, blocked_patterns
            )
            is True
        )
        assert (
            InputValidator.validate_file_path(
                "/home/user/doc.pdf", allowed_patterns, blocked_patterns
            )
            is True
        )
        assert (
            InputValidator.validate_file_path(
                "/var/log/test.log", allowed_patterns, blocked_patterns
            )
            is False
        )

    def test_validate_file_path_blocked(self):
        """Test validating file paths with blocked patterns."""
        allowed_patterns = []
        blocked_patterns = [r"/etc/.*", r"/root/.*"]

        assert (
            InputValidator.validate_file_path(
                "/tmp/test.txt", allowed_patterns, blocked_patterns
            )
            is True
        )
        assert (
            InputValidator.validate_file_path(
                "/etc/passwd", allowed_patterns, blocked_patterns
            )
            is False
        )
        assert (
            InputValidator.validate_file_path(
                "/root/secret.txt", allowed_patterns, blocked_patterns
            )
            is False
        )

    def test_validate_domain_allowed(self):
        """Test validating domains with allowed list."""
        allowed_domains = {"example.com", "*.google.com"}
        blocked_domains = set()

        assert (
            InputValidator.validate_domain(
                "example.com", allowed_domains, blocked_domains
            )
            is True
        )
        assert (
            InputValidator.validate_domain(
                "api.google.com", allowed_domains, blocked_domains
            )
            is True
        )
        assert (
            InputValidator.validate_domain(
                "malicious.com", allowed_domains, blocked_domains
            )
            is False
        )

    def test_validate_domain_blocked(self):
        """Test validating domains with blocked list."""
        allowed_domains = set()
        blocked_domains = {"malicious.com", "*.spam.com"}

        assert (
            InputValidator.validate_domain(
                "example.com", allowed_domains, blocked_domains
            )
            is True
        )
        assert (
            InputValidator.validate_domain(
                "malicious.com", allowed_domains, blocked_domains
            )
            is False
        )
        assert (
            InputValidator.validate_domain(
                "ads.spam.com", allowed_domains, blocked_domains
            )
            is False
        )

    def test_sanitize_input_string(self):
        """Test sanitizing string input."""
        dangerous_input = "<script>alert('xss')</script>"
        sanitized = InputValidator.sanitize_input(dangerous_input)

        assert "<" not in sanitized
        assert ">" not in sanitized
        assert "script" in sanitized  # Content should remain

    def test_sanitize_input_dict(self):
        """Test sanitizing dictionary input."""
        dangerous_dict = {
            "safe_key": "safe_value",
            "dangerous_key": "<script>alert('xss')</script>",
        }
        sanitized = InputValidator.sanitize_input(dangerous_dict)

        assert sanitized["safe_key"] == "safe_value"
        assert "<" not in sanitized["dangerous_key"]
        assert ">" not in sanitized["dangerous_key"]


class TestAuditLogger:
    """Test cases for AuditLogger class."""

    @pytest.fixture
    def audit_logger(self):
        """Create an audit logger for testing."""
        return AuditLogger()

    @pytest.fixture
    def sample_context(self):
        """Create a sample security context."""
        return SecurityContext(
            user_id="user123",
            session_id="session456",
            tool_name="test_tool",
            tool_version="1.0.0",
            execution_id="exec789",
            timestamp=time.time(),
        )

    @pytest.fixture
    def sample_request(self, sample_context):
        """Create a sample validation request."""
        return ValidationRequest(
            context=sample_context,
            tool_parameters={"param1": "value1"},
            requested_permissions={PermissionType.READ},
        )

    def test_log_validation_request(self, audit_logger, sample_request):
        """Test logging a validation request."""
        sample_request.metadata = {"api_key": "secret_value"}
        with patch.object(audit_logger.logger, "info") as mock_info:
            audit_id = audit_logger.log_validation_request(sample_request)

            assert audit_id is not None
            assert len(audit_id) == 16  # SHA256 hash truncated to 16 chars
            mock_info.assert_called_once()
            log_text = mock_info.call_args[0][0]
            assert "secret_value" not in log_text
            assert "***REDACTED***" in log_text

    def test_log_validation_response(self, audit_logger):
        """Test logging a validation response."""
        response = ValidationResponse(
            result=ValidationResult.ALLOWED,
            policy_applied="medium",
            message="Test message",
        )

        with patch.object(audit_logger.logger, "info") as mock_info:
            audit_logger.log_validation_response(response, "test_audit_id")
            mock_info.assert_called_once()

    def test_log_security_violation(self, audit_logger, sample_context):
        """Test logging a security violation."""
        with patch.object(audit_logger.logger, "warning") as mock_warning:
            audit_logger.log_security_violation(
                sample_context,
                "unauthorized_access",
                "User attempted to access restricted resource",
            )
            mock_warning.assert_called_once()


class TestARKSecurityManager:
    """Test cases for ARKSecurityManager class."""

    @pytest.fixture
    def security_manager(self):
        """Create a security manager for testing."""
        return ARKSecurityManager()

    @pytest.fixture
    def sample_context(self):
        """Create a sample security context."""
        return SecurityContext(
            user_id="user123",
            session_id="session456",
            tool_name="test_tool",
            tool_version="1.0.0",
            execution_id="exec789",
            timestamp=time.time(),
        )

    def test_security_manager_initialization(self, security_manager):
        """Test security manager initialization."""
        assert len(security_manager.policies) == 4  # Default policies
        assert "low" in security_manager.policies
        assert "medium" in security_manager.policies
        assert "high" in security_manager.policies
        assert "critical" in security_manager.policies
        assert security_manager.rate_limiter is not None
        assert security_manager.input_validator is not None
        assert security_manager.audit_logger is not None

    def test_add_policy(self, security_manager):
        """Test adding a security policy."""
        policy = SecurityPolicy(
            name="custom_policy",
            description="Custom test policy",
            security_level=SecurityLevel.MEDIUM,
            allowed_permissions={PermissionType.READ},
        )

        result = security_manager.add_policy(policy)
        assert result is True
        assert "custom_policy" in security_manager.policies

    def test_remove_policy(self, security_manager):
        """Test removing a security policy."""
        # Add a policy first
        policy = SecurityPolicy(
            name="temp_policy",
            description="Temporary policy",
            security_level=SecurityLevel.LOW,
            allowed_permissions={PermissionType.READ},
        )
        security_manager.add_policy(policy)

        # Remove the policy
        result = security_manager.remove_policy("temp_policy")
        assert result is True
        assert "temp_policy" not in security_manager.policies

        # Try to remove non-existent policy
        result = security_manager.remove_policy("non_existent")
        assert result is False

    def test_get_policy(self, security_manager):
        """Test getting a security policy."""
        policy = security_manager.get_policy("medium")
        assert policy is not None
        assert policy.name == "medium"
        assert policy.security_level == SecurityLevel.MEDIUM

        # Test non-existent policy
        policy = security_manager.get_policy("non_existent")
        assert policy is None

    def test_list_policies(self, security_manager):
        """Test listing all policies."""
        policies = security_manager.list_policies()
        assert len(policies) == 4
        assert "low" in policies
        assert "medium" in policies
        assert "high" in policies
        assert "critical" in policies

    def test_register_custom_validator(self, security_manager):
        """Test registering a custom validator."""

        async def custom_validator(request, policy):
            return True

        result = security_manager.register_custom_validator(
            "test_validator", custom_validator
        )
        assert result is True
        assert "test_validator" in security_manager.custom_validators

    @pytest.mark.asyncio
    async def test_validate_tool_execution_allowed(
        self, security_manager, sample_context
    ):
        """Test tool validation when execution is allowed."""
        request = ValidationRequest(
            context=sample_context,
            tool_parameters={"param1": "value1"},
            requested_permissions={PermissionType.READ},
        )

        response = await security_manager.validate_tool_execution(request, "low")

        assert response.result == ValidationResult.ALLOWED
        assert response.policy_applied == "low"
        assert PermissionType.READ in response.allowed_permissions

    @pytest.mark.asyncio
    async def test_validate_tool_execution_denied_permission(
        self, security_manager, sample_context
    ):
        """Test tool validation when permission is denied."""
        request = ValidationRequest(
            context=sample_context,
            tool_parameters={"param1": "value1"},
            requested_permissions={PermissionType.SYSTEM},  # Not allowed in low policy
        )

        response = await security_manager.validate_tool_execution(request, "low")

        assert response.result == ValidationResult.DENIED
        assert "Missing permissions" in response.message

    @pytest.mark.asyncio
    async def test_validate_tool_execution_requires_approval(
        self, security_manager, sample_context
    ):
        """Test tool validation when approval is required."""
        request = ValidationRequest(
            context=sample_context,
            tool_parameters={"param1": "value1"},
            requested_permissions={PermissionType.READ},
        )

        response = await security_manager.validate_tool_execution(request, "high")

        assert response.result == ValidationResult.REQUIRES_APPROVAL
        assert response.policy_applied == "high"
        assert len(security_manager.approval_queue) == 1

    @pytest.mark.asyncio
    async def test_validate_tool_execution_rate_limited(
        self, security_manager, sample_context
    ):
        """Test tool validation when rate limited."""
        request = ValidationRequest(
            context=sample_context,
            tool_parameters={"param1": "value1"},
            requested_permissions={PermissionType.READ},
        )

        # Execute multiple times to hit rate limit
        policy = security_manager.get_policy("low")
        for _ in range(policy.rate_limit_per_minute):
            await security_manager.validate_tool_execution(request, "low")

        # Next execution should be rate limited
        response = await security_manager.validate_tool_execution(request, "low")
        assert response.result == ValidationResult.RATE_LIMITED

    @pytest.mark.asyncio
    async def test_validate_tool_execution_input_size_exceeded(
        self, security_manager, sample_context
    ):
        """Test tool validation when input size is exceeded."""
        large_input = "x" * (1024 * 1024 + 1)  # Exceed 1MB limit
        request = ValidationRequest(
            context=sample_context,
            tool_parameters={"param1": "value1"},
            requested_permissions={PermissionType.READ},
            input_data=large_input,
        )

        response = await security_manager.validate_tool_execution(request, "low")

        assert response.result == ValidationResult.DENIED
        assert "Input size exceeds limit" in response.message

    @pytest.mark.asyncio
    async def test_validate_tool_execution_blocked_file_path(
        self, security_manager, sample_context
    ):
        """Test tool validation with blocked file path."""
        # Modify policy to block certain file patterns
        policy = security_manager.get_policy("low")
        policy.blocked_file_patterns = [r"/etc/.*"]

        request = ValidationRequest(
            context=sample_context,
            tool_parameters={"param1": "value1"},
            requested_permissions={PermissionType.READ},
            target_resources=["file:///etc/passwd"],
        )

        response = await security_manager.validate_tool_execution(request, "low")

        assert response.result == ValidationResult.DENIED
        assert "File path not allowed" in response.message

    @pytest.mark.asyncio
    async def test_validate_tool_execution_blocked_domain(
        self, security_manager, sample_context
    ):
        """Test tool validation with blocked domain."""
        # Modify policy to block certain domains
        policy = security_manager.get_policy("low")
        policy.blocked_domains = {"malicious.com"}

        request = ValidationRequest(
            context=sample_context,
            tool_parameters={"param1": "value1"},
            requested_permissions={PermissionType.NETWORK},
            target_resources=["https://malicious.com/api"],
        )

        response = await security_manager.validate_tool_execution(request, "low")

        assert response.result == ValidationResult.DENIED
        assert "Domain not allowed" in response.message

    @pytest.mark.asyncio
    async def test_validate_tool_execution_custom_validator_fail(
        self, security_manager, sample_context
    ):
        """Test tool validation when custom validator fails."""

        # Register a failing custom validator
        async def failing_validator(request, policy):
            return False

        security_manager.register_custom_validator(
            "failing_validator", failing_validator
        )

        # Modify policy to use the custom validator
        policy = security_manager.get_policy("low")
        policy.custom_validators = ["failing_validator"]

        request = ValidationRequest(
            context=sample_context,
            tool_parameters={"param1": "value1"},
            requested_permissions={PermissionType.READ},
        )

        response = await security_manager.validate_tool_execution(request, "low")

        assert response.result == ValidationResult.DENIED
        assert "Custom validator 'failing_validator' failed" in response.message

    @pytest.mark.asyncio
    async def test_validate_tool_execution_invalid_policy(
        self, security_manager, sample_context
    ):
        """Test tool validation with invalid policy."""
        request = ValidationRequest(
            context=sample_context,
            tool_parameters={"param1": "value1"},
            requested_permissions={PermissionType.READ},
        )

        response = await security_manager.validate_tool_execution(
            request, "non_existent"
        )

        assert response.result == ValidationResult.DENIED
        assert "Security policy 'non_existent' not found" in response.message

    @pytest.mark.asyncio
    async def test_get_security_status(self, security_manager):
        """Test getting security status."""
        status = await security_manager.get_security_status()

        assert "policies_count" in status
        assert "policies" in status
        assert "custom_validators_count" in status
        assert "approval_queue_size" in status
        assert status["policies_count"] == 4
        assert status["rate_limiter_active"] is True

    @pytest.mark.asyncio
    async def test_clear_approval_queue(self, security_manager, sample_context):
        """Test clearing the approval queue."""
        # Add some items to the queue
        request = ValidationRequest(
            context=sample_context,
            tool_parameters={"param1": "value1"},
            requested_permissions={PermissionType.READ},
        )

        # Add to queue by triggering approval requirement
        await security_manager.validate_tool_execution(request, "high")
        await security_manager.validate_tool_execution(request, "high")

        assert len(security_manager.approval_queue) == 2

        # Clear the queue
        count = await security_manager.clear_approval_queue()
        assert count == 2
        assert len(security_manager.approval_queue) == 0

    def test_get_approval_queue(self, security_manager):
        """Test getting the approval queue."""
        # Initially empty
        queue = security_manager.get_approval_queue()
        assert len(queue) == 0

        # Add an item directly
        context = SecurityContext(
            user_id="user123",
            session_id="session456",
            tool_name="test_tool",
            tool_version="1.0.0",
            execution_id="exec789",
            timestamp=time.time(),
        )
        request = ValidationRequest(
            context=context,
            tool_parameters={"param1": "value1"},
            requested_permissions={PermissionType.READ},
        )
        security_manager.approval_queue.append(request)

        queue = security_manager.get_approval_queue()
        assert len(queue) == 1
        assert queue[0].context.user_id == "user123"

    def test_load_save_configuration(self, security_manager):
        """Test loading and saving configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_file = f.name

        try:
            # Save configuration
            result = security_manager.save_configuration(config_file)
            assert result is True

            # Verify file exists and has content
            assert Path(config_file).exists()

            # Create new manager and load configuration
            new_manager = ARKSecurityManager()
            result = new_manager.load_configuration(config_file)
            assert result is True

            # Verify policies were loaded
            assert len(new_manager.policies) >= 4

        finally:
            # Clean up
            Path(config_file).unlink(missing_ok=True)


class TestSecurityManagerIntegration:
    """Integration tests for the security manager."""

    @pytest.mark.asyncio
    async def test_complete_validation_workflow(self):
        """Test a complete validation workflow."""
        security_manager = ARKSecurityManager()

        # Create a security context
        context = SecurityContext(
            user_id="integration_user",
            session_id="integration_session",
            tool_name="integration_tool",
            tool_version="1.0.0",
            execution_id="integration_exec",
            timestamp=time.time(),
        )

        # Create a validation request
        request = ValidationRequest(
            context=context,
            tool_parameters={"action": "read_file", "file_path": "/tmp/test.txt"},
            requested_permissions={PermissionType.READ, PermissionType.FILESYSTEM},
            target_resources=["file:///tmp/test.txt"],
        )

        # Validate with medium policy
        response = await security_manager.validate_tool_execution(request, "medium")

        # Should be allowed
        assert response.result == ValidationResult.ALLOWED
        assert response.policy_applied == "medium"
        assert PermissionType.READ in response.allowed_permissions
        assert PermissionType.FILESYSTEM in response.allowed_permissions

        # Check security status
        status = await security_manager.get_security_status()
        assert status["policies_count"] >= 4
        assert status["rate_limiter_active"] is True

    @pytest.mark.asyncio
    async def test_multi_user_rate_limiting(self):
        """Test rate limiting across multiple users."""
        security_manager = ARKSecurityManager()

        # Create contexts for different users
        user1_context = SecurityContext(
            user_id="user1",
            session_id="session1",
            tool_name="test_tool",
            tool_version="1.0.0",
            execution_id="exec1",
            timestamp=time.time(),
        )

        user2_context = SecurityContext(
            user_id="user2",
            session_id="session2",
            tool_name="test_tool",
            tool_version="1.0.0",
            execution_id="exec2",
            timestamp=time.time(),
        )

        # Create requests
        user1_request = ValidationRequest(
            context=user1_context,
            tool_parameters={"action": "test"},
            requested_permissions={PermissionType.READ},
        )

        user2_request = ValidationRequest(
            context=user2_context,
            tool_parameters={"action": "test"},
            requested_permissions={PermissionType.READ},
        )

        # User 1 hits rate limit
        policy = security_manager.get_policy("low")
        for _ in range(policy.rate_limit_per_minute):
            response = await security_manager.validate_tool_execution(
                user1_request, "low"
            )
            assert response.result == ValidationResult.ALLOWED

        # User 1 should now be rate limited
        response = await security_manager.validate_tool_execution(user1_request, "low")
        assert response.result == ValidationResult.RATE_LIMITED

        # User 2 should still be allowed
        response = await security_manager.validate_tool_execution(user2_request, "low")
        assert response.result == ValidationResult.ALLOWED

    @pytest.mark.asyncio
    async def test_policy_hierarchy_enforcement(self):
        """Test that different security policies are enforced correctly."""
        security_manager = ARKSecurityManager()

        context = SecurityContext(
            user_id="test_user",
            session_id="test_session",
            tool_name="test_tool",
            tool_version="1.0.0",
            execution_id="test_exec",
            timestamp=time.time(),
        )

        # Test system permission request
        system_request = ValidationRequest(
            context=context,
            tool_parameters={"action": "system_call"},
            requested_permissions={PermissionType.SYSTEM},
        )

        # Should be denied in low policy
        response = await security_manager.validate_tool_execution(system_request, "low")
        assert response.result == ValidationResult.DENIED

        # Should be denied in medium policy
        response = await security_manager.validate_tool_execution(
            system_request, "medium"
        )
        assert response.result == ValidationResult.DENIED

        # Should require approval in high policy
        response = await security_manager.validate_tool_execution(
            system_request, "high"
        )
        assert (
            response.result == ValidationResult.DENIED
        )  # Still denied as SYSTEM not in high policy

        # Should require approval in critical policy
        response = await security_manager.validate_tool_execution(
            system_request, "critical"
        )
        assert response.result == ValidationResult.REQUIRES_APPROVAL
