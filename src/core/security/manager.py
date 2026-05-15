"""
Tool Validation Security Manager for ARK Engine.

This module provides comprehensive security validation and management
for tool execution, including permission checking, input validation,
rate limiting, and audit logging.
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SecurityLevel(Enum):
    """Security levels for tool execution."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PermissionType(Enum):
    """Types of permissions for tool execution."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    SYSTEM = "system"
    EXTERNAL_API = "external_api"


class ValidationResult(Enum):
    """Results of security validation."""

    ALLOWED = "allowed"
    DENIED = "denied"
    REQUIRES_APPROVAL = "requires_approval"
    RATE_LIMITED = "rate_limited"


@dataclass
class SecurityPolicy:
    """
    Security policy configuration for tool execution.

    Defines the security rules and constraints for tool validation.
    """

    name: str
    description: str
    security_level: SecurityLevel
    allowed_permissions: set[PermissionType]
    denied_permissions: set[PermissionType] = field(default_factory=set)
    max_execution_time: float = 30.0  # seconds
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    requires_approval: bool = False
    allowed_domains: set[str] = field(default_factory=set)
    blocked_domains: set[str] = field(default_factory=set)
    allowed_file_patterns: list[str] = field(default_factory=list)
    blocked_file_patterns: list[str] = field(default_factory=list)
    max_input_size: int = 1024 * 1024  # 1MB
    custom_validators: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert policy to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "security_level": self.security_level.value,
            "allowed_permissions": [p.value for p in self.allowed_permissions],
            "denied_permissions": [p.value for p in self.denied_permissions],
            "max_execution_time": self.max_execution_time,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "rate_limit_per_hour": self.rate_limit_per_hour,
            "requires_approval": self.requires_approval,
            "allowed_domains": list(self.allowed_domains),
            "blocked_domains": list(self.blocked_domains),
            "allowed_file_patterns": self.allowed_file_patterns,
            "blocked_file_patterns": self.blocked_file_patterns,
            "max_input_size": self.max_input_size,
            "custom_validators": self.custom_validators,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SecurityPolicy":
        """Create policy from dictionary representation."""
        return cls(
            name=data["name"],
            description=data["description"],
            security_level=SecurityLevel(data["security_level"]),
            allowed_permissions={
                PermissionType(p) for p in data["allowed_permissions"]
            },
            denied_permissions={
                PermissionType(p) for p in data.get("denied_permissions", [])
            },
            max_execution_time=data.get("max_execution_time", 30.0),
            rate_limit_per_minute=data.get("rate_limit_per_minute", 60),
            rate_limit_per_hour=data.get("rate_limit_per_hour", 1000),
            requires_approval=data.get("requires_approval", False),
            allowed_domains=set(data.get("allowed_domains", [])),
            blocked_domains=set(data.get("blocked_domains", [])),
            allowed_file_patterns=data.get("allowed_file_patterns", []),
            blocked_file_patterns=data.get("blocked_file_patterns", []),
            max_input_size=data.get("max_input_size", 1024 * 1024),
            custom_validators=data.get("custom_validators", []),
        )


@dataclass
class SecurityContext:
    """
    Security context for tool execution.

    Contains information about the execution environment and user context.
    """

    user_id: str
    session_id: str
    tool_name: str
    tool_version: str
    execution_id: str
    timestamp: float
    source_ip: str | None = None
    user_agent: str | None = None
    permissions_granted: set[PermissionType] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary representation."""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "execution_id": self.execution_id,
            "timestamp": self.timestamp,
            "source_ip": self.source_ip,
            "user_agent": self.user_agent,
            "permissions_granted": [p.value for p in self.permissions_granted],
            "metadata": self.metadata,
        }


@dataclass
class ValidationRequest:
    """
    Request for tool validation.

    Contains all information needed to validate a tool execution request.
    """

    context: SecurityContext
    tool_parameters: dict[str, Any]
    requested_permissions: set[PermissionType]
    input_data: Any | None = None
    target_resources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResponse:
    """
    Response from tool validation.

    Contains the validation result and any additional information.
    """

    result: ValidationResult
    policy_applied: str
    message: str
    allowed_permissions: set[PermissionType] = field(default_factory=set)
    denied_permissions: set[PermissionType] = field(default_factory=set)
    execution_constraints: dict[str, Any] = field(default_factory=dict)
    audit_log_id: str | None = None
    retry_after: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert response to dictionary representation."""
        return {
            "result": self.result.value,
            "policy_applied": self.policy_applied,
            "message": self.message,
            "allowed_permissions": [p.value for p in self.allowed_permissions],
            "denied_permissions": [p.value for p in self.denied_permissions],
            "execution_constraints": self.execution_constraints,
            "audit_log_id": self.audit_log_id,
            "retry_after": self.retry_after,
        }


class RateLimiter:
    """
    Rate limiter for tool execution.

    Implements sliding window rate limiting with per-user tracking.
    """

    def __init__(self):
        """Initialize the rate limiter."""
        self.user_windows: dict[str, dict[str, list[float]]] = {}
        self.lock = asyncio.Lock()

    async def check_rate_limit(
        self, user_id: str, tool_name: str, per_minute_limit: int, per_hour_limit: int
    ) -> bool:
        """
        Check if the user is within rate limits for the tool.

        Args:
            user_id: User identifier
            tool_name: Name of the tool
            per_minute_limit: Maximum executions per minute
            per_hour_limit: Maximum executions per hour

        Returns:
            bool: True if within limits, False otherwise
        """
        async with self.lock:
            current_time = time.time()

            # Initialize user tracking if needed
            if user_id not in self.user_windows:
                self.user_windows[user_id] = {}

            if tool_name not in self.user_windows[user_id]:
                self.user_windows[user_id][tool_name] = []

            executions = self.user_windows[user_id][tool_name]

            # Clean old executions
            minute_ago = current_time - 60
            hour_ago = current_time - 3600

            executions[:] = [t for t in executions if t > hour_ago]

            # Check limits
            recent_minute = [t for t in executions if t > minute_ago]

            if len(recent_minute) >= per_minute_limit:
                return False

            if len(executions) >= per_hour_limit:
                return False

            # Record this execution
            executions.append(current_time)
            return True

    async def get_rate_limit_status(
        self, user_id: str, tool_name: str
    ) -> dict[str, Any]:
        """
        Get current rate limit status for a user and tool.

        Args:
            user_id: User identifier
            tool_name: Name of the tool

        Returns:
            Dict containing rate limit status information
        """
        async with self.lock:
            current_time = time.time()

            if (
                user_id not in self.user_windows
                or tool_name not in self.user_windows[user_id]
            ):
                return {
                    "executions_last_minute": 0,
                    "executions_last_hour": 0,
                    "next_reset_minute": current_time + 60,
                    "next_reset_hour": current_time + 3600,
                }

            executions = self.user_windows[user_id][tool_name]
            minute_ago = current_time - 60
            hour_ago = current_time - 3600

            recent_minute = [t for t in executions if t > minute_ago]
            recent_hour = [t for t in executions if t > hour_ago]

            return {
                "executions_last_minute": len(recent_minute),
                "executions_last_hour": len(recent_hour),
                "next_reset_minute": min(
                    [t + 60 for t in recent_minute], default=current_time
                ),
                "next_reset_hour": min(
                    [t + 3600 for t in recent_hour], default=current_time
                ),
            }


class InputValidator:
    """
    Input validator for tool parameters and data.

    Provides various validation methods for different types of input.
    """

    @staticmethod
    def validate_size(data: Any, max_size: int) -> bool:
        """
        Validate that data size is within limits.

        Args:
            data: Data to validate
            max_size: Maximum allowed size in bytes

        Returns:
            bool: True if within size limit
        """
        try:
            if isinstance(data, str):
                size = len(data.encode("utf-8"))
            elif isinstance(data, (dict, list)):
                size = len(json.dumps(data).encode("utf-8"))
            elif isinstance(data, bytes):
                size = len(data)
            else:
                size = len(str(data).encode("utf-8"))

            return size <= max_size
        except Exception:
            return False

    @staticmethod
    def validate_file_path(
        path: str, allowed_patterns: list[str], blocked_patterns: list[str]
    ) -> bool:
        """
        Validate file path against allowed and blocked patterns.

        Args:
            path: File path to validate
            allowed_patterns: List of allowed regex patterns
            blocked_patterns: List of blocked regex patterns

        Returns:
            bool: True if path is allowed
        """
        # Check blocked patterns first
        for pattern in blocked_patterns:
            if re.match(pattern, path):
                return False

        # If no allowed patterns, allow by default (unless blocked)
        if not allowed_patterns:
            return True

        # Check allowed patterns
        for pattern in allowed_patterns:
            if re.match(pattern, path):
                return True

        return False

    @staticmethod
    def validate_domain(
        domain: str, allowed_domains: set[str], blocked_domains: set[str]
    ) -> bool:
        """
        Validate domain against allowed and blocked lists.

        Args:
            domain: Domain to validate
            allowed_domains: Set of allowed domains
            blocked_domains: Set of blocked domains

        Returns:
            bool: True if domain is allowed
        """
        # Check blocked domains first
        if domain in blocked_domains:
            return False

        # Check for wildcard blocked domains
        for blocked in blocked_domains:
            if blocked.startswith("*.") and domain.endswith(blocked[2:]):
                return False

        # If no allowed domains, allow by default (unless blocked)
        if not allowed_domains:
            return True

        # Check allowed domains
        if domain in allowed_domains:
            return True

        # Check for wildcard allowed domains
        for allowed in allowed_domains:
            if allowed.startswith("*.") and domain.endswith(allowed[2:]):
                return True

        return False

    @staticmethod
    def sanitize_input(data: Any) -> Any:
        """
        Sanitize input data to remove potentially dangerous content.

        Args:
            data: Data to sanitize

        Returns:
            Sanitized data
        """
        if isinstance(data, str):
            # Remove potentially dangerous characters
            dangerous_chars = ["<", ">", "&", '"', "'", "\x00"]
            for char in dangerous_chars:
                data = data.replace(char, "")
            return data
        elif isinstance(data, dict):
            return {k: InputValidator.sanitize_input(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [InputValidator.sanitize_input(item) for item in data]
        else:
            return data


class AuditLogger:
    """
    Audit logger for security events.

    Logs all security-related events for compliance and monitoring.
    """

    def __init__(self, log_file: str | None = None):
        """
        Initialize the audit logger.

        Args:
            log_file: Optional file path for audit logs
        """
        self.logger = logging.getLogger("ark.security.audit")
        self.log_file = log_file

        if log_file:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _redact(self, value: Any) -> Any:
        sensitive_keys = {
            "api_key",
            "token",
            "secret",
            "password",
            "authorization",
            "auth",
            "cookie",
            "session_id",
        }
        if isinstance(value, dict):
            redacted: dict[str, Any] = {}
            for k, v in value.items():
                key_lower = str(k).lower()
                if any(s in key_lower for s in sensitive_keys):
                    redacted[k] = "***REDACTED***"
                else:
                    redacted[k] = self._redact(v)
            return redacted
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        return value

    def log_validation_request(self, request: ValidationRequest) -> str:
        """
        Log a validation request.

        Args:
            request: Validation request to log

        Returns:
            str: Audit log ID
        """
        audit_id = self._generate_audit_id(request.context)

        log_data = {
            "audit_id": audit_id,
            "event_type": "validation_request",
            "context": self._redact(request.context.to_dict()),
            "requested_permissions": [p.value for p in request.requested_permissions],
            "target_resources": self._redact(request.target_resources),
            "metadata": self._redact(request.metadata),
            "redacted": True,
        }

        self.logger.info(f"Validation request: {json.dumps(log_data)}")
        return audit_id

    def log_validation_response(self, response: ValidationResponse, audit_id: str):
        """
        Log a validation response.

        Args:
            response: Validation response to log
            audit_id: Associated audit ID
        """
        log_data = {
            "audit_id": audit_id,
            "event_type": "validation_response",
            "result": response.result.value,
            "policy_applied": response.policy_applied,
            "message": response.message,
            "allowed_permissions": [p.value for p in response.allowed_permissions],
            "denied_permissions": [p.value for p in response.denied_permissions],
        }

        self.logger.info(f"Validation response: {json.dumps(log_data)}")

    def log_security_violation(
        self, context: SecurityContext, violation_type: str, details: str
    ):
        """
        Log a security violation.

        Args:
            context: Security context
            violation_type: Type of violation
            details: Violation details
        """
        audit_id = self._generate_audit_id(context)

        log_data = {
            "audit_id": audit_id,
            "event_type": "security_violation",
            "violation_type": violation_type,
            "context": self._redact(context.to_dict()),
            "details": self._redact(details),
            "redacted": True,
        }

        self.logger.warning(f"Security violation: {json.dumps(log_data)}")

    def _generate_audit_id(self, context: SecurityContext) -> str:
        """
        Generate a unique audit ID.

        Args:
            context: Security context

        Returns:
            str: Unique audit ID
        """
        data = f"{context.user_id}:{context.session_id}:{context.execution_id}:{context.timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class ARKSecurityManager:
    """
    Main security manager for ARK tool validation.

    Coordinates all security validation activities including policy enforcement,
    rate limiting, input validation, and audit logging.
    """

    def __init__(
        self, config_file: str | None = None, audit_log_file: str | None = None
    ):
        """
        Initialize the security manager.

        Args:
            config_file: Optional configuration file path
            audit_log_file: Optional audit log file path
        """
        self.logger = logging.getLogger(__name__)
        self.policies: dict[str, SecurityPolicy] = {}
        self.rate_limiter = RateLimiter()
        self.input_validator = InputValidator()
        self.audit_logger = AuditLogger(audit_log_file)
        self.custom_validators: dict[str, Callable] = {}
        self.approval_queue: list[ValidationRequest] = []

        # Load default policies
        self._load_default_policies()

        # Load configuration if provided
        if config_file:
            self.load_configuration(config_file)

    def _load_default_policies(self):
        """Load default security policies."""
        # Low security policy
        self.policies["low"] = SecurityPolicy(
            name="low",
            description="Low security policy for safe operations",
            security_level=SecurityLevel.LOW,
            allowed_permissions={
                PermissionType.READ,
                PermissionType.NETWORK,
                PermissionType.EXTERNAL_API,
            },
            max_execution_time=10.0,
            rate_limit_per_minute=30,
            rate_limit_per_hour=500,
            requires_approval=False,
        )

        # Medium security policy
        self.policies["medium"] = SecurityPolicy(
            name="medium",
            description="Medium security policy for general operations",
            security_level=SecurityLevel.MEDIUM,
            allowed_permissions={
                PermissionType.READ,
                PermissionType.WRITE,
                PermissionType.FILESYSTEM,
                PermissionType.NETWORK,
                PermissionType.EXTERNAL_API,
            },
            max_execution_time=30.0,
            rate_limit_per_minute=20,
            rate_limit_per_hour=300,
            requires_approval=False,
        )

        # High security policy
        self.policies["high"] = SecurityPolicy(
            name="high",
            description="High security policy for sensitive operations",
            security_level=SecurityLevel.HIGH,
            allowed_permissions={
                PermissionType.READ,
                PermissionType.WRITE,
                PermissionType.EXECUTE,
                PermissionType.NETWORK,
                PermissionType.FILESYSTEM,
                PermissionType.EXTERNAL_API,
            },
            max_execution_time=60.0,
            rate_limit_per_minute=10,
            rate_limit_per_hour=100,
            requires_approval=True,
        )

        # Critical security policy
        self.policies["critical"] = SecurityPolicy(
            name="critical",
            description="Critical security policy for system operations",
            security_level=SecurityLevel.CRITICAL,
            allowed_permissions={
                PermissionType.READ,
                PermissionType.WRITE,
                PermissionType.EXECUTE,
                PermissionType.NETWORK,
                PermissionType.FILESYSTEM,
                PermissionType.SYSTEM,
                PermissionType.EXTERNAL_API,
            },
            max_execution_time=120.0,
            rate_limit_per_minute=5,
            rate_limit_per_hour=50,
            requires_approval=True,
        )

    def load_configuration(self, config_file: str) -> bool:
        """
        Load security configuration from file.

        Args:
            config_file: Path to configuration file

        Returns:
            bool: True if loaded successfully
        """
        try:
            with open(config_file) as f:
                config = json.load(f)

            # Load policies
            if "policies" in config:
                for policy_data in config["policies"]:
                    policy = SecurityPolicy.from_dict(policy_data)
                    self.policies[policy.name] = policy

            self.logger.info(f"Loaded security configuration from {config_file}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load security configuration: {e}")
            return False

    def save_configuration(self, config_file: str) -> bool:
        """
        Save security configuration to file.

        Args:
            config_file: Path to configuration file

        Returns:
            bool: True if saved successfully
        """
        try:
            config = {
                "policies": [policy.to_dict() for policy in self.policies.values()]
            }

            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)

            self.logger.info(f"Saved security configuration to {config_file}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save security configuration: {e}")
            return False

    def add_policy(self, policy: SecurityPolicy) -> bool:
        """
        Add a security policy.

        Args:
            policy: Security policy to add

        Returns:
            bool: True if added successfully
        """
        try:
            self.policies[policy.name] = policy
            self.logger.info(f"Added security policy: {policy.name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add security policy: {e}")
            return False

    def remove_policy(self, policy_name: str) -> bool:
        """
        Remove a security policy.

        Args:
            policy_name: Name of policy to remove

        Returns:
            bool: True if removed successfully
        """
        try:
            if policy_name in self.policies:
                del self.policies[policy_name]
                self.logger.info(f"Removed security policy: {policy_name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to remove security policy: {e}")
            return False

    def get_policy(self, policy_name: str) -> SecurityPolicy | None:
        """
        Get a security policy by name.

        Args:
            policy_name: Name of policy to get

        Returns:
            SecurityPolicy or None if not found
        """
        return self.policies.get(policy_name)

    def list_policies(self) -> list[str]:
        """
        List all available security policies.

        Returns:
            List of policy names
        """
        return list(self.policies.keys())

    def register_custom_validator(self, name: str, validator: Callable) -> bool:
        """
        Register a custom validator function.

        Args:
            name: Name of the validator
            validator: Validator function

        Returns:
            bool: True if registered successfully
        """
        try:
            self.custom_validators[name] = validator
            self.logger.info(f"Registered custom validator: {name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register custom validator: {e}")
            return False

    async def validate_tool_execution(
        self, request: ValidationRequest, policy_name: str = "medium"
    ) -> ValidationResponse:
        """
        Validate a tool execution request.

        Args:
            request: Validation request
            policy_name: Name of security policy to apply

        Returns:
            ValidationResponse with validation result
        """
        try:
            # Log the validation request
            audit_id = self.audit_logger.log_validation_request(request)

            # Get the security policy
            policy = self.policies.get(policy_name)
            if not policy:
                return ValidationResponse(
                    result=ValidationResult.DENIED,
                    policy_applied=policy_name,
                    message=f"Security policy '{policy_name}' not found",
                    audit_log_id=audit_id,
                )

            # Check rate limits
            rate_limit_ok = await self.rate_limiter.check_rate_limit(
                request.context.user_id,
                request.context.tool_name,
                policy.rate_limit_per_minute,
                policy.rate_limit_per_hour,
            )

            if not rate_limit_ok:
                response = ValidationResponse(
                    result=ValidationResult.RATE_LIMITED,
                    policy_applied=policy_name,
                    message="Rate limit exceeded",
                    audit_log_id=audit_id,
                    retry_after=60.0,  # Retry after 1 minute
                )
                self.audit_logger.log_validation_response(response, audit_id)
                return response

            # Check permissions
            denied_permissions = (
                request.requested_permissions & policy.denied_permissions
            )
            if denied_permissions:
                response = ValidationResponse(
                    result=ValidationResult.DENIED,
                    policy_applied=policy_name,
                    message=f"Denied permissions: {[p.value for p in denied_permissions]}",
                    denied_permissions=denied_permissions,
                    audit_log_id=audit_id,
                )
                self.audit_logger.log_validation_response(response, audit_id)
                return response

            allowed_permissions = (
                request.requested_permissions & policy.allowed_permissions
            )
            if allowed_permissions != request.requested_permissions:
                missing_permissions = (
                    request.requested_permissions - allowed_permissions
                )
                response = ValidationResponse(
                    result=ValidationResult.DENIED,
                    policy_applied=policy_name,
                    message=f"Missing permissions: {[p.value for p in missing_permissions]}",
                    denied_permissions=missing_permissions,
                    audit_log_id=audit_id,
                )
                self.audit_logger.log_validation_response(response, audit_id)
                return response

            # Validate input size
            if request.input_data and not self.input_validator.validate_size(
                request.input_data, policy.max_input_size
            ):
                response = ValidationResponse(
                    result=ValidationResult.DENIED,
                    policy_applied=policy_name,
                    message=f"Input size exceeds limit of {policy.max_input_size} bytes",
                    audit_log_id=audit_id,
                )
                self.audit_logger.log_validation_response(response, audit_id)
                return response

            # Validate file paths
            for resource in request.target_resources:
                if resource.startswith("file://"):
                    file_path = resource[7:]  # Remove 'file://' prefix
                    if not self.input_validator.validate_file_path(
                        file_path,
                        policy.allowed_file_patterns,
                        policy.blocked_file_patterns,
                    ):
                        response = ValidationResponse(
                            result=ValidationResult.DENIED,
                            policy_applied=policy_name,
                            message=f"File path not allowed: {file_path}",
                            audit_log_id=audit_id,
                        )
                        self.audit_logger.log_validation_response(response, audit_id)
                        return response

            # Validate domains
            for resource in request.target_resources:
                if resource.startswith("http://") or resource.startswith("https://"):
                    from urllib.parse import urlparse

                    domain = urlparse(resource).netloc
                    if not self.input_validator.validate_domain(
                        domain, policy.allowed_domains, policy.blocked_domains
                    ):
                        response = ValidationResponse(
                            result=ValidationResult.DENIED,
                            policy_applied=policy_name,
                            message=f"Domain not allowed: {domain}",
                            audit_log_id=audit_id,
                        )
                        self.audit_logger.log_validation_response(response, audit_id)
                        return response

            # Run custom validators
            for validator_name in policy.custom_validators:
                if validator_name in self.custom_validators:
                    try:
                        validator_result = await self.custom_validators[validator_name](
                            request, policy
                        )
                        if not validator_result:
                            response = ValidationResponse(
                                result=ValidationResult.DENIED,
                                policy_applied=policy_name,
                                message=f"Custom validator '{validator_name}' failed",
                                audit_log_id=audit_id,
                            )
                            self.audit_logger.log_validation_response(
                                response, audit_id
                            )
                            return response
                    except Exception as e:
                        self.logger.error(
                            f"Custom validator '{validator_name}' error: {e}"
                        )
                        response = ValidationResponse(
                            result=ValidationResult.DENIED,
                            policy_applied=policy_name,
                            message=f"Custom validator '{validator_name}' error",
                            audit_log_id=audit_id,
                        )
                        self.audit_logger.log_validation_response(response, audit_id)
                        return response

            # Check if approval is required
            if policy.requires_approval:
                self.approval_queue.append(request)
                response = ValidationResponse(
                    result=ValidationResult.REQUIRES_APPROVAL,
                    policy_applied=policy_name,
                    message="Tool execution requires approval",
                    allowed_permissions=allowed_permissions,
                    execution_constraints={
                        "max_execution_time": policy.max_execution_time
                    },
                    audit_log_id=audit_id,
                )
                self.audit_logger.log_validation_response(response, audit_id)
                return response

            # All validations passed
            response = ValidationResponse(
                result=ValidationResult.ALLOWED,
                policy_applied=policy_name,
                message="Tool execution allowed",
                allowed_permissions=allowed_permissions,
                execution_constraints={"max_execution_time": policy.max_execution_time},
                audit_log_id=audit_id,
            )
            self.audit_logger.log_validation_response(response, audit_id)
            return response

        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return ValidationResponse(
                result=ValidationResult.DENIED,
                policy_applied=policy_name,
                message=f"Validation error: {str(e)}",
            )

    async def get_security_status(self) -> dict[str, Any]:
        """
        Get current security manager status.

        Returns:
            Dict containing security status information
        """
        return {
            "policies_count": len(self.policies),
            "policies": list(self.policies.keys()),
            "custom_validators_count": len(self.custom_validators),
            "custom_validators": list(self.custom_validators.keys()),
            "approval_queue_size": len(self.approval_queue),
            "rate_limiter_active": True,
        }

    async def clear_approval_queue(self) -> int:
        """
        Clear the approval queue.

        Returns:
            int: Number of items cleared
        """
        count = len(self.approval_queue)
        self.approval_queue.clear()
        self.logger.info(f"Cleared {count} items from approval queue")
        return count

    def get_approval_queue(self) -> list[ValidationRequest]:
        """
        Get current approval queue.

        Returns:
            List of validation requests awaiting approval
        """
        return self.approval_queue.copy()
