"""
覆盖补全测试 - 主要模块边界情况

针对 security/manager.py, jarvis_agent.py, factory.py, prompt/manager.py,
exceptions.py, tasks.py, config.py 等模块的未覆盖边界。
"""

from unittest.mock import MagicMock

import pytest

# ═══════════════════════════════════════════════════════════════
# src/core/security/manager.py
# ═══════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════
# src/jarvis_agent.py
# ═══════════════════════════════════════════════════════════════


class TestJarvisAgentDetailedCoverage:
    """JarvisAgent 边界测试"""

    def test_jarvis_config_defaults(self):
        from src.jarvis_agent import JarvisConfig

        config = JarvisConfig()
        assert config.name == "Jarvis"
        assert config.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self):
        from src.jarvis_agent import JarvisAgent

        agent = JarvisAgent()
        agent.is_initialized = False
        result = await agent.health_check()
        assert "not_initialized" in str(result) or "unhealthy" in str(result)


# ═══════════════════════════════════════════════════════════════
# src/core/common/exceptions.py
# ═══════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════
# src/core/ark/tasks.py
# ═══════════════════════════════════════════════════════════════


class TestTaskDetailedCoverage:
    """Task 补充测试"""

    def test_task_with_result(self):
        from src.core.ark.tasks import Task

        task = Task(id="t1", description="test", result="done")
        assert task.result == "done"
        d = task.to_dict()
        assert d["result"] == "done"

    def test_task_from_dict_with_result(self):
        from src.core.ark.tasks import Task

        task = Task.from_dict({"id": "t1", "description": "test", "result": "ok"})
        assert task.result == "ok"

    def test_execute_manage_tasks_complete(self):
        from src.core.ark.tasks import execute_manage_tasks

        todo = [{"id": "t1", "description": "Task 1", "status": "pending"}]
        result = execute_manage_tasks({"action": "complete", "id": "t1"}, todo)
        assert "completed" in result
        assert todo[0]["status"] == "completed"

    def test_execute_manage_tasks_update_not_found(self):
        from src.core.ark.tasks import execute_manage_tasks

        todo = [{"id": "t1", "description": "Task", "status": "pending"}]
        result = execute_manage_tasks({"action": "update", "id": "nonexistent"}, todo)
        assert "not found" in result

    def test_execute_manage_tasks_unknown_action(self):
        from src.core.ark.tasks import execute_manage_tasks

        result = execute_manage_tasks({"action": "unknown"}, [])
        assert "Unknown" in result or "unknown" in result


# ═══════════════════════════════════════════════════════════════
# src/core/prompt/manager.py
# ═══════════════════════════════════════════════════════════════


class TestPromptManagerDetailedCoverage:
    """PromptManager 边界测试"""

    def test_prompt_manager_init(self):
        from src.core.prompt.manager import PromptManager

        pm = PromptManager()
        assert pm is not None
        assert hasattr(pm, "templates")

    def test_prompt_manager_list_templates(self):
        from src.core.prompt.manager import PromptManager

        pm = PromptManager()
        templates = pm.list_templates()
        assert isinstance(templates, list)


# ═══════════════════════════════════════════════════════════════
# src/core/llm/factory.py
# ═══════════════════════════════════════════════════════════════


class TestFactoryDetailedCoverage:
    """LLM Factory 补充测试"""

    def test_provider_info_str(self):
        from src.core.config.loader import ProviderConfig
        from src.core.llm.factory import ProviderInfo, ProviderType

        config = ProviderConfig(type="mock", enabled=True)
        info = ProviderInfo(
            name="test",
            type=ProviderType.OPENAI,
            client_class=MagicMock,
            enabled=True,
            config=config,
        )
        s = str(info)
        assert "test" in s or "ProviderInfo" in s

    def test_registry_not_registered(self):
        from src.core.llm.factory import LLMProviderRegistry

        registry = LLMProviderRegistry()
        assert registry.is_provider_registered("nonexistent") is False

    def test_registry_list_providers(self):
        from src.core.llm.factory import LLMProviderRegistry

        registry = LLMProviderRegistry()
        providers = registry.list_providers()
        assert isinstance(providers, list)


# ═══════════════════════════════════════════════════════════════
# src/core/ark/nodes/tools.py & master.py
# ═══════════════════════════════════════════════════════════════


class TestToolsNodeDetailedCoverage:
    """ToolsNode 边界测试"""

    def test_tools_node_create_with_mock(self):
        from src.core.ark.nodes.tools import ToolsNode

        mcp = MagicMock()
        node = ToolsNode(mcp_client=mcp)
        assert node is not None
        assert node.max_concurrency == 10


# ═══════════════════════════════════════════════════════════════
# src/core/security/manager.py 更多边界
# ═══════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════
# src/core/llm/factory.py 更多
# ═══════════════════════════════════════════════════════════════


class TestFactoryMoreCoverage:
    """LLM Factory 更多测试"""

    def test_registry_register_duplicate(self):
        from src.core.llm.client import BaseLLMClient
        from src.core.llm.factory import LLMProviderRegistry
        from src.core.llm.types import LLMConfig, LLMProvider

        class MockClient(BaseLLMClient):
            async def initialize(self):
                return True

            async def generate_response(self, messages, **kwargs):
                pass

            async def stream_response(self, messages, **kwargs):
                yield ""

        config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4")
        MockClient(config)

        registry = LLMProviderRegistry()
        registry.register_provider("mock", MockClient)
        providers_before = len(registry.list_providers())
        # Register same name again should not increase count
        registry.register_provider("mock", MockClient)
        assert len(registry.list_providers()) == providers_before

    def test_list_providers_returns_correctly(self):
        from src.core.llm.client import BaseLLMClient
        from src.core.llm.factory import LLMProviderRegistry

        class MockClient(BaseLLMClient):
            async def initialize(self):
                return True

            async def generate_response(self, messages, **kwargs):
                pass

            async def stream_response(self, messages, **kwargs):
                yield ""

        registry = LLMProviderRegistry()
        registry.register_provider("mock", MockClient)
        providers = registry.list_providers()
        assert "mock" in providers
        assert isinstance(providers, list)


# ═══════════════════════════════════════════════════════════════
# src/core/ark/nodes/master.py
# ═══════════════════════════════════════════════════════════════


class TestMasterNodeCoverage:
    """MasterNode 边界测试"""

    def test_master_node_creation(self):
        from src.core.ark.nodes.master import MasterNode

        llm = MagicMock()
        node = MasterNode(llm_manager=llm)
        assert node is not None
        assert node.agents == []
        assert node.agent_map == {}


# ═══════════════════════════════════════════════════════════════
# src/core/common/exceptions.py 更多
# ═══════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════
# src/core/llm/converters.py
# ═══════════════════════════════════════════════════════════════


class TestConvertersMoreCoverage:
    """转换器更多测试"""

    def test_convert_funtion_message(self):
        from langchain_core.messages import FunctionMessage

        from src.core.llm.converters import convert_langchain_to_llm_messages

        msg = FunctionMessage(content="result", name="func1")
        result = convert_langchain_to_llm_messages([msg])
        assert len(result) == 1
        # FunctionMessage maps to "user" role in the converter
        assert result[0].role == "user"

    def test_convert_tool_message(self):
        from langchain_core.messages import ToolMessage

        from src.core.llm.converters import convert_langchain_to_llm_messages

        msg = ToolMessage(content="tool output", tool_call_id="call1")
        result = convert_langchain_to_llm_messages([msg])
        assert len(result) == 1
        # ToolMessage maps to "user" role in the converter
        assert result[0].role == "user"


# ═══════════════════════════════════════════════════════════════
# src/core/llm/parsers.py
# ═══════════════════════════════════════════════════════════════


class TestParsersMoreCoverage:
    """Parser 更多测试"""

    def test_json_output_parser_invalid(self):
        from src.core.llm.parsers import JSONOutputParser

        parser = JSONOutputParser()
        with pytest.raises(ValueError):
            parser.parse("invalid json")

    def test_json_output_parser_markdown(self):
        from src.core.llm.parsers import JSONOutputParser

        parser = JSONOutputParser()
        result = parser.parse('```json\n{"key": "value"}\n```')
        assert result["key"] == "value"


# ═══════════════════════════════════════════════════════════════
# src/core/llm/factory.py 更多边界
# ═══════════════════════════════════════════════════════════════


class TestFactoryRemainingCoverage:
    """Factory 剩余边界测试"""

    def test_get_provider_class_nonexistent(self):
        from src.core.llm.factory import LLMProviderRegistry

        registry = LLMProviderRegistry()
        result = registry.get_provider_class("nonexistent")
        assert result is None

    def test_unregister_provider(self):
        from src.core.llm.client import BaseLLMClient
        from src.core.llm.factory import LLMProviderRegistry

        class MockClient(BaseLLMClient):
            async def initialize(self):
                return True

            async def generate_response(self, m, **k):
                pass

            async def stream_response(self, m, **k):
                yield ""

        registry = LLMProviderRegistry()
        registry.register_provider("test_client", MockClient)
        assert registry.is_provider_registered("test_client") is True
        registry.unregister_provider("test_client")
        assert registry.is_provider_registered("test_client") is False


# ═══════════════════════════════════════════════════════════════
# src/core/ark/nodes/tools.py 边界
# ═══════════════════════════════════════════════════════════════


class TestToolsNodeMethods:
    """ToolsNode 方法测试"""

    def test_tools_node_creation(self):
        from src.core.ark.nodes.tools import ToolsNode

        mcp = MagicMock()
        node = ToolsNode(mcp_client=mcp)
        assert node.max_concurrency == 10


# ═══════════════════════════════════════════════════════════════
# src/core/prompt/manager.py 边界
# ═══════════════════════════════════════════════════════════════


class TestPromptManagerRemaining:
    """PromptManager 更多测试"""

    def test_prompt_manager_list_templates(self):
        from src.core.prompt.manager import PromptManager

        pm = PromptManager()
        templates = pm.list_templates()
        assert isinstance(templates, list)
        assert len(templates) > 0

    def test_prompt_manager_str(self):
        from src.core.prompt.manager import PromptManager

        pm = PromptManager()
        s = str(pm)
        assert "PromptManager" in s


# ═══════════════════════════════════════════════════════════════
# src/core/common/exceptions.py 剩余边界
# ═══════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════
# src/core/security/manager.py 剩余边界
# ═══════════════════════════════════════════════════════════════


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
