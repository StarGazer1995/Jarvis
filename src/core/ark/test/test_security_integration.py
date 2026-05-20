"""
Tests for security integration in ToolsNode.

Verifies that tool calls are validated through the security manager
before execution, and that denied / rate-limited / approval-required
calls return normalised messages instead of executing.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.core.ark.nodes.tools import ToolsNode
from src.core.mcp.client import ARKMCPClient
from src.core.observability import MetricsRegistry
from src.core.security.manager import (
    ARKSecurityManager,
    PermissionType,
    ValidationResponse,
    ValidationResult,
)


@pytest.fixture
def mock_mcp_client():
    """Create a mock MCP client."""
    client = MagicMock(spec=ARKMCPClient)
    client.execute_tool = AsyncMock(return_value="mcp_result")
    return client


@pytest.fixture
def mock_security_manager():
    """Create a mock security manager that allows everything by default."""
    mgr = MagicMock(spec=ARKSecurityManager)
    mgr.validate_tool_execution = AsyncMock(
        return_value=ValidationResponse(
            result=ValidationResult.ALLOWED,
            policy_applied="low",
            message="Allowed",
        )
    )
    return mgr


@pytest.fixture
def tools_node(mock_mcp_client, mock_security_manager):
    """Create a ToolsNode with mocked dependencies."""
    node = ToolsNode(
        mcp_client=mock_mcp_client,
        security_manager=mock_security_manager,
    )
    return node


class TestToolsNodeSecurityIntegration:
    """Security validation integration tests for ToolsNode."""

    @pytest.mark.asyncio
    async def test_allowed_tool_execution(self, tools_node, mock_mcp_client):
        """A tool that passes security validation should execute normally."""
        # Simulate LLM response with a tool call
        last_message = AIMessage(
            content="Let me search.",
            tool_calls=[
                {
                    "name": "web_search",
                    "args": {"query": "test"},
                    "id": "call_1",
                }
            ],
        )

        state = {
            "messages": [last_message],
            "todo_list": [],
            "sender": "master",
        }

        result = await tools_node(state, config=None)

        # Tool should have been executed
        mock_mcp_client.execute_tool.assert_called_once_with(
            "web_search", {"query": "test"}
        )

        # Result should be a ToolMessage
        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert isinstance(msg, ToolMessage)
        assert msg.name == "web_search"
        assert msg.content == "mcp_result"

    @pytest.mark.asyncio
    async def test_denied_tool_execution(self, tools_node, mock_security_manager):
        """A denied tool should return a security message without executing."""
        mock_security_manager.validate_tool_execution.return_value = ValidationResponse(
            result=ValidationResult.DENIED,
            policy_applied="high",
            message="Denied: missing permission",
        )

        last_message = AIMessage(
            content="Let me execute.",
            tool_calls=[
                {
                    "name": "execute_command",
                    "args": {"command": "rm -rf /"},
                    "id": "call_1",
                }
            ],
        )

        state = {
            "messages": [last_message],
            "todo_list": [],
            "sender": "master",
        }

        with patch.object(tools_node.mcp_client, "execute_tool") as mock_exec:
            result = await tools_node(state, config=None)

            # Tool should NOT have been executed
            mock_exec.assert_not_called()

            # Result should be a security denial message
            assert len(result["messages"]) == 1
            msg = result["messages"][0]
            assert isinstance(msg, ToolMessage)
            assert "Security:" in msg.content
            assert "denied" in msg.content.lower()
            assert "execute_command" in msg.content

    @pytest.mark.asyncio
    async def test_rate_limited_tool(self, tools_node, mock_security_manager):
        """A rate-limited tool should return a retry message."""
        mock_security_manager.validate_tool_execution.return_value = ValidationResponse(
            result=ValidationResult.RATE_LIMITED,
            policy_applied="medium",
            message="Rate limit exceeded",
            retry_after=30.0,
        )

        last_message = AIMessage(
            content="Let me search.",
            tool_calls=[
                {
                    "name": "web_search",
                    "args": {"query": "test"},
                    "id": "call_1",
                }
            ],
        )

        state = {
            "messages": [last_message],
            "todo_list": [],
            "sender": "master",
        }

        with patch.object(tools_node.mcp_client, "execute_tool") as mock_exec:
            result = await tools_node(state, config=None)

            mock_exec.assert_not_called()
            assert len(result["messages"]) == 1
            msg = result["messages"][0]
            assert "rate limited" in msg.content.lower()
            assert "30" in msg.content

    @pytest.mark.asyncio
    async def test_approval_required_tool(self, tools_node, mock_security_manager):
        """An approval-required tool should return an approval message."""
        mock_security_manager.validate_tool_execution.return_value = ValidationResponse(
            result=ValidationResult.REQUIRES_APPROVAL,
            policy_applied="high",
            message="Requires approval",
        )

        last_message = AIMessage(
            content="Let me write.",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"file_path": "/tmp/test.txt", "content": "data"},
                    "id": "call_1",
                }
            ],
        )

        state = {
            "messages": [last_message],
            "todo_list": [],
            "sender": "master",
        }

        with patch.object(tools_node.mcp_client, "execute_tool") as mock_exec:
            result = await tools_node(state, config=None)

            mock_exec.assert_not_called()
            assert len(result["messages"]) == 1
            msg = result["messages"][0]
            assert "approval" in msg.content.lower()

    @pytest.mark.asyncio
    async def test_mixed_parallel_allowed_and_denied(
        self, tools_node, mock_security_manager
    ):
        """In parallel execution, allowed tools run and denied tools are skipped."""

        # Return different responses based on tool name
        async def validate_side_effect(request, policy_name="medium"):
            if request.context.tool_name == "web_search":
                return ValidationResponse(
                    result=ValidationResult.ALLOWED,
                    policy_applied="low",
                    message="Allowed",
                )
            return ValidationResponse(
                result=ValidationResult.DENIED,
                policy_applied="high",
                message="Denied",
            )

        mock_security_manager.validate_tool_execution = AsyncMock(
            side_effect=validate_side_effect
        )

        last_message = AIMessage(
            content="Do multiple things.",
            tool_calls=[
                {
                    "name": "web_search",
                    "args": {"query": "safe"},
                    "id": "call_1",
                },
                {
                    "name": "execute_command",
                    "args": {"command": "rm -rf /"},
                    "id": "call_2",
                },
            ],
        )

        state = {
            "messages": [last_message],
            "todo_list": [],
            "sender": "master",
        }

        with patch.object(tools_node.mcp_client, "execute_tool") as mock_exec:
            mock_exec.return_value = "search_result"
            result = await tools_node(state, config=None)

            # Only web_search should have executed
            mock_exec.assert_called_once_with("web_search", {"query": "safe"})

            # Both tools should have messages
            assert len(result["messages"]) == 2

            # First message: allowed tool result
            assert result["messages"][0].content == "search_result"

            # Second message: denied security message
            assert "Security:" in result["messages"][1].content

    @pytest.mark.asyncio
    async def test_no_tool_calls(self, tools_node, mock_mcp_client):
        """When there are no tool calls, ToolsNode should return early."""
        last_message = AIMessage(content="Just a thought.")

        state = {
            "messages": [last_message],
            "todo_list": [],
            "sender": "master",
        }

        result = await tools_node(state, config=None)

        mock_mcp_client.execute_tool.assert_not_called()
        assert result["sender"] == "tools"

    @pytest.mark.asyncio
    async def test_local_tool_security_check(self, tools_node, mock_security_manager):
        """Local tools should also be validated through security."""

        # Register a local tool
        async def my_local_tool(**kwargs):
            return "local_result"

        tools_node.register_tool("my_local_tool", my_local_tool)

        mock_security_manager.validate_tool_execution.return_value = ValidationResponse(
            result=ValidationResult.ALLOWED,
            policy_applied="low",
            message="Allowed",
        )

        last_message = AIMessage(
            content="Use local tool.",
            tool_calls=[
                {
                    "name": "my_local_tool",
                    "args": {"param": "value"},
                    "id": "call_1",
                }
            ],
        )

        state = {
            "messages": [last_message],
            "todo_list": [],
            "sender": "master",
        }

        result = await tools_node(state, config=None)

        # Should have called security validation
        mock_security_manager.validate_tool_execution.assert_called()

        # Should have executed the tool
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "local_result"

    @pytest.mark.asyncio
    async def test_rate_limited_with_retry_after(
        self, tools_node, mock_security_manager
    ):
        """RATE_LIMITED with retry_after should include the retry time."""
        mock_security_manager.validate_tool_execution.return_value = ValidationResponse(
            result=ValidationResult.RATE_LIMITED,
            policy_applied="medium",
            message="Rate limit exceeded",
            retry_after=120.0,
        )

        last_message = AIMessage(
            content="search",
            tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "call_1"}],
        )
        state = {"messages": [last_message], "todo_list": [], "sender": "master"}

        with patch.object(tools_node.mcp_client, "execute_tool") as mock_exec:
            result = await tools_node(state, config=None)
            mock_exec.assert_not_called()
            assert "120" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_approval_required_with_policy(
        self, tools_node, mock_security_manager
    ):
        """REQUIRES_APPROVAL should mention the tool name."""
        mock_security_manager.validate_tool_execution.return_value = ValidationResponse(
            result=ValidationResult.REQUIRES_APPROVAL,
            policy_applied="high",
            message="Requires approval",
        )

        last_message = AIMessage(
            content="write",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"file_path": "/tmp/f", "content": "data"},
                    "id": "call_1",
                }
            ],
        )
        state = {"messages": [last_message], "todo_list": [], "sender": "master"}

        with patch.object(tools_node.mcp_client, "execute_tool") as mock_exec:
            result = await tools_node(state, config=None)
            mock_exec.assert_not_called()
            assert "approval" in result["messages"][0].content.lower()
            assert "write_file" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_security_validation_exception_caught(
        self, tools_node, mock_security_manager
    ):
        """When security validation raises, the tool should be denied gracefully."""
        mock_security_manager.validate_tool_execution.side_effect = RuntimeError(
            "Unexpected validation crash"
        )

        last_message = AIMessage(
            content="search",
            tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "call_1"}],
        )
        state = {"messages": [last_message], "todo_list": [], "sender": "master"}

        with patch.object(tools_node.mcp_client, "execute_tool") as mock_exec:
            result = await tools_node(state, config=None)
            mock_exec.assert_not_called()
            # Should return a validation error message
            assert "validation error" in result["messages"][0].content.lower()

    @pytest.mark.asyncio
    async def test_unknown_validation_result(self, tools_node, mock_security_manager):
        """An unknown ValidationResult should produce a validation error message."""

        # Create a custom result object that has a .value but is not one of the
        # known ValidationResult enum members. Use a simple class to avoid
        # MagicMock's auto-equal behavior.
        class _CustomResult:
            value = "custom_value"

            def __eq__(self, other):
                return False  # Not equal to any enum member

            def __ne__(self, other):
                return True  # Not equal to any enum member

            def __hash__(self):
                return 0

        mock_resp = ValidationResponse(
            result=_CustomResult(),  # type: ignore
            policy_applied="test_policy",
            message="Custom result",
        )
        mock_security_manager.validate_tool_execution = AsyncMock(
            return_value=mock_resp
        )

        last_message = AIMessage(
            content="test",
            tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "call_1"}],
        )
        state = {"messages": [last_message], "todo_list": [], "sender": "master"}

        with patch.object(tools_node.mcp_client, "execute_tool") as mock_exec:
            result = await tools_node(state, config=None)
            mock_exec.assert_not_called()
            # Should return a validation error message (the else branch)
            assert "validation error" in result["messages"][0].content.lower()

    @pytest.mark.asyncio
    async def test_tool_error_results_in_error_status(
        self, tools_node, mock_security_manager
    ):
        """When MCP tool execution raises, metric status should be 'error'."""
        mock_security_manager.validate_tool_execution.return_value = ValidationResponse(
            result=ValidationResult.ALLOWED,
            policy_applied="low",
            message="Allowed",
        )
        # Make MCP client raise an exception so the tool result starts with "Error:"
        tools_node.mcp_client.execute_tool = AsyncMock(
            side_effect=RuntimeError("Connection failed")
        )
        last_message = AIMessage(
            content="search",
            tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "call_1"}],
        )
        state = {"messages": [last_message], "todo_list": [], "sender": "master"}
        result = await tools_node(state, config=None)
        assert len(result["messages"]) == 1
        assert "Error" in result["messages"][0].content
        assert result["messages"][0].name == "web_search"
        # Verify the error metric was recorded
        error_count = MetricsRegistry.tool_calls_total.labels(
            tool_name="web_search", server="mcp", status="error"
        )._value.get()
        assert error_count >= 1

    @pytest.mark.asyncio
    async def test_metrics_exception_does_not_crash(
        self, tools_node, mock_security_manager
    ):
        """When MetricsRegistry raises, the node should handle it gracefully."""
        mock_security_manager.validate_tool_execution.return_value = ValidationResponse(
            result=ValidationResult.ALLOWED,
            policy_applied="low",
            message="Allowed",
        )
        last_message = AIMessage(
            content="search",
            tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "call_1"}],
        )
        state = {"messages": [last_message], "todo_list": [], "sender": "master"}
        with patch.object(
            MetricsRegistry,
            "record_tool_call",
            side_effect=RuntimeError("Metrics failed"),
        ):
            result = await tools_node(state, config=None)
            assert len(result["messages"]) == 1

    @pytest.mark.asyncio
    async def test_parallel_execution_fallback_on_error(
        self, tools_node, mock_security_manager
    ):
        """When ParallelExecutor.run() fails, should fall back to sequential."""
        mock_security_manager.validate_tool_execution.return_value = ValidationResponse(
            result=ValidationResult.ALLOWED,
            policy_applied="low",
            message="Allowed",
        )

        last_message = AIMessage(
            content="search",
            tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "call_1"}],
        )
        state = {"messages": [last_message], "todo_list": [], "sender": "master"}

        # Patch ParallelExecutor.run to raise an exception
        with patch(
            "src.core.ark.nodes.tools.ParallelExecutor.run",
            new=AsyncMock(side_effect=RuntimeError("Executor crashed")),
        ):
            result = await tools_node(state, config=None)
            # Should still produce a result via sequential fallback
            assert len(result["messages"]) == 1
            # The tool should have been executed via sequential path
            assert result["messages"][0].name == "web_search"


class TestSecurityHelperMethods:
    """Unit tests for the security helper methods."""

    def test_infer_permissions_read_tool(self):
        """infer_permissions_from_tool should return READ for read_file."""
        perms = ARKSecurityManager.infer_permissions_from_tool("read_file")
        assert PermissionType.FILESYSTEM in perms
        assert PermissionType.READ in perms

    def test_infer_permissions_write_tool(self):
        """infer_permissions_from_tool should return WRITE for write_file."""
        perms = ARKSecurityManager.infer_permissions_from_tool("write_file")
        assert PermissionType.FILESYSTEM in perms
        assert PermissionType.WRITE in perms

    def test_infer_permissions_network_tool(self):
        """infer_permissions_from_tool should return NETWORK for web_search."""
        perms = ARKSecurityManager.infer_permissions_from_tool("web_search")
        assert PermissionType.NETWORK in perms
        assert PermissionType.EXTERNAL_API in perms

    def test_infer_permissions_unknown_tool_default(self):
        """Unknown tools should get READ + EXTERNAL_API default."""
        perms = ARKSecurityManager.infer_permissions_from_tool("unknown_tool")
        assert PermissionType.READ in perms
        assert PermissionType.EXTERNAL_API in perms

    def test_infer_permissions_file_args(self):
        """Arguments with file-like keys should add FILESYSTEM permission."""
        perms = ARKSecurityManager.infer_permissions_from_tool(
            "process_data", {"file_path": "/tmp/data.csv"}
        )
        assert PermissionType.FILESYSTEM in perms

    def test_infer_permissions_url_args(self):
        """Arguments with URL-like keys should add NETWORK permission."""
        perms = ARKSecurityManager.infer_permissions_from_tool(
            "fetch_data", {"url": "https://example.com"}
        )
        assert PermissionType.NETWORK in perms
        assert PermissionType.EXTERNAL_API in perms

    def test_infer_permissions_write_args(self):
        """Arguments with write-like keys should add WRITE permission."""
        perms = ARKSecurityManager.infer_permissions_from_tool(
            "save_data", {"content": "some data"}
        )
        assert PermissionType.WRITE in perms

    def test_extract_resources_file_path(self):
        """extract_resources_from_args should detect file paths."""
        resources = ARKSecurityManager.extract_resources_from_args(
            "read_file", {"file_path": "/etc/passwd"}
        )
        assert any("file:///etc/passwd" in r for r in resources)

    def test_extract_resources_url(self):
        """extract_resources_from_args should detect URLs."""
        resources = ARKSecurityManager.extract_resources_from_args(
            "web_fetch", {"url": "https://example.com"}
        )
        assert any("https://example.com" in r for r in resources)

    def test_build_validation_request(self):
        """build_validation_request should produce a valid request."""
        request = ARKSecurityManager.build_validation_request(
            user_id="user1",
            session_id="session1",
            execution_id="exec1",
            tool_name="read_file",
            arguments={"file_path": "/tmp/test.txt"},
        )

        assert request.context.user_id == "user1"
        assert request.context.tool_name == "read_file"
        assert PermissionType.FILESYSTEM in request.requested_permissions
        assert any("file:///tmp/test.txt" in r for r in request.target_resources)

    def test_policy_for_tool_low_risk(self):
        """Low-risk tools should map to 'low' policy."""
        policy = ToolsNode._policy_for_tool("web_search")
        assert policy == "low"

    def test_policy_for_tool_medium_risk(self):
        """Medium-risk tools should map to 'medium' policy."""
        policy = ToolsNode._policy_for_tool("read_file")
        assert policy == "medium"

    def test_policy_for_tool_high_risk(self):
        """High-risk tools should map to 'high' policy."""
        policy = ToolsNode._policy_for_tool("execute_command")
        assert policy == "high"

    def test_policy_for_tool_unknown(self):
        """Unknown tools should map to 'low' policy."""
        policy = ToolsNode._policy_for_tool("unknown")
        assert policy == "low"

    # ── Additional security manager path coverage ─────────────────

    def test_infer_permissions_exec_args(self):
        """Arguments with exec-like keys should add EXECUTE permission."""
        perms = ARKSecurityManager.infer_permissions_from_tool(
            "run_script", {"command": "ls -la"}
        )
        assert PermissionType.EXECUTE in perms

    def test_infer_permissions_shell_args(self):
        """Arguments with shell/bash keys should add SYSTEM permission."""
        perms = ARKSecurityManager.infer_permissions_from_tool(
            "run_script", {"shell": "/bin/bash", "command": "ls"}
        )
        assert PermissionType.SYSTEM in perms
        assert PermissionType.EXECUTE in perms

    def test_extract_resources_url_in_url_key(self):
        """URL keys with http value should produce https:// resource."""
        resources = ARKSecurityManager.extract_resources_from_args(
            "fetch", {"endpoint": "https://api.example.com/data"}
        )
        assert any("https://api.example.com/data" in r for r in resources)

    def test_extract_resources_url_key_with_domain(self):
        """URL keys with domain value should produce https:// resource."""
        resources = ARKSecurityManager.extract_resources_from_args(
            "fetch", {"url": "example.com/api"}
        )
        assert any("https://example.com/api" in r for r in resources)

    def test_extract_resources_embedded_url(self):
        """Arbitrary string value starting with http:// should be detected as URL."""
        # Use a key not in file_path_keys or url_keys so we reach the URL-scanning branch
        resources = ARKSecurityManager.extract_resources_from_args(
            "process",
            {"description": "https://example.com/data"},
        )
        assert any("https://example.com/data" in r for r in resources)

    def test_extract_resources_absolute_file_path_value(self):
        """Arbitrary string value starting with / should be detected as file path."""
        resources = ARKSecurityManager.extract_resources_from_args(
            "process",
            {"note": "/etc/config.yaml"},
        )
        # "note" is not in file_path_keys, so the value-starting-with-/ branch applies
        assert any("file:///etc/config.yaml" in r for r in resources)

    def test_extract_resources_relative_path(self):
        """A value starting with ./ should be detected as a file path."""
        resources = ARKSecurityManager.extract_resources_from_args(
            "read", {"path": "./local/file.txt"}
        )
        assert any("file://./local/file.txt" in r for r in resources)

    def test_extract_resources_absolute_path(self):
        """Arbitrary string value starting with / should be detected as file."""
        resources = ARKSecurityManager.extract_resources_from_args(
            "read", {"target": "/var/log/system.log"}
        )
        assert any("file:///var/log/system.log" in r for r in resources)

    def test_extract_resources_non_string_arg(self):
        """Non-string argument values should be skipped without error."""
        resources = ARKSecurityManager.extract_resources_from_args(
            "analyze", {"count": 42, "items": [1, 2, 3]}
        )
        assert resources == []

    def test_extract_resources_key_ending_with_path(self):
        """Keys ending with _path should be treated as file paths."""
        resources = ARKSecurityManager.extract_resources_from_args(
            "custom", {"output_path": "./results.txt"}
        )
        assert any("file://./results.txt" in r for r in resources)

    def test_extract_resources_file_path_with_dot(self):
        """File path key with value containing a dot but no leading slash should match regex."""
        resources = ARKSecurityManager.extract_resources_from_args(
            "read", {"filename": "data.file.csv"}
        )
        # "filename" is in file_path_keys, value doesn't start with / or ./
        # but contains "." -> should match the regex branch
        assert any("file://data.file.csv" in r for r in resources)
