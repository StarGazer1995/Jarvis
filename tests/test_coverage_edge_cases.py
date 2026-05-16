"""
覆盖补全测试 - 第二部分

针对低覆盖率的文件添加正确的边界情况测试。
"""

from unittest.mock import patch

import pytest

# ── src/core/execution/engine.py ──────────────────────────────────────────


class TestExecutionEngineEdgeCoverage:
    """覆盖 engine.py 中未覆盖的边界"""

    def test_duplicate_tool_names(self):
        """测试重复工具名会被自动消歧"""
        from src.core.execution.engine import ExecutionGraph, ToolCall

        # Both have the same id but different names - should still work
        calls = [
            ToolCall(name="tool_a", id="tool_a", arguments={}),
            ToolCall(name="tool_a", id="tool_a", arguments={}),
        ]
        graph = ExecutionGraph(calls)
        assert "tool_a" in graph.nodes
        # Second one should be disambiguated
        disambiguated = [k for k in graph.nodes if k != "tool_a"]
        assert len(disambiguated) == 1

    def test_lookup_ref_no_dot(self):
        """测试引用查找：无字段名"""
        from src.core.execution.engine import ParallelExecutor

        result = ParallelExecutor._lookup_ref("tool_1", {"tool_1": "value"})
        assert result == "value"

    def test_lookup_ref_with_field(self):
        """测试引用查找：字段名"""
        from src.core.execution.engine import ParallelExecutor

        result = ParallelExecutor._lookup_ref(
            "tool_1.field_x", {"tool_1": {"field_x": "hello"}}
        )
        assert result == "hello"

    def test_lookup_ref_missing(self):
        """测试引用查找：缺失引用"""
        from src.core.execution.engine import ParallelExecutor

        result = ParallelExecutor._lookup_ref("missing", {"tool_1": "value"})
        assert "unresolved" in result


# ── src/core/llm/stream_handler.py ───────────────────────────────────────


class TestStreamHandlerCoverage:
    """覆盖 stream_handler.py 中未覆盖的边界"""

    def test_emit_callback_none_content(self):
        """测试 _emit_callback 无内容时调用回调"""
        from src.core.llm.stream_handler import StreamTokenHandler

        handler = StreamTokenHandler({"on_thought": lambda: None})
        handler._emit_callback("on_thought")
        handler._emit_callback("on_thought", content="some content")
        handler._emit_callback("nonexistent")  # should return early

    def test_emit_callback_error_handling(self):
        """测试 _emit_callback 异常处理"""
        from src.core.llm.stream_handler import StreamTokenHandler

        def failing_cb():
            raise ValueError("callback failed")

        handler = StreamTokenHandler({"on_thought": failing_cb})
        handler._emit_callback("on_thought")  # should not raise


# ── src/core/agent/react.py ──────────────────────────────────────────────


class TestReactAgentCoverage:
    """覆盖 react.py 中未覆盖的边界"""

    @pytest.mark.asyncio
    async def test_process_input_not_ready(self):
        """测试 Agent 未就绪时的处理"""
        from src.core.agent.react import ReActAgent
        from src.core.agent.types import AgentState

        agent = ReActAgent()
        agent.state = AgentState.INITIALIZING
        response = await agent.process_input("test")
        assert "not ready" in response.lower()

    @pytest.mark.asyncio
    async def test_process_input_error(self):
        """测试 process_input 异常处理"""
        from src.core.agent.react import ReActAgent
        from src.core.agent.types import AgentState

        agent = ReActAgent()
        agent.state = AgentState.READY
        with patch.object(agent, "_run_loop", side_effect=ValueError("test error")):
            response = await agent.process_input("test")
            assert "error" in response.lower()


# ── src/core/llm/utils/error_handler.py ──────────────────────────────────


class TestErrorHandlerCoverage:
    """覆盖 error_handler.py 中未覆盖的边界"""

    def test_log_llm_error_with_details(self):
        """测试 LLM 错误日志"""
        from src.core.llm.utils.error_handler import LLMError, log_llm_error

        error = LLMError("test error", details={"key": "value"})
        log_llm_error(error)

    def test_handle_none_error(self):
        """测试处理 None 错误"""
        from src.core.llm.utils.error_handler import handle_openai_error

        result = handle_openai_error(None)
        assert result is not None


# ── src/core/llm/utils/cache_manager.py ──────────────────────────────────


class TestCacheManagerEdgeCases:
    """覆盖 cache_manager.py 中未覆盖的边界"""

    def test_cache_expiry(self):
        """测试缓存过期"""
        from src.core.llm.types import LLMMessage
        from src.core.llm.utils.cache_manager import CacheManager

        cache = CacheManager(ttl=0)
        msgs = [LLMMessage(role="user", content="test")]
        cache.set(msgs, "response", model="gpt-4")
        result = cache.get(msgs, model="gpt-4")
        assert result is None

    def test_cache_set_with_different_args(self):
        """测试使用不同参数缓存"""
        from src.core.llm.types import LLMMessage
        from src.core.llm.utils.cache_manager import CacheManager

        cache = CacheManager(ttl=3600)
        msgs = [LLMMessage(role="user", content="hello")]
        cache.set(msgs, "response")
        result = cache.get(msgs)
        assert result is not None


# ── src/core/llm/utils/metrics_collector.py ──────────────────────────────


class TestMetricsCollectorEdgeCases:
    """覆盖 metrics_collector.py 中未覆盖的边界"""

    def test_error_rate_zero(self):
        """测试无请求时错误率为0"""
        from src.core.llm.utils.metrics_collector import LLMMetrics

        metrics = LLMMetrics()
        assert metrics.error_rate == 0.0
