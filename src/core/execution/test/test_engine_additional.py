"""Additional execution engine tests migrated from `src/test`."""


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
