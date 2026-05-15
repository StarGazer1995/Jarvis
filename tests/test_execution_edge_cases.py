"""
Execution Engine 补充测试

覆盖 execution/engine.py 中未测试的边界情况。
"""


from src.core.execution.engine import ExecutionGraph, ExecutionPlan, ToolCall


class TestExecutionGraphEdgeCases:
    def test_tool_call_with_depends_on(self):
        tc = ToolCall(name="tool_b", input={}, depends_on=["tool_a"])
        assert tc.depends_on == ["tool_a"]

    def test_tool_call_no_depends(self):
        tc = ToolCall(name="tool_a", input={})
        assert tc.depends_on == []

    def test_graph_get_ready_tools_after_completion(self):
        calls = [
            ToolCall(name="a", input={}),
            ToolCall(name="b", input={}, depends_on=["a"]),
        ]
        graph = ExecutionGraph(calls)
        # Complete 'a'
        graph.completed.add("a")
        ready = graph.get_ready_tools()
        assert len(ready) == 1
        assert ready[0].name == "b"

    def test_graph_all_completed(self):
        calls = [ToolCall(name="a", input={})]
        graph = ExecutionGraph(calls)
        assert not graph.all_completed()
        graph.completed.add("a")
        assert graph.all_completed()


class TestExecutionPlanEdgeCases:
    def test_plan_with_diamond_dependency(self):
        """A → B, A → C, B → D, C → D"""
        calls = [
            ToolCall(name="A", input={}),
            ToolCall(name="B", input={}, depends_on=["A"]),
            ToolCall(name="C", input={}, depends_on=["A"]),
            ToolCall(name="D", input={}, depends_on=["B", "C"]),
        ]
        plan = ExecutionPlan.resolve(calls)
        assert len(plan.batches) == 3  # [A], [B, C], [D]

    def test_plan_empty(self):
        plan = ExecutionPlan.resolve([])
        assert len(plan.batches) == 0
