"""Additional task tests migrated from legacy `src/test` files."""


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


class TestTasksFinal:
    """tasks.py 最后覆盖"""

    def test_task_status_enum(self):
        from src.core.ark.tasks import TaskStatus

        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"

    def test_execute_manage_add_no_description(self):
        from src.core.ark.tasks import execute_manage_tasks

        result = execute_manage_tasks({"action": "add"}, [])
        assert "missing" in result.lower() or "description" in result.lower()
