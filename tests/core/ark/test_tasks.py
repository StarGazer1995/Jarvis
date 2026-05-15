"""
Task 模块单元测试

测试 Task 数据类、tasks_from_dicts、tasks_to_dicts、execute_manage_tasks 等函数。
"""


from src.core.ark.tasks import (
    Task,
    TaskStatus,
    execute_manage_tasks,
    tasks_from_dicts,
    tasks_to_dicts,
)


class TestTaskDataClass:
    """测试 Task 数据类"""

    def test_task_default_status(self):
        task = Task(id="123", description="Test")
        assert task.description == "Test"
        assert task.status == TaskStatus.PENDING
        assert task.id == "123"

    def test_task_with_id(self):
        task = Task(description="Test", id="custom-123")
        assert task.id == "custom-123"

    def test_task_with_status(self):
        task = Task(id="456", description="Test", status=TaskStatus.COMPLETED)
        assert task.status == TaskStatus.COMPLETED

    def test_task_to_dict(self):
        task = Task(id="t1", description="Hello", status=TaskStatus.IN_PROGRESS)
        d = task.to_dict()
        assert d["id"] == "t1"
        assert d["description"] == "Hello"
        assert d["status"] == "in_progress"

    def test_task_from_dict_full(self):
        d = {"id": "t1", "description": "World", "status": "pending"}
        task = Task.from_dict(d)
        assert task.id == "t1"
        assert task.description == "World"
        assert task.status == TaskStatus.PENDING

    def test_task_from_dict_missing_status(self):
        d = {"description": "No status"}
        task = Task.from_dict(d)
        assert task.status == TaskStatus.PENDING

    def test_task_from_dict_invalid_status(self):
        d = {"description": "Bad", "status": "unknown_status"}
        task = Task.from_dict(d)
        assert task.status == TaskStatus.PENDING

    def test_task_repr(self):
        task = Task(id="t1", description="Test", status=TaskStatus.PENDING)
        r = repr(task)
        assert "t1" in r
        assert "Test" in r
        assert "pending" in r


class TestTaskConversions:
    """测试 tasks_from_dicts / tasks_to_dicts 转换"""

    def test_tasks_to_dicts(self):
        tasks = [
            Task(id="t1", description="A", status=TaskStatus.PENDING),
            Task(id="t2", description="B", status=TaskStatus.COMPLETED),
        ]
        dicts = tasks_to_dicts(tasks)
        assert len(dicts) == 2
        assert dicts[0]["id"] == "t1"
        assert dicts[0]["status"] == "pending"
        assert dicts[1]["status"] == "completed"

    def test_tasks_from_dicts(self):
        dicts = [
            {"id": "t1", "description": "A", "status": "pending"},
            {"id": "t2", "description": "B", "status": "completed"},
        ]
        tasks = tasks_from_dicts(dicts)
        assert len(tasks) == 2
        assert tasks[0].id == "t1"
        assert tasks[1].status == TaskStatus.COMPLETED

    def test_roundtrip(self):
        original = [Task(id="t1", description="X"), Task(id="t2", description="Y")]
        dicts = tasks_to_dicts(original)
        restored = tasks_from_dicts(dicts)
        assert len(restored) == 2
        assert restored[0].description == "X"


class TestExecuteManageTasks:
    """测试 execute_manage_tasks 的增删改查操作"""

    def test_add_task(self):
        todo = []
        result = execute_manage_tasks(
            {"action": "add", "description": "Buy milk"}, todo
        )
        assert "added" in result
        assert len(todo) == 1
        assert todo[0]["description"] == "Buy milk"
        assert todo[0]["status"] == "pending"

    def test_update_task_status(self):
        todo = [{"id": "t1", "description": "Task 1", "status": "pending"}]
        result = execute_manage_tasks(
            {"action": "update", "id": "t1", "status": "in_progress"}, todo
        )
        assert "updated" in result
        assert todo[0]["status"] == "in_progress"

    def test_complete_task(self):
        todo = [{"id": "t1", "description": "Task 1", "status": "pending"}]
        result = execute_manage_tasks({"action": "complete", "id": "t1"}, todo)
        assert "completed" in result
        assert todo[0]["status"] == "completed"

    def test_update_task_not_found(self):
        todo = [{"id": "t1", "description": "Task 1", "status": "pending"}]
        result = execute_manage_tasks({"action": "update", "id": "nonexistent"}, todo)
        assert "not found" in result

    def test_unknown_action(self):
        result = execute_manage_tasks({"action": "unknown"}, [])
        assert "Unknown" in result or "unknown" in result
