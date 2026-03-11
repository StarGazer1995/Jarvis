from dataclasses import dataclass
from enum import Enum
from typing import Any


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        status_raw = data.get("status", TaskStatus.PENDING.value)
        try:
            status = TaskStatus(status_raw)
        except ValueError:
            status = TaskStatus.PENDING
        return cls(
            id=str(data.get("id", "")),
            description=str(data.get("description", "")),
            status=status,
            result=data.get("result"),
        )


def tasks_to_dicts(tasks: list[Task]) -> list[dict[str, Any]]:
    return [task.to_dict() for task in tasks]


def tasks_from_dicts(items: list[dict[str, Any]]) -> list[Task]:
    return [Task.from_dict(item) for item in items]


def parse_task_args(params: Any) -> dict[str, Any] | None:
    if isinstance(params, dict):
        return params
    if isinstance(params, str):
        import json

        try:
            parsed = json.loads(params)
        except Exception:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def execute_manage_tasks(
    params: Any,
    todo_list: list[dict[str, Any]],
    invalid_params_message: str = "Error: Invalid params format",
) -> str:
    parsed = parse_task_args(params)
    if parsed is None:
        return invalid_params_message
    return apply_manage_tasks(parsed, todo_list)


def apply_manage_tasks(args: dict[str, Any], todo_list: list[dict[str, Any]]) -> str:
    action = args.get("action")

    if action == "add":
        description = args.get("description")
        if not description:
            return "Error: Description required for adding task."
        task_id = str(len(todo_list) + 1)
        todo_list.append(
            {
                "id": task_id,
                "description": str(description),
                "status": TaskStatus.PENDING.value,
                "result": None,
            }
        )
        return f"Task added: [{task_id}] {description}"

    if action == "update":
        task_id = str(args.get("id") or args.get("task_id"))
        status = args.get("status")
        result = args.get("result")
        task = next((t for t in todo_list if t.get("id") == task_id), None)
        if not task:
            return f"Error: Task {task_id} not found."
        if status:
            task["status"] = str(status)
        if result is not None:
            task["result"] = str(result)
        return f"Task {task_id} updated."

    if action == "complete":
        task_id = str(args.get("id") or args.get("task_id"))
        result = args.get("result")
        task = next((t for t in todo_list if t.get("id") == task_id), None)
        if not task:
            return f"Error: Task {task_id} not found."
        task["status"] = TaskStatus.COMPLETED.value
        if result is not None:
            task["result"] = str(result)
        return f"Task {task_id} completed."

    return f"Error: Unknown action {action}."
