"""
Parallel Execution Engine

Resolves dependencies between tool calls and executes them in optimal order,
running independent tools concurrently via asyncio.gather.

Supports:
- Dependency declaration via ``depends_on`` field
- Automatic topological sort of tool calls
- Parallel execution of independent tools
- Sequential execution of dependent tools
- Variable interpolation from previous outputs (``$ref{tool_name}``)
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Pattern for referencing outputs from previous tools: $ref{tool_name} or $ref{tool_name.field}
REF_PATTERN = re.compile(r"\$ref\{([^}]+)\}")


@dataclass
class ToolCall:
    """A single tool call request with optional dependency info."""

    name: str
    arguments: Dict[str, Any]
    id: str = ""
    depends_on: List[str] = field(default_factory=list)


@dataclass
class ExecutionNode:
    """Internal node representing a stage in the execution DAG."""

    tool_call: ToolCall
    deps_remaining: int = 0  # Count of unmet dependencies
    result: Any = None
    error: Optional[str] = None


class ExecutionGraph:
    """
    Builds and validates a dependency graph from a list of ToolCalls.

    Supports a simple dependency model: each tool can declare which other
    tool(s) it depends on via the ``depends_on`` field.
    """

    def __init__(self, tool_calls: List[ToolCall]):
        self.tool_calls = tool_calls
        self.nodes: Dict[str, ExecutionNode] = {}
        self._build()

    def _build(self):
        """Build the execution graph and validate dependencies."""
        name_counts: Dict[str, int] = {}

        # Create nodes, assign IDs if missing
        for tc in self.tool_calls:
            tid = tc.id or tc.name
            if tid in name_counts:
                # Disambiguate duplicate tool names
                name_counts[tid] += 1
                tid = f"{tid}_{name_counts[tid]}"
            else:
                name_counts[tid] = 1

            self.nodes[tid] = ExecutionNode(tool_call=tc)
            tc.id = tid

        # Validate dependencies
        for tid, node in self.nodes.items():
            for dep in node.tool_call.depends_on:
                if dep not in self.nodes:
                    raise ValueError(
                        f"Tool '{tid}' depends on '{dep}', "
                        f"but '{dep}' is not in the execution list. "
                        f"Available tools: {list(self.nodes.keys())}"
                    )

        # Count dependencies
        for tid, node in self.nodes.items():
            node.deps_remaining = len(node.tool_call.depends_on)

    def get_ready_tools(self, completed: set) -> List[str]:
        """Get tool IDs whose dependencies are all satisfied."""
        ready = []
        for tid, node in self.nodes.items():
            if tid in completed:
                continue
            if all(dep in completed for dep in node.tool_call.depends_on):
                ready.append(tid)
        return ready

    def are_all_completed(self, completed: set) -> bool:
        """Check if all tools have been completed."""
        return len(completed) >= len(self.nodes)


class ExecutionPlan:
    """
    Resolves the execution plan: which tools run in which order,
    grouped into parallel batches.
    """

    def __init__(self, tool_calls: List[ToolCall]):
        self.graph = ExecutionGraph(tool_calls)

    def resolve(self) -> List[List[str]]:
        """
        Resolve the execution plan into batches.

        Returns:
            A list of batches, where each batch is a list of tool IDs
            that can be executed in parallel.
        """
        completed: set = set()
        batches: List[List[str]] = []

        # Guard: empty input
        if not self.graph.tool_calls:
            return batches

        while not self.graph.are_all_completed(completed):
            ready = self.graph.get_ready_tools(completed)
            if not ready:
                # This shouldn't happen if dependencies are valid
                remaining = set(self.graph.nodes.keys()) - completed
                raise RuntimeError(
                    f"Deadlock detected: tools {remaining} have unmet "
                    f"dependencies but none are ready."
                )
            batches.append(ready)
            completed.update(ready)

        return batches


class ParallelExecutor:
    """
    Executes tool calls in parallel batches according to their dependencies.

    Usage::

        executor = ParallelExecutor(execute_fn=my_executor)
        results = await executor.run([
            ToolCall(name="web_search", arguments={"query": "AI", "depends_on": []}),
            ToolCall(name="get_time", arguments={}),
        ])
    """

    def __init__(
        self,
        execute_fn: Callable[[str, Dict[str, Any]], Any],
    ):
        """
        Args:
            execute_fn: Async callable ``(tool_name, arguments) -> result``.
                This is typically ``ToolsNode._execute_tool_call()``.
        """
        self.execute_fn = execute_fn

    async def run(
        self,
        tool_calls: List[ToolCall],
        shared_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute tool calls respecting dependencies.

        Args:
            tool_calls: List of tool calls to execute.
            shared_context: Optional shared context passed to each tool call.

        Returns:
            Dict mapping tool IDs to their results.
        """
        if not tool_calls:
            return {}

        plan = ExecutionPlan(tool_calls)
        batches = plan.resolve()
        results: Dict[str, Any] = {}
        errors: Dict[str, str] = {}

        for batch_idx, batch in enumerate(batches):
            logger.info(
                f"Executing batch {batch_idx + 1}/{len(batches)} "
                f"with {len(batch)} tool(s): {batch}"
            )

            async def _execute_one(tid: str) -> tuple:
                node = plan.graph.nodes[tid]
                tc = node.tool_call

                # Resolve argument references from previous results
                resolved_args = self._resolve_refs(tc.arguments, results)

                try:
                    result = await self.execute_fn(tc.name, resolved_args)
                    results[tid] = result
                    node.result = result
                    return tid, result, None
                except Exception as e:
                    error_msg = f"Error executing tool '{tc.name}': {e}"
                    logger.error(error_msg)
                    errors[tid] = error_msg
                    node.error = error_msg
                    return tid, None, error_msg

            # Execute all tools in this batch concurrently
            batch_tasks = [_execute_one(tid) for tid in batch]
            await asyncio.gather(*batch_tasks)

        return results

    def _resolve_refs(
        self,
        args: Dict[str, Any],
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Replace $ref{tool_name} references with actual results."""
        resolved = {}
        for key, value in args.items():
            if isinstance(value, str):
                resolved[key] = REF_PATTERN.sub(
                    lambda m: self._lookup_ref(m.group(1), results),
                    value,
                )
            elif isinstance(value, dict):
                resolved[key] = self._resolve_refs(value, results)
            elif isinstance(value, list):
                resolved[key] = [
                    self._resolve_refs(v, results) if isinstance(v, dict) else v
                    for v in value
                ]
            else:
                resolved[key] = value
        return resolved

    @staticmethod
    def _lookup_ref(ref: str, results: Dict[str, Any]) -> str:
        """Look up a reference like ``tool_name`` or ``tool_name.field``."""
        if "." in ref:
            tool_id, field = ref.split(".", 1)
        else:
            tool_id, field = ref, None

        if tool_id not in results:
            logger.warning(f"Reference $ref{{{ref}}} not found in results")
            return f"{{{{unresolved:{ref}}}}}"

        result = results.get(tool_id)
        if field is not None:
            if isinstance(result, dict):
                return str(result.get(field, str(result)))
            return str(result)
        return str(result)
