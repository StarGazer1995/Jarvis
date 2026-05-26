"""
Deep Research Agent Implementation
"""

import logging
from copy import deepcopy
from datetime import datetime
from typing import Any

from src.capabilities.deep_research_tools import DeepResearchTools
from src.core.agent.react import ReActAgent
from src.core.llm.parsers import JSONOutputParser


class DeepResearchAgent(ReActAgent):
    """
    Agent implementing the Deep Research paradigm with ReAct loop and JSON-based tool calling.
    Inherits from ReActAgent to reuse JSON parsing and execution loop logic.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.logger = logging.getLogger("agent.deep_research")
        # Deep research typically requires more steps
        self.max_steps = self.config.get("max_steps", 30)
        self.parser = JSONOutputParser(protocol="deep_research")
        self.tools = DeepResearchTools(self.llm_manager)
        self.raw_response_history: list[str] = []
        self.parsed_response_history: list[dict[str, Any]] = []
        self.last_raw_response: str | None = None
        self.last_parsed_response: dict[str, Any] | None = None

    def _get_system_prompt(self) -> list[Any]:
        """Override system prompt with Deep Research specific prompt."""
        current_date = datetime.now().strftime("%Y-%m-%d")
        messages = self.prompt_manager.render_template(
            "deep_research_system", current_date=current_date
        )
        return messages

    async def execute_tool(self, name: str, params: Any) -> Any:
        """
        Execute Deep Research tools.

        Args:
            name: Tool name
            params: Tool parameters (dict)
        """
        try:
            if name == "search":
                return await self.tools.search(params.get("query", []))
            elif name == "visit":
                return await self.tools.visit(
                    params.get("url", []), params.get("goal", "")
                )
            elif name == "google_scholar":
                return await self.tools.google_scholar(params.get("query", []))
            elif name == "parse_file":
                return await self.tools.parse_file(params.get("files", []))
            elif name == "PythonInterpreter":
                # Code is provided in arguments
                if "code" in params:
                    return await self.tools.python_interpreter(params["code"])
                return "Error: PythonInterpreter must provide 'code' argument."
            else:
                return f"Error: Unknown tool '{name}'"

        except Exception as e:
            return f"Error executing tool '{name}': {str(e)}"

    def _record_protocol_response(
        self, raw_response: str, parsed_response: dict[str, Any]
    ) -> None:
        """
        Persist Deep Research protocol responses for debugging and future replay.

        Args:
            raw_response: Raw response string returned by the model.
            parsed_response: Normalized parsed response object.
        """
        self.last_raw_response = raw_response
        self.last_parsed_response = deepcopy(parsed_response)
        self.raw_response_history.append(raw_response)
        self.parsed_response_history.append(deepcopy(parsed_response))
