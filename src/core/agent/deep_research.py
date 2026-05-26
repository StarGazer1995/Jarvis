"""
Deep Research Agent Implementation
"""

import logging
import re
from copy import deepcopy
from datetime import datetime
from typing import Any

from src.capabilities.deep_research_tools import DeepResearchTools
from src.capabilities.refinement import DeepResearchAuditor
from src.core.agent.react import ReActAgent
from src.core.agent.types import AgentStep
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
        self.auditor = DeepResearchAuditor(llm_manager=self.llm_manager)
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

    async def _validate_final_answer(
        self, content: Any, rendered_answer: str, steps: list[AgentStep]
    ) -> str | None:
        """
        Enforce a minimal evidence gate for Deep Research final answers.

        Args:
            content: Parsed final answer content from the protocol response.
            rendered_answer: User-facing final answer text.
            steps: Completed agent steps collected during the loop.

        Returns:
            Optional validation error text. Returns None when the answer satisfies
            the Deep Research evidence requirements.
        """
        normalized_answer = rendered_answer.strip()
        lowered_answer = normalized_answer.lower()
        structured_insufficient_evidence = (
            bool(content.get("insufficient_evidence"))
            if isinstance(content, dict)
            else False
        )
        declares_insufficient_evidence = structured_insufficient_evidence or any(
            phrase in lowered_answer
            for phrase in (
                "insufficient evidence",
                "evidence is insufficient",
                "insufficient information",
            )
        )

        if not steps and not declares_insufficient_evidence:
            return (
                "Deep Research must gather tool-backed evidence before returning "
                "a final answer."
            )

        if declares_insufficient_evidence:
            return None

        has_sources_section = "sources:" in lowered_answer
        has_url = re.search(r"https?://\S+", normalized_answer) is not None
        if not has_sources_section or not has_url:
            return (
                "Deep Research final answers with factual claims must include a "
                "'Sources:' section with at least one supporting URL."
            )

        # Keep the auditor aligned with the agent's active LLM manager, especially in tests.
        self.auditor.llm_manager = self.llm_manager
        audit_verdict = await self.auditor.review_with_observations(
            normalized_answer, steps=steps
        )
        if audit_verdict != "PASS":
            return audit_verdict.removeprefix("RETRY: ").strip()

        return None

    def _render_final_answer(self, content: Any) -> str:
        """
        Render structured Deep Research answers into readable text.

        Args:
            content: Parsed answer content from the protocol response.

        Returns:
            User-facing answer text.
        """
        if isinstance(content, str):
            return content

        if not isinstance(content, dict):
            return str(content)

        summary = str(content.get("summary", "")).strip()
        claims = content.get("claims", [])
        sources = content.get("sources", [])
        insufficient_evidence = bool(content.get("insufficient_evidence", False))

        sections: list[str] = []
        if summary:
            sections.append(summary)

        if claims:
            claim_lines = [
                (
                    f"{index}. {claim.get('statement', '').strip()} "
                    f"[Sources: {', '.join(claim.get('source_urls', []))}]"
                ).rstrip()
                for index, claim in enumerate(claims, start=1)
            ]
            sections.append("Claims:\n" + "\n".join(claim_lines))

        if sources:
            source_lines = [
                (
                    f"- {source.get('title', '').strip()}: {source.get('url', '').strip()}\n"
                    f"  Evidence: {source.get('evidence', '').strip()}"
                )
                for source in sources
            ]
            sections.append("Sources:\n" + "\n".join(source_lines))

        if insufficient_evidence and "insufficient evidence" not in summary.lower():
            sections.append("Evidence status: insufficient evidence")

        return "\n\n".join(section for section in sections if section).strip()

    def _render_tool_observation(
        self, tool_name: str | None, tool_args: Any, result: Any
    ) -> str:
        """
        Render structured Deep Research tool results into observation text.

        Args:
            tool_name: Tool name associated with the result.
            tool_args: Tool arguments used for execution.
            result: Raw tool result object.

        Returns:
            Observation string for the main ReAct loop.
        """
        if not isinstance(result, dict):
            return str(result)

        tool = result.get("tool")
        entries = result.get("results", [])
        if tool in {"search", "google_scholar"} and isinstance(entries, list):
            sections: list[str] = []
            grouped_entries: dict[str, list[dict[str, str]]] = {}
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                grouped_entries.setdefault(entry.get("query", ""), []).append(entry)

            for query, query_entries in grouped_entries.items():
                query_lines = [f"Query: {query}"] if query else []
                for entry in query_entries:
                    title = entry.get("title", "")
                    url = entry.get("url", "")
                    snippet = entry.get("snippet", "")
                    if title:
                        query_lines.append(f"Title: {title}")
                    if url:
                        query_lines.append(f"URL: {url}")
                    if snippet:
                        query_lines.append(f"Snippet: {snippet}")
                sections.append("\n".join(query_lines).strip())
            return "\n=======\n".join(section for section in sections if section)

        if tool == "visit" and isinstance(entries, list):
            rendered_entries: list[str] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                url = entry.get("url", "")
                rational = entry.get("rational", "")
                evidence = entry.get("evidence", "")
                summary = entry.get("summary", "")
                error = entry.get("error", "")
                lines = [f"URL: {url}"] if url else []
                if error:
                    lines.append(f"Error: {error}")
                else:
                    if rational:
                        lines.append(f"Rationale: {rational}")
                    if evidence:
                        lines.append(f"Evidence: {evidence}")
                    if summary:
                        lines.append(f"Summary: {summary}")
                rendered_entries.append("\n".join(lines).strip())
            return "\n=======\n".join(
                section for section in rendered_entries if section
            )

        return str(result)
