"""
LLM Output Parsers

This module provides parsers for LLM responses, specifically handling JSON output
and ensuring strict schema validation.
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ValidationError

try:
    import json_repair
except ImportError:
    json_repair = None


class BaseOutputParser(ABC):
    """Base class for output parsers."""

    @abstractmethod
    def parse(self, text: str) -> Any:
        """Parse the output text."""
        pass


class JSONOutputParser(BaseOutputParser):
    """
    Parses JSON output from LLM responses.

    Features:
    - Handles Markdown code blocks (```json ... ```)
    - strict schema validation (optional)
    - robust error handling using json_repair (if installed)
    """

    def __init__(
        self,
        pydantic_model: type[BaseModel] | None = None,
        allow_repair: bool = False,
        protocol: str | None = None,
    ):
        """
        Initialize the parser.

        Args:
            pydantic_model: Optional Pydantic model to validate against.
            allow_repair: Whether to attempt JSON repair when parsing fails.
            protocol: Optional protocol name for additional structural validation.
        """
        self.pydantic_model = pydantic_model
        self.allow_repair = allow_repair
        self.protocol = protocol
        self.logger = logging.getLogger("llm.parsers.json")

    def parse(self, text: str) -> dict[str, Any] | BaseModel:
        """
        Parse the text into a dictionary or Pydantic model.

        Args:
            text: The text to parse.

        Returns:
            Parsed dictionary or Pydantic model instance.

        Raises:
            ValueError: If parsing fails.
        """
        cleaned_text = self._clean_json_text(text)

        try:
            # First attempt: standard json.loads
            data = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            # Second attempt: use json_repair if available
            if json_repair and self.allow_repair:
                try:
                    self.logger.warning(
                        f"Standard JSON parse failed, attempting repair: {e}"
                    )
                    decoded_object = json_repair.repair_json(
                        cleaned_text, return_objects=True
                    )
                    if isinstance(decoded_object, (dict, list)):
                        data = decoded_object
                    else:
                        raise ValueError(
                            f"Repair returned non-JSON object: {type(decoded_object)}"
                        )
                except Exception as repair_error:
                    self.logger.error(
                        f"JSON repair failed: {repair_error}\nText: {text}"
                    )
                    raise ValueError(
                        f"Invalid JSON output (repair failed): {repair_error}"
                    )
            else:
                self.logger.error(
                    f"Failed to parse JSON and json_repair not installed: {e}\nText: {text}"
                )
                raise ValueError(f"Invalid JSON output: {e}")

        if self.protocol:
            if not isinstance(data, dict):
                raise ValueError(f"Expected JSON object, got {type(data)}")
            self._validate_protocol(data)

        if self.pydantic_model:
            try:
                # If data is a list (e.g. from json_repair), validation might fail if model expects dict
                if not isinstance(data, dict):
                    raise ValueError(f"Expected JSON object, got {type(data)}")
                return self.pydantic_model.model_validate(data)
            except ValidationError as e:
                self.logger.error(f"Schema validation failed: {e}\nData: {data}")
                raise ValueError(f"Schema validation failed: {e}")

        return data

    def _validate_protocol(self, data: dict[str, Any]) -> None:
        """
        Validate a parsed JSON object against a known protocol contract.

        Args:
            data: Parsed JSON object.

        Raises:
            ValueError: If the object does not satisfy the selected protocol.
        """
        if self.protocol == "deep_research":
            self._validate_deep_research_response(data)
            return
        if self.protocol == "deep_research_audit":
            self._validate_deep_research_audit_response(data)
            return

        raise ValueError(f"Unsupported parser protocol: {self.protocol}")

    def _validate_deep_research_response(self, data: dict[str, Any]) -> None:
        """
        Validate the formal Deep Research response protocol.

        Args:
            data: Parsed response object.

        Raises:
            ValueError: If the response violates the Deep Research protocol.
        """
        thought = data.get("thought")
        if not isinstance(thought, str):
            raise ValueError("Deep Research protocol requires 'thought' to be a string")

        msg_type = data.get("type")
        allowed_types = {"answer", "tool_call", "tool_calls", "error"}
        if msg_type not in allowed_types:
            raise ValueError(
                f"Deep Research protocol requires 'type' to be one of {sorted(allowed_types)}"
            )

        if "content" not in data:
            raise ValueError("Deep Research protocol requires a 'content' field")

        content = data["content"]
        if msg_type == "answer":
            self._validate_deep_research_answer_content(content)
            return

        if msg_type == "tool_call":
            self._validate_tool_call(content)
            return

        if msg_type == "tool_calls":
            if not isinstance(content, list):
                raise ValueError(
                    "Deep Research protocol requires 'tool_calls' content to be an array"
                )
            for tool_call in content:
                self._validate_tool_call(tool_call)
            return

        if msg_type == "error":
            if not isinstance(content, dict):
                raise ValueError(
                    "Deep Research protocol requires 'error' content to be an object"
                )
            code = content.get("code")
            message = content.get("message")
            if not isinstance(code, str) or not code:
                raise ValueError(
                    "Deep Research protocol requires 'error.code' to be a non-empty string"
                )
            if not isinstance(message, str) or not message:
                raise ValueError(
                    "Deep Research protocol requires 'error.message' to be a non-empty string"
                )

    def _validate_deep_research_answer_content(self, content: Any) -> None:
        """
        Validate the structured Deep Research answer payload.

        Args:
            content: Deep Research answer content payload.

        Raises:
            ValueError: If the structured answer payload is invalid.
        """
        if not isinstance(content, dict):
            raise ValueError(
                "Deep Research protocol requires 'answer' content to be an object"
            )

        summary = content.get("summary")
        claims = content.get("claims")
        sources = content.get("sources")
        insufficient_evidence = content.get("insufficient_evidence")

        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(
                "Deep Research structured answers require a non-empty 'summary' string"
            )
        if not isinstance(claims, list):
            raise ValueError(
                "Deep Research structured answers require 'claims' to be an array"
            )
        if not isinstance(sources, list):
            raise ValueError(
                "Deep Research structured answers require 'sources' to be an array"
            )
        if not isinstance(insufficient_evidence, bool):
            raise ValueError(
                "Deep Research structured answers require 'insufficient_evidence' to be a boolean"
            )

        source_urls: set[str] = set()
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError(
                    "Deep Research structured answer sources must be objects"
                )
            url = source.get("url")
            title = source.get("title")
            evidence = source.get("evidence")
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                raise ValueError(
                    "Deep Research structured answer sources require a valid 'url'"
                )
            if not isinstance(title, str) or not title.strip():
                raise ValueError(
                    "Deep Research structured answer sources require a non-empty 'title'"
                )
            if not isinstance(evidence, str) or not evidence.strip():
                raise ValueError(
                    "Deep Research structured answer sources require a non-empty 'evidence'"
                )
            source_urls.add(url)

        if insufficient_evidence:
            if claims:
                raise ValueError(
                    "Deep Research insufficient-evidence answers must not include claims"
                )
            lowered_summary = summary.lower()
            if not any(
                phrase in lowered_summary
                for phrase in (
                    "insufficient evidence",
                    "evidence is insufficient",
                    "insufficient information",
                )
            ):
                raise ValueError(
                    "Deep Research insufficient-evidence answers must explain that evidence is insufficient"
                )
            return

        if not claims:
            raise ValueError(
                "Deep Research grounded answers require at least one claim"
            )
        if not sources:
            raise ValueError(
                "Deep Research grounded answers require at least one source"
            )

        for claim in claims:
            if not isinstance(claim, dict):
                raise ValueError(
                    "Deep Research structured answer claims must be objects"
                )
            statement = claim.get("statement")
            claim_source_urls = claim.get("source_urls")
            if not isinstance(statement, str) or not statement.strip():
                raise ValueError(
                    "Deep Research structured answer claims require a non-empty 'statement'"
                )
            if not isinstance(claim_source_urls, list) or not claim_source_urls:
                raise ValueError(
                    "Deep Research structured answer claims require non-empty 'source_urls'"
                )
            for source_url in claim_source_urls:
                if not isinstance(source_url, str) or source_url not in source_urls:
                    raise ValueError(
                        "Deep Research structured answer claim sources must reference top-level sources"
                    )

    def _validate_tool_call(self, tool_call: Any) -> None:
        """
        Validate a Deep Research tool call object.

        Args:
            tool_call: Tool call payload to validate.

        Raises:
            ValueError: If the tool call shape is invalid.
        """
        if not isinstance(tool_call, dict):
            raise ValueError("Deep Research protocol requires tool calls to be objects")

        name = tool_call.get("name")
        arguments = tool_call.get("arguments")
        if not isinstance(name, str) or not name:
            raise ValueError(
                "Deep Research protocol requires tool calls to include a non-empty 'name'"
            )
        if not isinstance(arguments, dict):
            raise ValueError(
                "Deep Research protocol requires tool calls to include an object 'arguments'"
            )
        if "depends_on" in tool_call:
            raise ValueError(
                "Deep Research protocol does not support 'depends_on' in tool calls"
            )

    def _validate_deep_research_audit_response(self, data: dict[str, Any]) -> None:
        """
        Validate the Deep Research LLM audit verdict payload.

        Args:
            data: Parsed audit verdict object.

        Raises:
            ValueError: If the audit verdict schema is invalid.
        """
        supported = data.get("supported")
        issues = data.get("issues")
        per_source = data.get("per_source")

        if not isinstance(supported, bool):
            raise ValueError(
                "Deep Research audit verdict requires 'supported' to be a boolean"
            )
        if not isinstance(issues, list):
            raise ValueError(
                "Deep Research audit verdict requires 'issues' to be an array"
            )
        for issue in issues:
            if not isinstance(issue, str) or not issue.strip():
                raise ValueError(
                    "Deep Research audit verdict issues must be non-empty strings"
                )

        if not isinstance(per_source, list):
            raise ValueError(
                "Deep Research audit verdict requires 'per_source' to be an array"
            )
        for source_verdict in per_source:
            if not isinstance(source_verdict, dict):
                raise ValueError(
                    "Deep Research audit verdict source entries must be objects"
                )
            url = source_verdict.get("url")
            source_supported = source_verdict.get("supported")
            reason = source_verdict.get("reason")
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                raise ValueError(
                    "Deep Research audit verdict source entries require a valid 'url'"
                )
            if not isinstance(source_supported, bool):
                raise ValueError(
                    "Deep Research audit verdict source entries require 'supported' to be a boolean"
                )
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(
                    "Deep Research audit verdict source entries require a non-empty 'reason'"
                )

    def _clean_json_text(self, text: str) -> str:
        """
        Clean the text to extract JSON.

        - Removes Markdown code blocks.
        - Strips whitespace.
        """
        text = text.strip()

        # Check for markdown code blocks
        json_block_match = re.search(
            r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE
        )
        if json_block_match:
            return json_block_match.group(1).strip()

        return text
