"""
ReAct Agent Implementation
"""

import asyncio
import inspect
import logging
from collections.abc import Callable
from typing import Any

from ..llm.client import LLMMessage
from ..llm.converters import convert_langchain_to_llm_messages
from ..llm.parsers import JSONOutputParser
from ..llm.stream_handler import StreamTokenHandler
from ..prompt.manager import PromptManager
from .base import BaseAgent
from .types import AgentState, AgentStep


class ReActAgent(BaseAgent):
    """
    Agent implementing the ReAct (Reasoning and Acting) framework.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.max_steps = self.config.get("max_steps", 15)
        self.logger = logging.getLogger("agent.react")
        self.prompt_manager = PromptManager()
        self.parser = JSONOutputParser()

    async def process_input(
        self, user_input: str, callbacks: dict[str, Callable] | None = None, **kwargs
    ) -> str:
        """
        Process user input using the ReAct loop.
        """
        if self.state != AgentState.READY:
            return "Agent is not ready. Please wait for initialization."

        try:
            self.state = AgentState.PROCESSING
            self.logger.info(f"Processing input: '{user_input[:50]}...'")

            response = await self._run_loop(user_input, callbacks)
            logging.info(f"ReAct loop completed. Response: {response}")

            self.state = AgentState.READY
            return response

        except Exception as e:
            self.state = AgentState.ERROR
            self.logger.error(f"Error processing input: {e}")
            return f"Error: {str(e)}"

    async def _run_loop(
        self, user_input: str, callbacks: dict[str, Callable] | None = None
    ) -> str:
        """Execute the ReAct loop."""
        steps: list[AgentStep] = []

        # Initial message
        messages = self._build_initial_messages(user_input)

        for i in range(self.max_steps):
            self.logger.info(f"ReAct Step {i + 1}/{self.max_steps}")

            # 1. Get LLM Response
            try:
                response_text = await self._generate_and_stream_response(
                    messages, callbacks
                )
            except Exception as e:
                return f"Error communicating with LLM: {e}"

            self.logger.debug(f"LLM Response: {response_text}")

            # 2. Parse JSON Response
            try:
                parsed_response = self.parser.parse(response_text)
            except ValueError as e:
                self.logger.error(f"Failed to parse JSON response: {e}")
                messages.append(LLMMessage(role="assistant", content=response_text))
                messages.append(
                    LLMMessage(
                        role="user",
                        content=f"Error: Invalid JSON output. Please output valid JSON matching the schema. Error: {e}",
                    )
                )
                continue

            thought = parsed_response.get("thought", "")
            msg_type = parsed_response.get("type", "answer")
            content = parsed_response.get("content", "")
            self._record_protocol_response(response_text, parsed_response)

            messages.append(LLMMessage(role="assistant", content=response_text))

            if msg_type == "answer":
                return content if isinstance(content, str) else str(content)

            elif msg_type == "tool_call":
                if isinstance(content, dict):
                    tool_name = content.get("name")
                    tool_args = content.get("arguments", {})

                    self.logger.info(f"Executing tool: {tool_name}")

                    try:
                        result = await self.execute_tool(tool_name, tool_args)
                    except Exception as e:
                        result = f"Error executing tool: {e}"

                    self.logger.info(f"Tool result: {result}")

                    messages.append(
                        LLMMessage(role="user", content=f"Observation: {result}")
                    )

                    steps.append(
                        AgentStep(
                            thought=thought,
                            action=tool_name,
                            action_input=tool_args,
                            observation=str(result),
                        )
                    )
                else:
                    messages.append(
                        LLMMessage(
                            role="user",
                            content="Error: Tool call content must be a JSON object.",
                        )
                    )
            elif msg_type == "tool_calls":
                if isinstance(content, list):
                    results = await asyncio.gather(
                        *[
                            self.execute_tool(
                                tool_call.get("name"),
                                tool_call.get("arguments", {}),
                            )
                            for tool_call in content
                        ],
                        return_exceptions=True,
                    )
                    observations: list[str] = []
                    for tool_call, result in zip(content, results, strict=False):
                        tool_name = tool_call.get("name")
                        tool_args = tool_call.get("arguments", {})
                        observation = (
                            f"Error executing tool: {result}"
                            if isinstance(result, Exception)
                            else str(result)
                        )
                        observations.append(
                            f"{tool_name}({tool_args}) => {observation}"
                        )
                        steps.append(
                            AgentStep(
                                thought=thought,
                                action=tool_name,
                                action_input=tool_args,
                                observation=observation,
                            )
                        )
                    messages.append(
                        LLMMessage(
                            role="user",
                            content="Observation: " + "\n".join(observations),
                        )
                    )
                else:
                    messages.append(
                        LLMMessage(
                            role="user",
                            content="Error: Parallel tool calls must be a JSON array.",
                        )
                    )
            elif msg_type == "error":
                if isinstance(content, dict):
                    error_code = content.get("code")
                    error_message = content.get("message", "Unknown error")
                    if error_code:
                        return f"Error ({error_code}): {error_message}"
                    return f"Error: {error_message}"
                return f"Error: {content}"
            else:
                self.logger.warning(f"Unknown message type: {msg_type}")
                # Treat as continue?
                pass

        return "Reached maximum steps without finding a final answer."

    async def _generate_and_stream_response(
        self,
        messages: list[LLMMessage],
        callbacks: dict[str, Callable] | None = None,
    ) -> str:
        """
        Generate response with streaming and callbacks using StreamTokenHandler.
        Parses JSON during streaming.
        """
        handler = StreamTokenHandler(callbacks)
        stream_method = getattr(self.llm_manager, "stream_response", None)
        if callable(stream_method):
            stream_result = stream_method(messages)
            if inspect.isawaitable(stream_result):
                stream_result = await stream_result
            if inspect.isasyncgen(stream_result):
                return await handler.process_stream(stream_result)

        generate_method = getattr(self.llm_manager, "generate_response", None)
        if callable(generate_method):
            response = await generate_method(messages)
            if hasattr(response, "content"):
                return str(response.content)
            return str(response)

        raise RuntimeError("LLM manager does not provide a usable response method")

    def _build_initial_messages(self, user_input: str) -> list[LLMMessage]:
        """Build the initial prompt messages."""
        system_messages = self._get_system_prompt()
        llm_system_messages = convert_langchain_to_llm_messages(system_messages)

        return llm_system_messages + [LLMMessage(role="user", content=user_input)]

    async def execute_tool(self, name: str, params: Any) -> Any:
        """Execute a tool. Override in subclasses."""
        return f"Error: Tool execution not implemented for '{name}'"

    def _get_system_prompt(self) -> list[Any]:
        """Get the system prompt. Override in subclasses."""
        messages = self.prompt_manager.render_template("react_system")
        return messages

    def _record_protocol_response(
        self, raw_response: str, parsed_response: dict[str, Any]
    ) -> None:
        """
        Record protocol-level response data for runtimes that need persistence.

        Args:
            raw_response: Raw response string returned by the model.
            parsed_response: Normalized parsed response object.
        """
        return None
