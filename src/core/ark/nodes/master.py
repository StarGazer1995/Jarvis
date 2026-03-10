import json
import logging
from typing import Dict, Any, List, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.runnables import RunnableConfig

from ..state import JarvisState
from ...llm.client import LLMManager, LLMMessage
from ...llm.converters import convert_langchain_to_llm_messages
from ..utils import AgentSpec
from ...llm.stream_handler import StreamTokenHandler
from ...llm.parsers import JSONOutputParser
from ...prompt.manager import PromptManager

logger = logging.getLogger("ark.nodes.master")


class MasterNode:
    """
    The main reasoning node (Agent) for Jarvis.
    Merges responsibilities of Supervisor (Scheduling) and Executor (Tools).
    """

    def __init__(
        self, llm_manager: LLMManager, agents: Optional[List[AgentSpec]] = None
    ):
        self.llm_manager = llm_manager
        self.agents = agents or []
        self.agent_map = {a.name: a for a in self.agents}
        self.prompt_manager = PromptManager()
        self.parser = JSONOutputParser()

    async def __call__(
        self, state: JarvisState, config: RunnableConfig
    ) -> Dict[str, Any]:
        """
        Execute the master agent logic.
        """
        # 1. Convert State Messages to LLM Messages
        messages = self._convert_messages(state["messages"])

        # 2. Dynamic System Prompt Injection
        system_messages = self._get_system_prompt(state)
        llm_system_messages = convert_langchain_to_llm_messages(system_messages)

        # Check if the first message is system; if so, update it; otherwise insert it
        if messages and messages[0].role == "system":
            messages = llm_system_messages + messages
        else:
            messages = llm_system_messages + messages

        # 3. Call LLM
        logger.debug("MasterNode calling LLM...")
        try:
            callbacks = (
                config.get("configurable", {}).get("callbacks") if config else None
            )
            handler = StreamTokenHandler(callbacks)

            # Enforce JSON mode for the Orchestrator
            raw_content = await handler.process_stream(
                self.llm_manager.stream_response(
                    messages,
                )
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raw_content = f"Error: Failed to generate response from LLM: {e}"

        # 4. Clean and Parse JSON Response
        try:
            # Clean raw_content to handle Markdown code blocks and common JSON issues
            cleaned_content = raw_content.strip()
            # Remove Markdown code blocks if present
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:]
            elif cleaned_content.startswith("```"):
                cleaned_content = cleaned_content[3:]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]
            cleaned_content = cleaned_content.strip()

            parsed_response = self.parser.parse(cleaned_content)
        except ValueError as e:
            logger.error(
                f"Failed to parse JSON response: {e}. Raw content: {raw_content[:100]}..."
            )

            # Fallback: Try to use the raw content directly if it looks like an answer
            # This is a robust fallback for when LLM fails to output valid JSON but gives a valid text answer
            if not raw_content.startswith("{"):
                return {
                    "messages": [
                        AIMessage(
                            content=raw_content,
                            additional_kwargs={
                                "raw_json": raw_content,
                                "parse_error": str(e),
                            },
                        )
                    ],
                    "sender": "master",
                }

            return {
                "messages": [
                    AIMessage(content=f"Error: Invalid JSON response: {raw_content}")
                ],
                "sender": "master",
            }

        thought = parsed_response.get("thought", "")
        response_type = parsed_response.get("type", "answer")
        content = parsed_response.get("content", "")

        # 5. Handle Tool Calls
        if response_type == "tool_call" and isinstance(content, dict):
            tool_name = content.get("name")
            tool_args = content.get("arguments", {})

            # Construct AIMessage with tool_calls
            lc_tool_call = {
                "name": tool_name,
                "args": tool_args,
                "id": f"call_{len(state['messages'])}",
            }

            # Keep thought in context but don't expose it as main content if possible
            # But for ReAct, thought is usually prepended.
            # Let's keep it consistent: thought is internal reasoning.
            return {
                "messages": [
                    AIMessage(
                        content=thought,
                        tool_calls=[lc_tool_call],
                        additional_kwargs={"raw_json": raw_content},
                    )
                ],
                "sender": "master",
            }
        else:
            # Normal response (Answer)
            # Only return the content to the user, keep thought in metadata
            final_content = content if isinstance(content, str) else str(content)

            # Log the thought for debugging/audit
            logger.info(f"MasterNode Thought: {thought}")

            return {
                "messages": [
                    AIMessage(
                        content=final_content,
                        additional_kwargs={"raw_json": raw_content, "thought": thought},
                    )
                ],
                "sender": "master",
            }

    def _convert_messages(self, lc_messages: List[BaseMessage]) -> List[LLMMessage]:
        """Convert LangChain messages to internal LLMMessage format."""
        out = []
        for m in lc_messages:
            role = "user"
            if isinstance(m, AIMessage):
                role = "assistant"
            elif isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, HumanMessage):
                role = "user"
            # Skip ToolMessages here as they should be incorporated into history
            # differently depending on LLM provider support.
            # For ReAct (text-based), we append them as Observation text.
            # But here we are building a list of messages.
            # If the previous message was an AI message with tool_calls,
            # the LLM expects the tool output next.
            # Since our LLMManager is generic text-based (mostly), we might need to
            # flatten ToolMessages into the previous User/Assistant message or
            # send them as "user" role with "Observation: ..." prefix.

            # For simplicity in this ReAct implementation:
            # We treat ToolMessage as User message saying "Observation: ..."

            content = m.content
            if m.type == "tool":
                role = "user"
                content = f"Observation: {content}"
            elif m.name and m.name in self.agent_map:
                # Worker Output
                content = f"[WORKER OUTPUT from {m.name}]:\n{content}"
                role = "user"

            out.append(LLMMessage(role=role, content=str(content)))
        return out

    def _get_system_prompt(self, state: JarvisState) -> List[BaseMessage]:
        """Generate the system prompt based on state."""
        todo_list = state.get("todo_list", [])

        # 1. Todo List Status
        todo_status = "No tasks in todo list."
        if todo_list:
            todo_status = "Current Todo List:\n"
            for task in todo_list:
                status = task.get("status", "pending")
                desc = task.get("description", "")
                result = task.get("result", "")
                task_id = task.get("id", "")

                todo_status += f"- [{task_id}] {status}: {desc}"
                if result:
                    todo_status += f" (Result: {result})"
                todo_status += "\n"

        # 2. Available Tools
        tools_desc = "Available Tools:\n"
        available_tools = state.get("available_tools", {})
        if available_tools:
            for name, info in available_tools.items():
                desc = info.get("description", "No description")
                schema = json.dumps(info.get("input_schema", {}), indent=2)
                tools_desc += f"- {name}: {desc}\n  Schema: {schema}\n"
        else:
            tools_desc += "No tools available.\n"

        # 3. Available Agents (Workers)
        agents_desc = ""
        if self.agents:
            agents_desc = "Available Worker Agents:\n"
            for agent in self.agents:
                agents_desc += f"- {agent.name}: {agent.description}\n"

        messages = self.prompt_manager.render_template(
            "master_system",
            todo_status=todo_status,
            tools_desc=tools_desc,
            agents_desc=agents_desc,
        )
        return messages
