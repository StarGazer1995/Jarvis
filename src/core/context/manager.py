"""
Conversation Context Manager for ARK Engine

This module manages conversation context, memory, and state across
interactions in the ARK-powered Jarvis system.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from ..config.loader import load_llm_config
from ..llm.types import LLMMessage


@dataclass
class ConversationTurn:
    """
    Represents a single conversation turn between user and agent.
    """

    user_input: str
    agent_response: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    intent: str | None = None
    entities: dict[str, Any] = field(default_factory=dict)
    tools_used: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_response: str | None = None

    @property
    def processing_time(self) -> float:
        """Get processing time from metadata."""
        return self.metadata.get("processing_time", 0.0)

    @processing_time.setter
    def processing_time(self, value: float) -> None:
        """Set processing time in metadata."""
        self.metadata["processing_time"] = value

    def to_dict(self) -> dict[str, Any]:
        """
        Convert conversation turn to dictionary.

        Returns:
            Dictionary representation of the turn
        """
        return {
            "user_input": self.user_input,
            "agent_response": self.agent_response,
            "timestamp": self.timestamp,
            "intent": self.intent,
            "entities": self.entities,
            "tools_used": self.tools_used,
            "metadata": self.metadata,
            "raw_response": self.raw_response,
        }

    def to_langchain_message(self) -> list[BaseMessage]:
        """
        Convert conversation turn to LangChain messages.

        Returns:
            List of LangChain BaseMessage objects (HumanMessage, AIMessage)
        """
        messages = []
        if self.user_input:
            messages.append(HumanMessage(content=self.user_input))
        if self.agent_response:
            # Use raw JSON response if available, otherwise use processed response
            content = self.raw_response if self.raw_response else self.agent_response
            messages.append(AIMessage(content=content))
        return messages

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationTurn":
        """
        Create conversation turn from dictionary.

        Args:
            data: Dictionary containing turn data

        Returns:
            ConversationTurn instance
        """
        return cls(
            user_input=data["user_input"],
            agent_response=data["agent_response"],
            timestamp=data.get("timestamp", datetime.now().timestamp()),
            intent=data.get("intent"),
            entities=data.get("entities", {}),
            tools_used=data.get("tools_used", []),
            metadata=data.get("metadata", {}),
            raw_response=data.get("raw_response"),
        )


class ConversationContext:
    """
    Manages conversation context and memory for ARK engine.

    This class maintains conversation history, user preferences,
    and contextual information that helps ARK make better decisions.
    """

    def __init__(
        self,
        max_history: int = 100,
        session_id: str | None = None,
        llm_manager: Any | None = None,
    ):
        """
        Initialize conversation context manager.

        Args:
            max_history: Maximum number of conversation turns to keep in memory
            session_id: Optional session identifier. If not provided, a new one will be generated
            llm_manager: Optional shared LLM manager used for history compression
        """
        self.max_history = max_history
        self.conversation_history: list[ConversationTurn] = []
        self.user_preferences: dict[str, Any] = {}
        self.session_metadata: dict[str, Any] = {}
        self.current_context: dict[str, Any] = {}
        self.llm_manager = llm_manager
        self.ark_logger = logging.getLogger("ark.context")
        self.ark_parent_logger = logging.getLogger("ark")
        # Reset leaked logger state from earlier tests, but preserve an explicit
        # DEBUG level already configured by caplog for this logger.
        if self.ark_logger.level != logging.DEBUG:
            self.ark_logger.setLevel(logging.NOTSET)
        if self.ark_parent_logger.level != logging.DEBUG:
            self.ark_parent_logger.setLevel(logging.NOTSET)

        # Initialize session
        self.session_start = datetime.now()
        self.created_at = self.session_start.timestamp()
        self.last_activity = self.created_at

        # Set session_id as instance attribute for easy access
        self.session_id = session_id if session_id else self._generate_session_id()

        self.session_metadata = {
            "session_id": self.session_id,
            "start_time": self.session_start.isoformat(),
            "turn_count": 0,
        }

        self.ark_logger.info(
            f"ARK context manager initialized for session: {self.session_metadata['session_id']}"
        )

    @property
    def user_memory(self) -> dict[str, Any]:
        """Alias for user_preferences for backward compatibility."""
        return self.user_preferences

    @user_memory.setter
    def user_memory(self, value: dict[str, Any]) -> None:
        """Setter for user_memory alias."""
        self.user_preferences = value

    def _generate_session_id(self) -> str:
        """Generate a unique session identifier."""
        import uuid

        return str(uuid.uuid4())[:8]

    def add_exchange(
        self,
        user_input: str,
        agent_response: str,
        intent: str | None = None,
        tools_used: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        raw_response: str | None = None,
    ) -> None:
        """
        Add a conversation exchange to the context.

        Args:
            user_input: The user's input
            agent_response: The agent's response
            intent: Recognized intent (if any)
            tools_used: List of tools used in this exchange
            metadata: Additional metadata for this turn
            raw_response: Raw JSON response from LLM
        """
        turn = ConversationTurn(
            user_input=user_input,
            agent_response=agent_response,
            timestamp=datetime.now().timestamp(),
            intent=intent,
            tools_used=tools_used or [],
            metadata=metadata or {},
            raw_response=raw_response,
        )

        self.conversation_history.append(turn)
        self.ark_logger.debug(
            f"ARK context: Added exchange with intent '{intent}', tools: {tools_used}"
        )

        # Maintain history size limit
        if len(self.conversation_history) > self.max_history:
            removed_turn = self.conversation_history.pop(0)
            self.ark_logger.debug(
                f"ARK context: Removed old turn from {datetime.fromtimestamp(removed_turn.timestamp)}"
            )

        # Update session metadata and activity timestamp
        self.session_metadata["turn_count"] += 1
        self.session_metadata["last_activity"] = datetime.fromtimestamp(
            turn.timestamp
        ).isoformat()
        self.last_activity = turn.timestamp

        self.ark_logger.debug(
            f"ARK context: Added exchange with intent '{intent}', tools: {tools_used}"
        )

    def add_turn(self, turn: ConversationTurn) -> None:
        """
        Add a conversation turn directly to the history.

        Args:
            turn: ConversationTurn object to add
        """
        self.conversation_history.append(turn)

        # Maintain history size limit
        if len(self.conversation_history) > self.max_history:
            removed_turn = self.conversation_history.pop(0)
            self.ark_logger.debug(
                f"ARK context: Removed old turn from {datetime.fromtimestamp(removed_turn.timestamp)}"
            )

        # Update session metadata and activity timestamp
        self.session_metadata["turn_count"] += 1
        self.session_metadata["last_activity"] = datetime.fromtimestamp(
            turn.timestamp
        ).isoformat()
        self.last_activity = turn.timestamp

        self.ark_logger.debug(
            f"ARK context: Added turn with intent '{turn.intent}', tools: {turn.tools_used}"
        )

    def update_memory(self, key: str, value: Any) -> None:
        """
        Update user memory with a key-value pair.

        Args:
            key: Memory key
            value: Memory value
        """
        self.user_preferences[key] = value
        self.ark_logger.debug(f"ARK context: Updated memory {key}")

    def get_memory(self, key: str, default: Any = None) -> Any:
        """
        Get a value from user memory.

        Args:
            key: Memory key
            default: Default value if key not found

        Returns:
            Memory value or default
        """
        return self.user_preferences.get(key, default)

    def clear_memory(self) -> None:
        """
        Clear all user memory.
        """
        self.user_preferences.clear()
        self.ark_logger.debug("ARK context: Cleared user memory")

    def __len__(self) -> int:
        """Return the number of conversation turns."""
        return len(self.conversation_history)

    def __iter__(self):
        """Iterate over conversation turns."""
        return iter(self.conversation_history)

    def reset(self) -> None:
        """
        Reset the conversation context while preserving session ID.
        """
        old_session_id = self.session_id
        self.conversation_history.clear()
        self.user_preferences.clear()
        self.session_metadata.clear()

        # Restore session ID and reinitialize timestamps
        self.session_id = old_session_id
        self.session_start = datetime.now()
        self.created_at = self.session_start.timestamp()
        self.last_activity = self.created_at

        self.ark_logger.info(
            f"ARK context: Reset context for session {self.session_id}"
        )

    def __str__(self) -> str:
        """String representation of the context."""
        return f"ConversationContext(session_id={self.session_id}, turns={len(self.conversation_history)})"

    def __repr__(self) -> str:
        """Detailed string representation of the context."""
        return (
            f"ConversationContext(session_id={self.session_id}, "
            f"turns={len(self.conversation_history)}, "
            f"max_history={self.max_history})"
        )

    def get_recent_turns(self, num_turns: int = 5) -> list[ConversationTurn]:
        """
        Get the most recent conversation turns in reverse chronological order.

        Args:
            num_turns: Number of recent turns to retrieve

        Returns:
            List of recent ConversationTurn objects, most recent first
        """
        if not self.conversation_history:
            return []

        # Get the last num_turns and reverse to get most recent first
        recent = self.conversation_history[-num_turns:]
        return list(reversed(recent))

    def get_conversation_summary(self) -> dict[str, Any]:
        """
        Generate a summary of the current conversation context.

        Returns:
            Dictionary summary of the conversation context
        """
        if not self.conversation_history:
            return {
                "total_turns": 0,
                "session_id": self.session_metadata["session_id"],
                "intents": [],
                "duration": 0,
            }

        # Collect unique intents
        intents = list(
            set(turn.intent for turn in self.conversation_history if turn.intent)
        )

        # Calculate session duration
        first_turn = self.conversation_history[0]
        last_turn = self.conversation_history[-1]
        duration = last_turn.timestamp - first_turn.timestamp

        return {
            "total_turns": len(self.conversation_history),
            "session_id": self.session_metadata["session_id"],
            "intents": intents,
            "duration": duration,
        }

    def get_cleaned_history(self, max_messages: int = 20) -> list[BaseMessage]:
        """
        Get cleaned conversation history formatted as LangChain messages.

        Implements cleaning rules R1-R6:
        R1: Clear history before "reset"/"clear" commands.
        R2: Filter invalid/error messages.
        R3: Remove trailing assistant messages.
        R4: Merge consecutive user messages.
        R5: Truncate based on token/count budget.
        R6: Ensure history starts with user message.

        Args:
            max_messages: Maximum number of messages to return (R5)

        Returns:
            List of cleaned LangChain BaseMessage objects
        """
        # Convert all turns to a flat list of messages
        raw_messages: list[BaseMessage] = []
        for turn in self.conversation_history:
            raw_messages.extend(turn.to_langchain_message())

        if not raw_messages:
            return []

        # R1: Clear history before "reset"/"clear" commands
        # Find the last occurrence of a reset command
        reset_index = -1
        reset_keywords = {"reset", "clear", "restart", "start over"}

        for i, msg in enumerate(raw_messages):
            if (
                isinstance(msg, HumanMessage)
                and msg.content.strip().lower() in reset_keywords
            ):
                reset_index = i

        if reset_index != -1:
            # Keep messages starting from AFTER the reset command?
            # Or should we keep the reset command itself?
            # Usually if I say "reset", the history should be empty for the NEXT turn.
            # But here we are retrieving history.
            # If the history contains "reset", it means the user said it in the past.
            # If the user said "reset", we probably want to ignore everything before it.
            # Let's drop everything up to and including the reset command.
            # Wait, if there was a response to "reset" (e.g. "Okay"), we might want to drop that too?
            # But let's start simple: drop everything up to the reset command.
            raw_messages = raw_messages[reset_index + 1 :]

        if not raw_messages:
            return []

        cleaned_messages: list[BaseMessage] = []

        for msg in raw_messages:
            if not msg.content or not msg.content.strip():
                continue
            cleaned_messages.append(msg)

        if not cleaned_messages:
            return []

        # R3: Remove trailing assistant messages
        # Refined R3: Only remove if empty or invalid. R2 already handles empty.
        # So we generally WANT history to end with AI, so that when we append User, we get H-A-H pattern.
        # However, if the last message is AI and we are about to add another AI (unlikely here as we add User),
        # or if we want to force user to answer user (unlikely).
        # The only case to remove trailing AI is if it's "incomplete" or we are retrying.
        # Since R2 filters empty, we assume existing AIs are valid.
        # So we SKIP unconditional removal.
        # while cleaned_messages and isinstance(cleaned_messages[-1], AIMessage):
        #    cleaned_messages.pop()

        if not cleaned_messages:
            return []

        # R4: Merge consecutive user messages

        merged_messages: list[BaseMessage] = []
        current_user_buffer: list[str] = []

        for msg in cleaned_messages:
            if isinstance(msg, HumanMessage):
                current_user_buffer.append(msg.content)
            else:
                if current_user_buffer:
                    # Flush user buffer
                    merged_content = "\n".join(current_user_buffer)
                    merged_messages.append(HumanMessage(content=merged_content))
                    current_user_buffer = []
                merged_messages.append(msg)

        # Flush remaining user buffer
        if current_user_buffer:
            merged_content = "\n".join(current_user_buffer)
            merged_messages.append(HumanMessage(content=merged_content))

        cleaned_messages = merged_messages

        if not cleaned_messages:
            return []

        # R5: Truncate based on count budget
        if len(cleaned_messages) > max_messages:
            cleaned_messages = cleaned_messages[-max_messages:]

        # R6: Ensure history starts with user message
        # If the first message is an AIMessage, remove it.
        if cleaned_messages and isinstance(cleaned_messages[0], AIMessage):
            cleaned_messages.pop(0)

        return cleaned_messages

    def update_user_preference(self, key: str, value: Any) -> None:
        """
        Update a user preference.

        Args:
            key: Preference key
            value: Preference value
        """
        self.user_preferences[key] = value
        self.ark_logger.debug(f"ARK context: Updated user preference '{key}' = {value}")

    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """
        Get a user preference.

        Args:
            key: Preference key
            default: Default value if preference not found

        Returns:
            Preference value or default
        """
        return self.user_preferences.get(key, default)

    def set_context_variable(self, key: str, value: Any) -> None:
        """
        Set a context variable for the current session.

        Args:
            key: Variable key
            value: Variable value
        """
        self.current_context[key] = value
        self.ark_logger.debug(f"ARK context: Set context variable '{key}' = {value}")

    def get_context_variable(self, key: str, default: Any = None) -> Any:
        """
        Get a context variable.

        Args:
            key: Variable key
            default: Default value if variable not found

        Returns:
            Variable value or default
        """
        return self.current_context.get(key, default)

    def clear_context_variables(self) -> None:
        """Clear all context variables."""
        self.current_context.clear()
        self.ark_logger.debug("ARK context: Cleared all context variables")

    def set_llm_manager(self, llm_manager: Any | None) -> None:
        """
        Attach a shared LLM manager to the context manager.

        Args:
            llm_manager: Shared LLM manager used by compression flows
        """
        self.llm_manager = llm_manager

    def get_session_stats(self) -> dict[str, Any]:
        """
        Get statistics about the current session.

        Returns:
            Dictionary with session statistics
        """
        if not self.conversation_history:
            return {
                "total_turns": 0,
                "turn_count": 0,
                "unique_intents": 0,
                "tools_used": 0,
                "session_duration": 0,
                "duration_minutes": 0.0,
                "average_response_time": 0.0,
            }

        # Calculate session duration
        first_turn = self.conversation_history[0]
        last_turn = self.conversation_history[-1]
        session_duration = last_turn.timestamp - first_turn.timestamp

        # Collect unique intents
        intents_used = set(
            turn.intent for turn in self.conversation_history if turn.intent
        )

        # Collect unique tools used
        all_tools = set()
        for turn in self.conversation_history:
            all_tools.update(turn.tools_used)

        # Calculate average response time
        response_times = [
            turn.processing_time
            for turn in self.conversation_history
            if hasattr(turn, "processing_time") and turn.processing_time > 0
        ]
        average_response_time = (
            sum(response_times) / len(response_times) if response_times else 0.0
        )

        return {
            "total_turns": len(self.conversation_history),
            "turn_count": len(self.conversation_history),
            "unique_intents": len(intents_used),
            "tools_used": len(all_tools),
            "session_duration": session_duration,
            "duration_minutes": session_duration / 60.0,
            "average_response_time": average_response_time,
        }

    async def _summarize_with_llm_manager(self, conversation_text: str) -> str:
        """
        Summarize conversation history via the shared LLM manager.

        Args:
            conversation_text: Flattened conversation text to summarize

        Returns:
            Summary text generated by the shared LLM manager
        """
        if self.llm_manager is None:
            raise ValueError("LLM manager is not configured for compression")

        messages = [
            LLMMessage(
                role="system",
                content=(
                    "Summarize the following conversation concisely, capturing "
                    "key information, user preferences, and decisions made."
                ),
            ),
            LLMMessage(
                role="user",
                content=f"Conversation:\n{conversation_text}\n\nSummary:",
            ),
        ]
        response = await self.llm_manager.generate_response(messages, temperature=0.3)
        summary = response.content.strip()
        if not summary:
            raise ValueError("Compression summary cannot be empty")
        return summary

    def export_conversation(self, export_format: str = "json") -> str:
        """
        Export conversation history in specified format.

        Args:
            export_format: Export format ("json" or "text")

        Returns:
            Exported conversation data
        """
        if export_format == "json":
            export_data = {
                "session_metadata": self.session_metadata,
                "user_preferences": self.user_preferences,
                "conversation_history": [
                    {
                        "timestamp": datetime.fromtimestamp(turn.timestamp).isoformat(),
                        "user_input": turn.user_input,
                        "agent_response": turn.agent_response,
                        "intent": turn.intent,
                        "tools_used": turn.tools_used,
                        "metadata": turn.metadata,
                    }
                    for turn in self.conversation_history
                ],
            }
            return json.dumps(export_data, indent=2)

        elif export_format == "text":
            lines = []
            lines.append(
                f"Conversation Export - Session: {self.session_metadata['session_id']}"
            )
            lines.append(f"Started: {self.session_start.isoformat()}")
            lines.append("=" * 50)

            for i, turn in enumerate(self.conversation_history, 1):
                lines.append(
                    f"\nTurn {i} - {datetime.fromtimestamp(turn.timestamp).strftime('%H:%M:%S')}"
                )
                lines.append(f"User: {turn.user_input}")
                lines.append(f"Jarvis: {turn.agent_response}")
                if turn.intent:
                    lines.append(f"Intent: {turn.intent}")
                if turn.tools_used:
                    lines.append(f"Tools: {', '.join(turn.tools_used)}")
                lines.append("-" * 30)

            return "\n".join(lines)

        else:
            raise ValueError(f"Unsupported export format: {export_format}")

    async def compress_history(self, threshold: int = 20, keep_recent: int = 5) -> None:
        """
        Compress conversation history if it exceeds the threshold.

        Args:
            threshold: Maximum number of turns before compression triggers
            keep_recent: Number of recent turns to keep uncompressed
        """
        if len(self.conversation_history) <= threshold:
            return

        self.ark_logger.info("ARK context: Triggering conversation compression")

        # Identify turns to compress
        turns_to_compress = self.conversation_history[:-keep_recent]
        if not turns_to_compress:
            return

        # Format turns for summarization
        conversation_text = ""
        for turn in turns_to_compress:
            conversation_text += f"User: {turn.user_input}\n"
            conversation_text += f"Agent: {turn.agent_response}\n\n"

        try:
            summary: str | None = None

            if self.llm_manager is not None:
                try:
                    summary = await self._summarize_with_llm_manager(conversation_text)
                except Exception as exc:
                    self.ark_logger.warning(
                        "ARK context: Shared LLM manager compression failed, "
                        f"falling back to provider config. Error: {exc}"
                    )

            # Load LLM config
            if summary is None:
                llm_config = load_llm_config()
                provider_name = llm_config.global_config.default_provider
                provider_config = llm_config.providers.get(provider_name)

                if not provider_config:
                    self.ark_logger.warning(
                        f"ARK context: Provider {provider_name} not found, skipping compression"
                    )
                    return

                # Initialize LangChain model based on provider
                llm = None
                provider_type = provider_config.type or provider_name

                # Prepare common parameters
                # Note: LangChain models expect 'model' or 'model_name' depending on the class,
                # but most support 'model' as alias or kwargs.

                if provider_type == "openai":
                    params = {
                        "model": provider_config.default_model,
                        "temperature": 0.3,
                    }
                    if provider_config.api_key:
                        params["api_key"] = provider_config.api_key
                    if provider_config.base_url:
                        params["openai_api_base"] = provider_config.base_url

                    llm = ChatOpenAI(**params)

                elif provider_type == "anthropic":
                    params = {
                        "model": provider_config.default_model,
                        "temperature": 0.3,
                    }
                    if provider_config.api_key:
                        params["api_key"] = provider_config.api_key
                    if provider_config.base_url:
                        params["anthropic_api_url"] = provider_config.base_url

                    llm = ChatAnthropic(**params)

                elif provider_type == "ollama":
                    params = {
                        "model": provider_config.default_model,
                        "temperature": 0.3,
                    }
                    if provider_config.base_url:
                        params["base_url"] = provider_config.base_url

                    llm = ChatOllama(**params)

                else:
                    self.ark_logger.warning(
                        f"ARK context: Unsupported provider type {provider_type} for compression"
                    )
                    return

                # Create summarization chain
                prompt = PromptTemplate.from_template(
                    "Summarize the following conversation concisely, capturing key information, user preferences, and decisions made.\n\n"
                    "Conversation:\n{conversation}\n\n"
                    "Summary:"
                )

                chain = prompt | llm | StrOutputParser()

                # Generate summary
                summary = await chain.ainvoke({"conversation": conversation_text})

            # Create summary turn
            summary_turn = ConversationTurn(
                user_input="System: Previous conversation summary",
                agent_response=summary,
                timestamp=datetime.now().timestamp(),
                metadata={
                    "is_summary": True,
                    "compressed_turns": len(turns_to_compress),
                },
            )

            # Update history
            self.conversation_history = [summary_turn] + self.conversation_history[
                -keep_recent:
            ]
            self.ark_logger.info(
                f"ARK context: Compressed {len(turns_to_compress)} turns into summary"
            )

        except Exception as e:
            self.ark_logger.error(f"ARK context: Compression failed: {e}")

    def reset_session(self) -> None:
        """Reset the conversation context for a new session."""
        old_session_id = self.session_metadata.get("session_id", "unknown")

        self.conversation_history.clear()
        self.current_context.clear()
        self.session_start = datetime.now()
        self.session_metadata = {
            "session_id": self._generate_session_id(),
            "start_time": self.session_start.isoformat(),
            "turn_count": 0,
        }
        self.session_id = self.session_metadata["session_id"]

        self.ark_logger.info(
            f"ARK context: Reset session from {old_session_id} to {self.session_metadata['session_id']}"
        )
