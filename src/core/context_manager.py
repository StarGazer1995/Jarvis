"""
Conversation Context Manager for ARK Engine

This module manages conversation context, memory, and state across
interactions in the ARK-powered Jarvis system.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class ConversationTurn:
    """
    Represents a single conversation turn between user and agent.
    """
    user_input: str
    agent_response: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    intent: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    tools_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def processing_time(self) -> float:
        """Get processing time from metadata."""
        return self.metadata.get("processing_time", 0.0)
    
    @processing_time.setter
    def processing_time(self, value: float) -> None:
        """Set processing time in metadata."""
        self.metadata["processing_time"] = value
    
    def to_dict(self) -> Dict[str, Any]:
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
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
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
            metadata=data.get("metadata", {})
        )


class ConversationContext:
    """
    Manages conversation context and memory for ARK engine.
    
    This class maintains conversation history, user preferences,
    and contextual information that helps ARK make better decisions.
    """
    
    def __init__(self, max_history: int = 100, session_id: Optional[str] = None):
        """
        Initialize conversation context manager.
        
        Args:
            max_history: Maximum number of conversation turns to keep in memory
            session_id: Optional session identifier. If not provided, a new one will be generated
        """
        self.max_history = max_history
        self.conversation_history: List[ConversationTurn] = []
        self.user_preferences: Dict[str, Any] = {}
        self.session_metadata: Dict[str, Any] = {}
        self.current_context: Dict[str, Any] = {}
        self.ark_logger = logging.getLogger('ark.context')
        
        # Initialize session
        self.session_start = datetime.now()
        self.created_at = self.session_start.timestamp()
        self.last_activity = self.created_at
        
        # Set session_id as instance attribute for easy access
        self.session_id = session_id if session_id else self._generate_session_id()
        
        self.session_metadata = {
            "session_id": self.session_id,
            "start_time": self.session_start.isoformat(),
            "turn_count": 0
        }
        
        self.ark_logger.info(f"ARK context manager initialized for session: {self.session_metadata['session_id']}")

    @property
    def user_memory(self) -> Dict[str, Any]:
        """Alias for user_preferences for backward compatibility."""
        return self.user_preferences
    
    @user_memory.setter
    def user_memory(self, value: Dict[str, Any]) -> None:
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
        intent: Optional[str] = None,
        tools_used: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a conversation exchange to the context.
        
        Args:
            user_input: The user's input
            agent_response: The agent's response
            intent: Recognized intent (if any)
            tools_used: List of tools used in this exchange
            metadata: Additional metadata for this turn
        """
        turn = ConversationTurn(
            user_input=user_input,
            agent_response=agent_response,
            timestamp=datetime.now().timestamp(),
            intent=intent,
            tools_used=tools_used or [],
            metadata=metadata or {}
        )
        
        self.conversation_history.append(turn)
        
        # Maintain history size limit
        if len(self.conversation_history) > self.max_history:
            removed_turn = self.conversation_history.pop(0)
            self.ark_logger.debug(f"ARK context: Removed old turn from {datetime.fromtimestamp(removed_turn.timestamp)}")
        
        # Update session metadata and activity timestamp
        self.session_metadata["turn_count"] += 1
        self.session_metadata["last_activity"] = datetime.fromtimestamp(turn.timestamp).isoformat()
        self.last_activity = turn.timestamp
        
        self.ark_logger.debug(f"ARK context: Added exchange with intent '{intent}', tools: {tools_used}")
    
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
            self.ark_logger.debug(f"ARK context: Removed old turn from {datetime.fromtimestamp(removed_turn.timestamp)}")
        
        # Update session metadata and activity timestamp
        self.session_metadata["turn_count"] += 1
        self.session_metadata["last_activity"] = datetime.fromtimestamp(turn.timestamp).isoformat()
        self.last_activity = turn.timestamp
        
        self.ark_logger.debug(f"ARK context: Added turn with intent '{turn.intent}', tools: {turn.tools_used}")
    
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
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'ConversationContext':
        """
        Load conversation context from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            ConversationContext instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file contains invalid JSON
        """
        import os
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Context file not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Use import_context to handle the data
            return cls.import_context(data)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in context file: {e}")
    
    def save_to_file(self, file_path: str) -> None:
        """
        Save conversation context to a JSON file.
        
        Args:
            file_path: Path to save the JSON file
        """
        export_data = {
            "max_history": self.max_history,
            "session_metadata": self.session_metadata,
            "user_memory": self.user_preferences,
            "conversation_history": [
                {
                    "timestamp": turn.timestamp,
                    "user_input": turn.user_input,
                    "agent_response": turn.agent_response,
                    "intent": turn.intent,
                    "entities": turn.entities,
                    "tools_used": turn.tools_used,
                    "metadata": turn.metadata
                }
                for turn in self.conversation_history
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.ark_logger.debug(f"ARK context: Saved context to {file_path}")
    
    def export_context(self) -> Dict[str, Any]:
        """
        Export conversation context as a dictionary.
        
        Returns:
            Dictionary containing all context data
        """
        return {
            "session_id": self.session_id,
            "conversation_history": [
                {
                    "user_input": turn.user_input,
                    "agent_response": turn.agent_response,
                    "intent": turn.intent,
                    "entities": turn.entities,
                    "tools_used": turn.tools_used,
                    "metadata": turn.metadata,
                    "timestamp": turn.timestamp,
                    "processing_time": turn.metadata.get("processing_time", 0.0)
                }
                for turn in self.conversation_history
            ],
            "user_memory": self.user_preferences,
            "session_metadata": self.session_metadata,
            "created_at": self.created_at,
            "last_activity": self.last_activity
        }
    
    @classmethod
    def import_context(cls, context_data: Dict[str, Any]) -> 'ConversationContext':
        """
        Import conversation context from a dictionary.
        
        Args:
            context_data: Dictionary containing context data
            
        Returns:
            ConversationContext instance
        """
        # Extract session_id from session_metadata if available
        session_metadata = context_data.get("session_metadata", {})
        session_id = session_metadata.get("session_id") or context_data.get("session_id")
        max_history = context_data.get("max_history", 100)
        
        # Create new context instance
        context = cls(max_history=max_history, session_id=session_id)
        
        # Restore data
        context.user_preferences = context_data.get("user_memory", {})
        context.session_metadata.update(context_data.get("session_metadata", {}))
        context.created_at = context_data.get("created_at", datetime.now().timestamp())
        context.last_activity = context_data.get("last_activity", datetime.now().timestamp())
        
        # Restore conversation history
        for turn_data in context_data.get("conversation_history", []):
            turn = ConversationTurn(
                user_input=turn_data["user_input"],
                agent_response=turn_data["agent_response"],
                timestamp=turn_data.get("timestamp", datetime.now().timestamp()),
                intent=turn_data.get("intent"),
                entities=turn_data.get("entities", {}),
                tools_used=turn_data.get("tools_used", []),
                metadata=turn_data.get("metadata", {})
            )
            context.conversation_history.append(turn)
        
        return context
    
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
        
        self.ark_logger.info(f"ARK context: Reset context for session {self.session_id}")
    
    def __str__(self) -> str:
        """String representation of the context."""
        return f"ConversationContext(session_id={self.session_id}, turns={len(self.conversation_history)})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the context."""
        return (f"ConversationContext(session_id={self.session_id}, "
                f"turns={len(self.conversation_history)}, "
                f"max_history={self.max_history})")
    
    def get_recent_context(self, num_turns: int = 5) -> List[ConversationTurn]:
        """
        Get recent conversation turns for context.
        
        Args:
            num_turns: Number of recent turns to retrieve
            
        Returns:
            List of recent conversation turns
        """
        return self.conversation_history[-num_turns:] if self.conversation_history else []

    def get_recent_turns(self, num_turns: int = 5) -> List[ConversationTurn]:
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

    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the current conversation context.
        
        Returns:
            Dictionary summary of the conversation context
        """
        if not self.conversation_history:
            return {
                "total_turns": 0,
                "session_id": self.session_metadata['session_id'],
                "intents": [],
                "duration": 0
            }
        
        # Collect unique intents
        intents = list(set(turn.intent for turn in self.conversation_history if turn.intent))
        
        # Calculate session duration
        first_turn = self.conversation_history[0]
        last_turn = self.conversation_history[-1]
        duration = last_turn.timestamp - first_turn.timestamp
        
        return {
            "total_turns": len(self.conversation_history),
            "session_id": self.session_metadata['session_id'],
            "intents": intents,
            "duration": duration
        }

    def get_context_summary(self) -> str:
        """
        Generate a string summary of the current conversation context.
        
        Returns:
            String summary of the conversation context
        """
        if not self.conversation_history:
            return "No conversation history available."
        
        recent_turns = self.get_recent_context(3)
        summary_parts = []
        
        summary_parts.append(f"Session: {self.session_metadata['session_id']}")
        summary_parts.append(f"Turn count: {self.session_metadata['turn_count']}")
        
        if recent_turns:
            summary_parts.append("Recent conversation:")
            for i, turn in enumerate(recent_turns, 1):
                summary_parts.append(f"  {i}. User: {turn.user_input[:50]}...")
                if turn.intent:
                    summary_parts.append(f"     Intent: {turn.intent}")
                if turn.tools_used:
                    summary_parts.append(f"     Tools: {', '.join(turn.tools_used)}")
        
        return "\n".join(summary_parts)
    
    def find_similar_exchanges(self, current_input: str, limit: int = 3) -> List[ConversationTurn]:
        """
        Find similar past exchanges based on input similarity.
        
        Args:
            current_input: Current user input to find similarities for
            limit: Maximum number of similar exchanges to return
            
        Returns:
            List of similar conversation turns
        """
        if not self.conversation_history:
            return []
        
        # Simple keyword-based similarity for now
        # In a production system, this could use embeddings or more sophisticated NLP
        current_words = set(current_input.lower().split())
        
        similarities = []
        for turn in self.conversation_history:
            turn_words = set(turn.user_input.lower().split())
            similarity = len(current_words.intersection(turn_words)) / len(current_words.union(turn_words))
            similarities.append((similarity, turn))
        
        # Sort by similarity and return top matches
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [turn for _, turn in similarities[:limit] if _ > 0.1]  # Only return if similarity > 10%
    
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
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current session.
        
        Returns:
            Dictionary with session statistics
        """
        if not self.conversation_history:
            return {
                "total_turns": 0,
                "unique_intents": 0,
                "tools_used": 0,
                "session_duration": 0,
                "average_response_time": 0.0
            }
        
        # Calculate session duration
        first_turn = self.conversation_history[0]
        last_turn = self.conversation_history[-1]
        session_duration = last_turn.timestamp - first_turn.timestamp
        
        # Collect unique intents
        intents_used = set(turn.intent for turn in self.conversation_history if turn.intent)
        
        # Collect unique tools used
        all_tools = set()
        for turn in self.conversation_history:
            all_tools.update(turn.tools_used)
        
        # Calculate average response time
        response_times = [turn.processing_time for turn in self.conversation_history if hasattr(turn, 'processing_time') and turn.processing_time > 0]
        average_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        
        return {
            "total_turns": len(self.conversation_history),
            "unique_intents": len(intents_used),
            "tools_used": len(all_tools),
            "session_duration": session_duration,
            "average_response_time": average_response_time
        }
    
    def export_conversation(self, format: str = "json") -> str:
        """
        Export conversation history in specified format.
        
        Args:
            format: Export format ("json" or "text")
            
        Returns:
            Exported conversation data
        """
        if format == "json":
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
                        "metadata": turn.metadata
                    }
                    for turn in self.conversation_history
                ]
            }
            return json.dumps(export_data, indent=2)
        
        elif format == "text":
            lines = []
            lines.append(f"Conversation Export - Session: {self.session_metadata['session_id']}")
            lines.append(f"Started: {self.session_start.isoformat()}")
            lines.append("=" * 50)
            
            for i, turn in enumerate(self.conversation_history, 1):
                lines.append(f"\nTurn {i} - {datetime.fromtimestamp(turn.timestamp).strftime('%H:%M:%S')}")
                lines.append(f"User: {turn.user_input}")
                lines.append(f"Jarvis: {turn.agent_response}")
                if turn.intent:
                    lines.append(f"Intent: {turn.intent}")
                if turn.tools_used:
                    lines.append(f"Tools: {', '.join(turn.tools_used)}")
                lines.append("-" * 30)
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def reset_session(self) -> None:
        """Reset the conversation context for a new session."""
        old_session_id = self.session_metadata.get("session_id", "unknown")
        
        self.conversation_history.clear()
        self.current_context.clear()
        self.session_start = datetime.now()
        self.session_metadata = {
            "session_id": self._generate_session_id(),
            "start_time": self.session_start.isoformat(),
            "turn_count": 0
        }
        
        self.ark_logger.info(f"ARK context: Reset session from {old_session_id} to {self.session_metadata['session_id']}")