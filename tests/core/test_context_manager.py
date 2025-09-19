"""
Tests for the Context Manager module.

This module contains comprehensive tests for the conversation context
management system, including memory handling, session tracking, and
context persistence.
"""

import pytest
import json
import tempfile
import os
from datetime import datetime
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from src.core.context_manager import ConversationTurn, ConversationContext


class TestConversationTurn:
    """Test cases for ConversationTurn class."""
    
    def test_turn_creation(self):
        """Test basic conversation turn creation."""
        turn = ConversationTurn(
            user_input="Hello, how are you?",
            agent_response="I'm doing well, thank you!",
            intent="greeting",
            entities={"greeting_type": "casual"},
            tools_used=["greeting_tool"],
            metadata={"confidence": 0.95}
        )
        
        assert turn.user_input == "Hello, how are you?"
        assert turn.agent_response == "I'm doing well, thank you!"
        assert turn.intent == "greeting"
        assert turn.entities["greeting_type"] == "casual"
        assert "greeting_tool" in turn.tools_used
        assert turn.metadata["confidence"] == 0.95
        assert isinstance(turn.timestamp, float)
    
    def test_turn_to_dict(self):
        """Test conversation turn serialization."""
        turn = ConversationTurn(
            user_input="Test input",
            agent_response="Test response",
            intent="test_intent"
        )
        
        turn_dict = turn.to_dict()
        
        assert turn_dict["user_input"] == "Test input"
        assert turn_dict["agent_response"] == "Test response"
        assert turn_dict["intent"] == "test_intent"
        assert "timestamp" in turn_dict
        assert isinstance(turn_dict["timestamp"], float)
    
    def test_turn_from_dict(self):
        """Test conversation turn deserialization."""
        turn_data = {
            "user_input": "Test input",
            "agent_response": "Test response",
            "intent": "test_intent",
            "entities": {"entity1": "value1"},
            "tools_used": ["tool1", "tool2"],
            "metadata": {"key": "value"},
            "timestamp": 1234567890.0
        }
        
        turn = ConversationTurn.from_dict(turn_data)
        
        assert turn.user_input == "Test input"
        assert turn.agent_response == "Test response"
        assert turn.intent == "test_intent"
        assert turn.entities["entity1"] == "value1"
        assert turn.tools_used == ["tool1", "tool2"]
        assert turn.metadata["key"] == "value"
        assert turn.timestamp == 1234567890.0
    
    def test_turn_duration_calculation(self):
        """Test conversation turn duration calculation."""
        turn = ConversationTurn(
            user_input="Test",
            agent_response="Response"
        )
        
        # Mock processing time
        turn.processing_time = 1.5
        
        assert turn.processing_time == 1.5


class TestConversationContext:
    """Test cases for ConversationContext class."""
    
    @pytest.fixture
    def context(self):
        """Create a conversation context for testing."""
        return ConversationContext(session_id="test_session")
    
    def test_context_initialization(self, context):
        """Test conversation context initialization."""
        assert context.session_id == "test_session"
        assert context.conversation_history == []
        assert context.user_memory == {}
        assert "session_id" in context.session_metadata
        assert "start_time" in context.session_metadata
        assert "turn_count" in context.session_metadata
        assert context.session_metadata["session_id"] == "test_session"
        assert context.session_metadata["turn_count"] == 0
        assert isinstance(context.created_at, float)
        assert isinstance(context.last_activity, float)
    
    def test_add_turn(self, context):
        """Test adding a conversation turn."""
        turn = ConversationTurn(
            user_input="Hello",
            agent_response="Hi there!"
        )
        
        context.add_turn(turn)
        
        assert len(context.conversation_history) == 1
        assert context.conversation_history[0] == turn
        assert context.last_activity > context.created_at
    
    def test_get_recent_turns(self, context):
        """Test getting recent conversation turns."""
        # Add multiple turns
        for i in range(5):
            turn = ConversationTurn(
                user_input=f"Input {i}",
                agent_response=f"Response {i}"
            )
            context.add_turn(turn)
        
        # Get recent turns
        recent_turns = context.get_recent_turns(3)
        
        assert len(recent_turns) == 3
        assert recent_turns[0].user_input == "Input 4"  # Most recent first
        assert recent_turns[1].user_input == "Input 3"
        assert recent_turns[2].user_input == "Input 2"
    
    def test_get_recent_turns_insufficient(self, context):
        """Test getting recent turns when there are fewer than requested."""
        # Add only 2 turns
        for i in range(2):
            turn = ConversationTurn(
                user_input=f"Input {i}",
                agent_response=f"Response {i}"
            )
            context.add_turn(turn)
        
        # Request 5 recent turns
        recent_turns = context.get_recent_turns(5)
        
        assert len(recent_turns) == 2
    
    def test_update_memory(self, context):
        """Test updating user memory."""
        context.update_memory("user_name", "Alice")
        context.update_memory("preferences", {"theme": "dark", "language": "en"})
        
        assert context.user_memory["user_name"] == "Alice"
        assert context.user_memory["preferences"]["theme"] == "dark"
        assert context.user_memory["preferences"]["language"] == "en"
    
    def test_get_memory(self, context):
        """Test retrieving user memory."""
        # Set some memory
        context.user_memory["user_name"] = "Bob"
        context.user_memory["age"] = 30
        
        # Get existing memory
        assert context.get_memory("user_name") == "Bob"
        assert context.get_memory("age") == 30
        
        # Get non-existent memory
        assert context.get_memory("non_existent") is None
        
        # Get with default value
        assert context.get_memory("non_existent", "default") == "default"
    
    def test_clear_memory(self, context):
        """Test clearing user memory."""
        # Set some memory
        context.update_memory("key1", "value1")
        context.update_memory("key2", "value2")
        
        assert len(context.user_memory) == 2
        
        # Clear memory
        context.clear_memory()
        
        assert len(context.user_memory) == 0
    
    def test_get_conversation_summary(self, context):
        """Test getting conversation summary."""
        # Add some turns
        turns = [
            ConversationTurn("Hello", "Hi!", intent="greeting"),
            ConversationTurn("What's the weather?", "It's sunny", intent="weather_query"),
            ConversationTurn("Thanks", "You're welcome", intent="gratitude")
        ]
        
        for turn in turns:
            context.add_turn(turn)
        
        summary = context.get_conversation_summary()
        
        assert summary["total_turns"] == 3
        assert summary["session_id"] == "test_session"
        assert "greeting" in summary["intents"]
        assert "weather_query" in summary["intents"]
        assert "gratitude" in summary["intents"]
        assert summary["duration"] > 0
    
    def test_get_session_stats(self, context):
        """Test getting session statistics."""
        # Add turns with different intents and tools
        turns = [
            ConversationTurn("Hello", "Hi!", intent="greeting", tools_used=["greeting_tool"]),
            ConversationTurn("Weather?", "Sunny", intent="weather", tools_used=["weather_tool"]),
            ConversationTurn("Thanks", "Welcome", intent="gratitude", tools_used=[])
        ]
        
        for turn in turns:
            context.add_turn(turn)
        
        stats = context.get_session_stats()
        
        assert stats["total_turns"] == 3
        assert stats["unique_intents"] == 3
        assert stats["tools_used"] == 2  # greeting_tool and weather_tool
        assert stats["session_duration"] > 0
        assert isinstance(stats["average_response_time"], float)
    
    def test_export_context(self, context):
        """Test exporting conversation context."""
        # Add some data
        context.update_memory("user_name", "Charlie")
        context.session_metadata["test_key"] = "test_value"
        
        turn = ConversationTurn("Test input", "Test response", intent="test_intent")
        context.add_turn(turn)
        
        # Export context
        exported = context.export_context()
        
        assert exported["session_id"] == "test_session"
        assert exported["user_memory"]["user_name"] == "Charlie"
        assert exported["session_metadata"]["test_key"] == "test_value"
        assert len(exported["conversation_history"]) == 1
        assert exported["conversation_history"][0]["user_input"] == "Test input"
    
    def test_import_context(self):
        """Test importing conversation context."""
        context_data = {
            "session_id": "imported_session",
            "conversation_history": [
                {
                    "user_input": "Imported input",
                    "agent_response": "Imported response",
                    "intent": "imported_intent",
                    "entities": {},
                    "tools_used": [],
                    "metadata": {},
                    "timestamp": 1234567890.0,
                    "processing_time": 0.5
                }
            ],
            "user_memory": {"imported_key": "imported_value"},
            "session_metadata": {"imported_meta": "meta_value"},
            "created_at": 1234567800.0,
            "last_activity": 1234567900.0
        }
        
        context = ConversationContext.import_context(context_data)
        
        assert context.session_id == "imported_session"
        assert len(context.conversation_history) == 1
        assert context.conversation_history[0].user_input == "Imported input"
        assert context.user_memory["imported_key"] == "imported_value"
        assert context.session_metadata["imported_meta"] == "meta_value"
        assert context.created_at == 1234567800.0
        assert context.last_activity == 1234567900.0
    
    def test_reset_context(self, context):
        """Test resetting conversation context."""
        # Add some data
        context.update_memory("key", "value")
        context.session_metadata["meta"] = "data"
        context.add_turn(ConversationTurn("input", "response"))
        
        # Reset context
        context.reset()
        
        assert len(context.conversation_history) == 0
        assert len(context.user_memory) == 0
        assert len(context.session_metadata) == 0
        # Session ID should remain the same
        assert context.session_id == "test_session"
    
    def test_context_persistence(self, context):
        """Test context persistence to file."""
        # Add some data
        context.update_memory("persistent_key", "persistent_value")
        context.add_turn(ConversationTurn("persistent input", "persistent response"))
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name
        
        try:
            # Save context
            context.save_to_file(temp_file)
            
            # Load context
            loaded_context = ConversationContext.load_from_file(temp_file)
            
            assert loaded_context.session_id == context.session_id
            assert loaded_context.user_memory["persistent_key"] == "persistent_value"
            assert len(loaded_context.conversation_history) == 1
            assert loaded_context.conversation_history[0].user_input == "persistent input"
            
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_context_file_not_found(self):
        """Test loading context from non-existent file."""
        with pytest.raises(FileNotFoundError):
            ConversationContext.load_from_file("non_existent_file.json")
    
    def test_context_invalid_json(self):
        """Test loading context from invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            f.write("invalid json content")
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError):
                ConversationContext.load_from_file(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_context_string_representation(self, context):
        """Test string representation of context."""
        context.add_turn(ConversationTurn("test", "response"))
        
        repr_str = repr(context)
        assert "ConversationContext" in repr_str
        assert "test_session" in repr_str
        assert "turns=1" in repr_str
    
    def test_context_length(self, context):
        """Test context length calculation."""
        assert len(context) == 0
        
        context.add_turn(ConversationTurn("input1", "response1"))
        assert len(context) == 1
        
        context.add_turn(ConversationTurn("input2", "response2"))
        assert len(context) == 2
    
    def test_context_iteration(self, context):
        """Test iterating over context turns."""
        turns = [
            ConversationTurn("input1", "response1"),
            ConversationTurn("input2", "response2"),
            ConversationTurn("input3", "response3")
        ]
        
        for turn in turns:
            context.add_turn(turn)
        
        # Test iteration
        iterated_turns = list(context)
        assert len(iterated_turns) == 3
        assert iterated_turns[0].user_input == "input1"
        assert iterated_turns[1].user_input == "input2"
        assert iterated_turns[2].user_input == "input3"


class TestConversationContextIntegration:
    """Integration tests for conversation context functionality."""
    
    def test_full_conversation_workflow(self):
        """Test complete conversation workflow."""
        context = ConversationContext(session_id="integration_test")
        
        # Simulate a conversation
        conversation_data = [
            ("Hello", "Hi there! How can I help you?", "greeting"),
            ("What's the weather like?", "Let me check the weather for you.", "weather_query"),
            ("Thank you", "You're welcome! Anything else?", "gratitude")
        ]
        
        for user_input, agent_response, intent in conversation_data:
            turn = ConversationTurn(
                user_input=user_input,
                agent_response=agent_response,
                intent=intent
            )
            context.add_turn(turn)
        
        # Update user memory
        context.update_memory("user_name", "Integration User")
        context.update_memory("location", "Test City")
        
        # Get summary
        summary = context.get_conversation_summary()
        assert summary["total_turns"] == 3
        assert "greeting" in summary["intents"]
        assert "weather_query" in summary["intents"]
        assert "gratitude" in summary["intents"]
        
        # Get recent turns
        recent = context.get_recent_turns(2)
        assert len(recent) == 2
        assert recent[0].intent == "gratitude"  # Most recent
        assert recent[1].intent == "weather_query"
        
        # Check memory
        assert context.get_memory("user_name") == "Integration User"
        assert context.get_memory("location") == "Test City"
    
    def test_memory_persistence_workflow(self):
        """Test memory persistence across sessions."""
        # Create first context
        context1 = ConversationContext(session_id="memory_test")
        context1.update_memory("persistent_data", "should_persist")
        context1.update_memory("user_preferences", {"theme": "dark"})
        
        # Export and import
        exported_data = context1.export_context()
        context2 = ConversationContext.import_context(exported_data)
        
        # Verify memory persisted
        assert context2.get_memory("persistent_data") == "should_persist"
        assert context2.get_memory("user_preferences")["theme"] == "dark"
        assert context2.session_id == "memory_test"
    
    def test_large_conversation_handling(self):
        """Test handling of large conversations."""
        context = ConversationContext(session_id="large_conversation")
        
        # Add many turns
        for i in range(100):
            turn = ConversationTurn(
                user_input=f"User input {i}",
                agent_response=f"Agent response {i}",
                intent=f"intent_{i % 10}"  # 10 different intents
            )
            context.add_turn(turn)
        
        # Test performance of operations
        assert len(context) == 100
        
        recent_turns = context.get_recent_turns(10)
        assert len(recent_turns) == 10
        assert recent_turns[0].user_input == "User input 99"  # Most recent
        
        summary = context.get_conversation_summary()
        assert summary["total_turns"] == 100
        assert len(summary["intents"]) == 10  # 10 unique intents
    
    def test_concurrent_context_operations(self):
        """Test concurrent operations on context."""
        import threading
        import time
        
        context = ConversationContext(session_id="concurrent_test")
        results = []
        
        def add_turns(start_idx, count):
            for i in range(start_idx, start_idx + count):
                turn = ConversationTurn(
                    user_input=f"Input {i}",
                    agent_response=f"Response {i}"
                )
                context.add_turn(turn)
                time.sleep(0.001)  # Small delay
            results.append(f"Thread {start_idx} completed")
        
        # Create multiple threads
        threads = []
        for i in range(0, 30, 10):
            thread = threading.Thread(target=add_turns, args=(i, 10))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(context) == 30
        assert len(results) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])