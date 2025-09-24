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
        
        original_session_id = context.session_id
        
        # Reset context
        context.reset()
        
        assert len(context.conversation_history) == 0
        assert len(context.user_memory) == 0
        # Session metadata should be cleared by reset
        assert len(context.session_metadata) == 0
        # Session ID should remain the same
        assert context.session_id == original_session_id
    
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
        
        # Test __str__ method
        str_repr = str(context)
        assert "ConversationContext" in str_repr
        assert "test_session" in str_repr
        assert "turns=1" in str_repr
        
        # Test __repr__ method
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
    
    def test_user_memory_setter(self, context):
        """Test user_memory setter functionality."""
        # Test setting user memory via the setter
        new_memory = {"preference": "dark_mode", "language": "en"}
        context.user_memory = new_memory
        
        # Verify the setter worked
        assert context.user_memory == new_memory
        
        # Test that changes persist
        context.user_memory = {"updated": "value"}
        assert context.user_memory == {"updated": "value"}
    
    def test_history_limit_removal_add_exchange(self):
        """Test history limit removal in add_exchange method."""
        # Create context with small history limit
        context = ConversationContext(max_history=2)
        
        # Add exchanges up to the limit
        context.add_exchange("First", "Response 1")
        context.add_exchange("Second", "Response 2")
        assert len(context.conversation_history) == 2
        
        # Add one more to trigger removal
        with patch.object(context.ark_logger, 'debug') as mock_debug:
            context.add_exchange("Third", "Response 3")
            
            # Verify old turn was removed
            assert len(context.conversation_history) == 2
            assert context.conversation_history[0].user_input == "Second"
            assert context.conversation_history[1].user_input == "Third"
            
            # Verify debug log was called for removal
            mock_debug.assert_called()
            # Check that the removal log was called (should contain "Removed old turn from")
            debug_calls = [str(call) for call in mock_debug.call_args_list]
            assert any("Removed old turn from" in call for call in debug_calls)
    
    def test_get_context_summary_empty(self):
        """Test get_context_summary with empty history."""
        context = ConversationContext()
        summary = context.get_context_summary()
        assert summary == "No conversation history available."
    
    def test_get_context_summary_with_history(self):
        """Test get_context_summary with conversation history."""
        context = ConversationContext()
        
        # Add some exchanges with different features
        context.add_exchange("Hello", "Hi there!", intent="greeting", tools_used=["greeting_tool"])
        context.add_exchange("What's the weather?", "It's sunny today", intent="weather", tools_used=["weather_api"])
        context.add_exchange("Tell me a joke", "Why did the chicken cross the road?")
        context.add_exchange("Another question", "Another response", intent="general")
        
        summary = context.get_context_summary()
        
        # Verify summary contains expected elements
        assert f"Session: {context.session_metadata['session_id']}" in summary
        assert "Turn count: 4" in summary
        assert "Recent conversation:" in summary
        
        # Verify recent turns are included (should show last 3)
        assert "Tell me a joke" in summary
        assert "Another question" in summary
        assert "What's the weather?" in summary
        
        # Verify intent and tools are shown
        assert "Intent: general" in summary
        assert "Tools: weather_api" in summary
        
        # Verify user input is truncated at 50 chars
        long_input = "This is a very long user input that should be truncated at fifty characters"
        context.add_exchange(long_input, "Response")
        summary = context.get_context_summary()
        assert long_input[:50] + "..." in summary
    
    def test_find_similar_exchanges_empty_history(self):
        """Test find_similar_exchanges with empty history."""
        context = ConversationContext()
        similar = context.find_similar_exchanges("test input")
        assert similar == []
    
    def test_find_similar_exchanges_with_matches(self):
        """Test find_similar_exchanges with matching exchanges."""
        context = ConversationContext()
        
        # Add some exchanges
        context.add_exchange("weather forecast today", "It's sunny")
        context.add_exchange("what is the weather like", "It's cloudy")
        context.add_exchange("tell me a joke", "Why did the chicken cross the road?")
        context.add_exchange("weather conditions tomorrow", "It will rain")
        
        # Find similar exchanges for weather-related query
        similar = context.find_similar_exchanges("weather today forecast", limit=2)
        
        # Should find weather-related exchanges
        assert len(similar) <= 2
        for turn in similar:
            assert isinstance(turn, ConversationTurn)
            # Should contain weather-related content
            assert "weather" in turn.user_input.lower()
    
    def test_find_similar_exchanges_similarity_threshold(self):
        """Test find_similar_exchanges similarity threshold."""
        context = ConversationContext()
        
        # Add exchanges with different similarity levels
        context.add_exchange("completely different topic", "Response 1")
        context.add_exchange("weather forecast", "Response 2")
        
        # Query with low similarity should return empty list
        similar = context.find_similar_exchanges("unrelated query with no common words")
        
        # Should filter out low similarity matches (< 10%)
        assert len(similar) == 0 or all(
            len(set("unrelated query with no common words".lower().split()).intersection(
                set(turn.user_input.lower().split())
            )) / len(set("unrelated query with no common words".lower().split()).union(
                set(turn.user_input.lower().split())
            )) > 0.1 for turn in similar
         )
    
    def test_export_conversation_text_format(self):
        """Test export_conversation with text format."""
        context = ConversationContext()
        context.add_exchange("Hello", "Hi there!")
        context.add_exchange("How are you?", "I'm doing well")
        
        # Export to text format
        content = context.export_conversation(format='text')
        
        # Verify text format structure
        assert "Conversation Export" in content
        assert f"Session: {context.session_metadata['session_id']}" in content
        assert "Turn 1" in content
        assert "User: Hello" in content
        assert "Jarvis: Hi there!" in content
        assert "Turn 2" in content
        assert "User: How are you?" in content
        assert "Jarvis: I'm doing well" in content
    
    def test_export_conversation_json_format(self):
        """Test export_conversation with json format."""
        context = ConversationContext()
        context.add_exchange("Hello", "Hi there!", intent="greeting", tools_used=["greeting_tool"])
        
        # Export to json format
        content = context.export_conversation(format='json')
        
        # Parse JSON to verify structure
        import json
        data = json.loads(content)
        
        assert "session_metadata" in data
        assert "conversation_history" in data
        assert len(data["conversation_history"]) == 1
        
        turn = data["conversation_history"][0]
        assert turn["user_input"] == "Hello"
        assert turn["agent_response"] == "Hi there!"
        assert turn["intent"] == "greeting"
        assert turn["tools_used"] == ["greeting_tool"]
    
    def test_export_conversation_invalid_format(self):
        """Test export_conversation with invalid format."""
        context = ConversationContext()
        context.add_exchange("Hello", "Hi there!")
        
        # Test invalid format raises ValueError
        with pytest.raises(ValueError, match="Unsupported export format"):
            context.export_conversation(format='xml')
    
    def test_string_representations(self):
        """Test __str__ and __repr__ methods."""
        context = ConversationContext()
        context.add_exchange("Hello", "Hi there")
        
        # Test __str__
        str_repr = str(context)
        assert "ConversationContext" in str_repr
        assert context.session_id in str_repr
        assert "turns=1" in str_repr
        
        # Test __repr__
        repr_str = repr(context)
        assert "ConversationContext" in repr_str
        
    def test_get_recent_turns_empty_history(self):
        """Test get_recent_turns with empty history."""
        context = ConversationContext()
        
        # Test empty history
        recent_turns = context.get_recent_turns(5)
        assert recent_turns == []
        
    def test_get_conversation_summary_single_turn(self):
        """Test get_conversation_summary with single turn for duration calculation."""
        context = ConversationContext()
        context.add_exchange("Hello", "Hi there")
        
        # Test single turn (duration should be 0)
        summary = context.get_conversation_summary()
        assert summary["duration"] == 0
        assert summary["total_turns"] == 1
        
    def test_user_preferences(self):
        """Test user preference management."""
        context = ConversationContext()
        
        # Test setting and getting preferences
        context.update_user_preference("theme", "dark")
        assert context.get_user_preference("theme") == "dark"
        
        # Test default value
        assert context.get_user_preference("nonexistent", "default") == "default"
        
    def test_context_variables(self):
        """Test context variable management."""
        context = ConversationContext()
        
        # Test setting and getting context variables
        context.set_context_variable("current_task", "testing")
        assert context.get_context_variable("current_task") == "testing"
        
        # Test default value
        assert context.get_context_variable("nonexistent", "default") == "default"
        
        # Test clearing context variables
        context.clear_context_variables()
        assert context.get_context_variable("current_task", "default") == "default"
        
    def test_get_session_stats_empty_history(self):
        """Test get_session_stats with empty conversation history."""
        context = ConversationContext()
        stats = context.get_session_stats()
        assert stats["total_turns"] == 0
        assert stats["unique_intents"] == 0
        assert stats["tools_used"] == 0
        assert stats["session_duration"] == 0
        assert stats["average_response_time"] == 0.0
        
    def test_history_limit_removal_add_turn(self):
        """Test history limit removal in add_turn method."""
        # Create context with small history limit
        context = ConversationContext(max_history=2)
        
        # Create turns manually
        turn1 = ConversationTurn("First", "Response 1")
        turn2 = ConversationTurn("Second", "Response 2")
        turn3 = ConversationTurn("Third", "Response 3")
        
        # Add turns up to the limit
        context.add_turn(turn1)
        context.add_turn(turn2)
        assert len(context.conversation_history) == 2
        
        # Add one more to trigger removal
        with patch.object(context.ark_logger, 'debug') as mock_debug:
            context.add_turn(turn3)
            
            # Verify old turn was removed
            assert len(context.conversation_history) == 2
            assert context.conversation_history[0].user_input == "Second"
            assert context.conversation_history[1].user_input == "Third"
            
            # Verify debug log was called for removal
            mock_debug.assert_called()
            # Check that the removal log was called (should contain "Removed old turn from")
            debug_calls = [str(call) for call in mock_debug.call_args_list]
            assert any("Removed old turn from" in call for call in debug_calls)
    
    def test_get_conversation_summary_empty_with_session_id(self):
        """Test get_conversation_summary with empty history accessing session_id."""
        context = ConversationContext()
        # Ensure session_metadata has session_id
        context.session_metadata['session_id'] = 'test_session'
        
        summary = context.get_conversation_summary()
        assert summary["total_turns"] == 0
        assert summary["session_id"] == 'test_session'
        assert summary["duration"] == 0  # This should trigger line 439

    def test_get_conversation_summary_empty_history_duration(self):
        """Test get_conversation_summary with completely empty history for duration calculation."""
        context = ConversationContext()
        # Ensure conversation_history is empty
        context.conversation_history.clear()
        
        # Verify the list is empty (falsy)
        assert not context.conversation_history
        assert len(context.conversation_history) == 0
        
        summary = context.get_conversation_summary()
        assert summary["total_turns"] == 0
        assert summary["duration"] == 0  # This should trigger the else branch on line 439
        assert summary["intents"] == []
    
    def test_get_conversation_summary_single_turn_duration(self):
        """Test get_conversation_summary with single turn for duration calculation."""
        context = ConversationContext()
        context.session_metadata['session_id'] = 'test_session'
        
        # Add a single turn
        turn = ConversationTurn("Hello", "Hi", intent="greeting")
        context.add_turn(turn)
        
        summary = context.get_conversation_summary()
        assert summary["total_turns"] == 1
        assert summary["duration"] == 0  # Single turn has 0 duration
        assert "greeting" in summary["intents"]

    def test_get_conversation_summary_multiple_turns_duration(self):
        """Test get_conversation_summary with multiple turns for duration calculation."""
        context = ConversationContext()
        context.session_metadata['session_id'] = 'test_session'
        
        # Add multiple turns with different timestamps
        import time
        turn1 = ConversationTurn("Hello", "Hi", intent="greeting")
        turn1.timestamp = 1000.0
        context.add_turn(turn1)
        
        turn2 = ConversationTurn("How are you?", "I'm fine", intent="status")
        turn2.timestamp = 1005.0  # 5 seconds later
        context.add_turn(turn2)
        
        summary = context.get_conversation_summary()
        assert summary["total_turns"] == 2
        assert summary["duration"] == 5.0  # 5 seconds duration
        assert "greeting" in summary["intents"]
        assert "status" in summary["intents"]
    
    def test_export_conversation_text_with_intent_and_tools(self):
        """Test export_conversation text format with intent and tools."""
        context = ConversationContext()
        
        # Add turn with intent and tools
        turn = ConversationTurn(
            "Search for weather", 
            "Here's the weather", 
            intent="weather_query",
            tools_used=["weather_api", "location_service"]
        )
        context.add_turn(turn)
        
        result = context.export_conversation("text")
        assert "Intent: weather_query" in result
        assert "Tools: weather_api, location_service" in result
        assert "Search for weather" in result
        assert "Here's the weather" in result
    
    def test_reset_session(self):
        """Test reset_session method."""
        context = ConversationContext()
        original_session_id = context.session_metadata['session_id']
        
        # Add some data
        context.add_exchange("Hello", "Hi")
        context.current_context["key"] = "value"
        
        with patch.object(context.ark_logger, 'info') as mock_info:
            context.reset_session()
            
            # Check that data is cleared and new session is created
            assert len(context.conversation_history) == 0
            assert len(context.current_context) == 0
            assert context.session_metadata['session_id'] != original_session_id
            assert "session_id" in context.session_metadata
            assert "start_time" in context.session_metadata
            assert "turn_count" in context.session_metadata
            
            # Check info log was called
            mock_info.assert_called_once()
            log_message = mock_info.call_args[0][0]
            assert "Reset session from" in log_message
            assert original_session_id in log_message
    
    def test_get_conversation_summary_truly_empty(self):
        """Test get_conversation_summary with a freshly created context to ensure else branch is hit."""
        context = ConversationContext()
        
        # Verify that conversation_history is truly empty
        assert context.conversation_history == []
        assert len(context.conversation_history) == 0
        assert not context.conversation_history
        
        # Call get_conversation_summary which should hit the else branch on line 439
        summary = context.get_conversation_summary()
        
        # Verify the results
        assert summary["total_turns"] == 0
        assert summary["duration"] == 0
        assert summary["intents"] == []
        assert "session_id" in summary

    def test_get_context_summary_mixed_intent_tools(self):
        """Test get_context_summary with mixed intent and tools scenarios."""
        context = ConversationContext()
        
        # Add exchanges with different combinations of intent and tools
        context.add_exchange(
            "What's the weather?", 
            "It's sunny",
            intent="weather_query",
            tools_used=["weather_api"]
        )
        context.add_exchange(
            "Tell me a joke", 
            "Why did the chicken cross the road?",
            intent="entertainment"
            # No tools_used
        )
        context.add_exchange(
            "Random question", 
            "Random answer"
            # No intent, no tools_used
        )
        
        summary = context.get_context_summary()
        
        # Check that the summary handles mixed scenarios correctly
        assert "Session:" in summary
        assert "Turn count: 3" in summary
        assert "Recent conversation:" in summary
        
        # Should contain intent and tools for first exchange
        assert "Intent: weather_query" in summary
        assert "Tools: weather_api" in summary
        
        # Should contain intent but no tools for second exchange
        assert "Intent: entertainment" in summary
        
        # Third exchange should have neither intent nor tools lines
        assert "User: Random question" in summary

    def test_get_context_summary_empty_tools_list(self):
        """Test get_context_summary with empty tools list."""
        context = ConversationContext()
        
        # Add exchange with empty tools list
        context.add_exchange(
            "Hello", 
            "Hi there",
            intent="greeting",
            tools_used=[]
        )
        
        summary = context.get_context_summary()
        
        # Should contain intent but not tools (empty list)
        assert "Intent: greeting" in summary
        assert "Tools:" not in summary

    def test_get_context_summary_empty_history(self):
        """Test get_context_summary with empty conversation history to cover the missing branch."""
        context = ConversationContext()
        
        # Ensure conversation history is empty
        assert len(context.conversation_history) == 0
        
        # Call get_context_summary
        summary = context.get_context_summary()
        
        # Should return the empty history message
        assert summary == "No conversation history available."

    def test_get_context_summary_no_intent_no_tools(self):
        """Test get_context_summary with turns that have no intent or tools."""
        context = ConversationContext()
        
        # Add exchanges without intent and tools
        context.add_exchange("Hello", "Hi there!")
        context.add_exchange("How are you?", "I'm doing well")
        
        summary = context.get_context_summary()
        
        # Check that the summary contains expected elements but no intent/tools lines
        assert "Session:" in summary
        assert "Turn count:" in summary
        assert "Recent conversation:" in summary
        assert "User: Hello" in summary
        assert "User: How are you?" in summary
        # Should not contain intent or tools lines
        assert "Intent:" not in summary
        assert "Tools:" not in summary

    def test_get_context_summary_only_intent(self):
        """Test get_context_summary with turns that have intent but no tools."""
        context = ConversationContext()
        
        # Add exchanges with intent but no tools
        context.add_exchange(
            "What's the weather?", 
            "It's sunny",
            intent="weather_query"
        )
        
        summary = context.get_context_summary()
        
        # Check that the summary contains intent but no tools
        assert "Session:" in summary
        assert "Turn count: 1" in summary
        assert "Recent conversation:" in summary
        assert "User: What's the weather?" in summary
        assert "Intent: weather_query" in summary
        assert "Tools:" not in summary

    def test_get_context_summary_only_tools(self):
        """Test get_context_summary with turns that have tools but no intent."""
        context = ConversationContext()
        
        # Add exchanges with tools but no intent
        context.add_exchange(
            "Search for something", 
            "Here are the results",
            tools_used=["search_api"]
        )
        
        summary = context.get_context_summary()
        
        # Check that the summary contains tools but no intent
        assert "Session:" in summary
        assert "Turn count: 1" in summary
        assert "Recent conversation:" in summary
        assert "User: Search for something" in summary
        assert "Tools: search_api" in summary
        assert "Intent:" not in summary
    
    def test_get_context_summary_no_recent_turns(self):
        """Test get_context_summary when get_recent_context returns empty list."""
        context = ConversationContext()
        
        # Add some conversation history
        context.add_exchange("Hello", "Hi there!")
        
        # Mock get_recent_context to return empty list
        original_method = context.get_recent_context
        context.get_recent_context = lambda num_turns: []
        
        try:
            summary = context.get_context_summary()
            
            # Should contain session info and turn count but no recent conversation
            assert "Session:" in summary
            assert "Turn count:" in summary
            assert "Recent conversation:" not in summary
        finally:
            # Restore original method
            context.get_recent_context = original_method

    def test_get_conversation_summary_cleared_history(self):
        """Test get_conversation_summary after clearing history to force the else branch."""
        context = ConversationContext()
        
        # Add some data first
        context.add_exchange("Hello", "Hi there")
        assert len(context.conversation_history) == 1
        
        # Now clear the history
        context.conversation_history.clear()
        assert len(context.conversation_history) == 0
        assert not context.conversation_history
        
        # Call get_conversation_summary which should hit the else branch on line 439
        summary = context.get_conversation_summary()
        
        # Verify the results
        assert summary["total_turns"] == 0
        assert summary["duration"] == 0
        assert summary["intents"] == []
        assert "session_id" in summary

    def test_get_context_summary_with_intent_and_tools(self):
        """Test get_context_summary with recent turns that have intent and tools."""
        context = ConversationContext()
        
        # Add exchanges with intent and tools
        context.add_exchange(
            "What's the weather like?", 
            "Let me check that for you.",
            intent="weather_query",
            tools_used=["weather_api", "location_service"]
        )
        context.add_exchange(
            "Set a reminder for tomorrow", 
            "I'll set that reminder.",
            intent="reminder_creation",
            tools_used=["calendar_api"]
        )
        
        summary = context.get_context_summary()
        
        # Check that the summary contains expected elements
        assert "Session:" in summary
        assert "Turn count:" in summary
        assert "Recent conversation:" in summary
        assert "User: What's the weather like?" in summary
        assert "Intent: weather_query" in summary
        assert "Tools: weather_api, location_service" in summary
        assert "User: Set a reminder for tomorrow" in summary
        assert "Intent: reminder_creation" in summary
        assert "Tools: calendar_api" in summary


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