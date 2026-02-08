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

from src.core.context.manager import ConversationTurn, ConversationContext


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
    
    def test_export_conversation_text_format(self):
        """Test export_conversation with text format."""
        context = ConversationContext()
        context.add_exchange("Hello", "Hi there!")
        context.add_exchange("How are you?", "I'm doing well")
        
        # Export to text format
        content = context.export_conversation(export_format='text')
        
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
        content = context.export_conversation(export_format='json')
        
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
            context.export_conversation(export_format='xml')
    
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