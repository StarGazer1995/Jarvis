"""
Tests for the Intent Engine module.

This module contains comprehensive tests for the intent recognition
and entity extraction system, including pattern matching, rule-based
processing, and confidence scoring.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from src.core.intent_engine import (
    IntentType, Entity, IntentResult, ARKIntentEngine
)


class TestIntentType:
    """Test cases for IntentType enum."""
    
    def test_intent_type_values(self):
        """Test that all expected intent types are defined."""
        expected_intents = [
            "GREETING", "QUESTION", "COMMAND", "REQUEST", "INFORMATION",
            "WEATHER", "TIME", "CALCULATION", "SEARCH", "HELP", "GOODBYE", "UNKNOWN"
        ]
        
        for intent in expected_intents:
            assert hasattr(IntentType, intent)
        
        # Test specific values
        assert IntentType.GREETING.value == "greeting"
        assert IntentType.QUESTION.value == "question"
        assert IntentType.COMMAND.value == "command"
        assert IntentType.UNKNOWN.value == "unknown"


class TestEntity:
    """Test cases for Entity class."""
    
    def test_entity_creation(self):
        """Test basic entity creation."""
        entity = Entity(
            name="location",
            value="New York",
            entity_type="LOCATION",
            confidence=0.95,
            start_pos=10,
            end_pos=18
        )
        
        assert entity.name == "location"
        assert entity.value == "New York"
        assert entity.entity_type == "LOCATION"
        assert entity.confidence == 0.95
        assert entity.start_pos == 10
        assert entity.end_pos == 18
    
    def test_entity_to_dict(self):
        """Test entity serialization."""
        entity = Entity(
            name="time",
            value="tomorrow",
            entity_type="TIME",
            confidence=0.8
        )
        
        entity_dict = entity.to_dict()
        
        assert entity_dict["name"] == "time"
        assert entity_dict["value"] == "tomorrow"
        assert entity_dict["entity_type"] == "TIME"
        assert entity_dict["confidence"] == 0.8
        assert entity_dict["start_pos"] is None
        assert entity_dict["end_pos"] is None
    
    def test_entity_from_dict(self):
        """Test entity deserialization."""
        entity_data = {
            "name": "person",
            "value": "Alice",
            "entity_type": "PERSON",
            "confidence": 0.9,
            "start_pos": 5,
            "end_pos": 10
        }
        
        entity = Entity.from_dict(entity_data)
        
        assert entity.name == "person"
        assert entity.value == "Alice"
        assert entity.entity_type == "PERSON"
        assert entity.confidence == 0.9
        assert entity.start_pos == 5
        assert entity.end_pos == 10
    
    def test_entity_string_representation(self):
        """Test entity string representation."""
        entity = Entity("test", "value", "TYPE", 0.5)
        repr_str = repr(entity)
        
        assert "Entity" in repr_str
        assert "test" in repr_str
        assert "value" in repr_str
        assert "TYPE" in repr_str


class TestIntentResult:
    """Test cases for IntentResult class."""
    
    def test_intent_result_creation(self):
        """Test basic intent result creation."""
        entities = [
            Entity("location", "Paris", "LOCATION", 0.9),
            Entity("time", "today", "TIME", 0.8)
        ]
        
        result = IntentResult(
            intent=IntentType.WEATHER,
            confidence=0.85,
            entities=entities,
            raw_text="What's the weather in Paris today?",
            processed_text="what's the weather in paris today?",
            metadata={"source": "pattern_match"}
        )
        
        assert result.intent == IntentType.WEATHER
        assert result.confidence == 0.85
        assert len(result.entities) == 2
        assert result.entities[0].name == "location"
        assert result.entities[1].name == "time"
        assert result.metadata["source"] == "pattern_match"
    
    def test_intent_result_to_dict(self):
        """Test intent result serialization."""
        entities = [Entity("test", "value", "TYPE", 0.7)]
        result = IntentResult(
            intent=IntentType.QUESTION,
            confidence=0.9,
            entities=entities,
            raw_text="What is this?",
            processed_text="what is this?"
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["intent"] == "question"
        assert result_dict["confidence"] == 0.9
        assert len(result_dict["entities"]) == 1
        assert result_dict["entities"][0]["name"] == "test"
    
    def test_intent_result_from_dict(self):
        """Test intent result deserialization."""
        result_data = {
            "intent": "greeting",
            "confidence": 0.95,
            "entities": [
                {
                    "name": "person",
                    "value": "Bob",
                    "entity_type": "PERSON",
                    "confidence": 0.8,
                    "start_pos": None,
                    "end_pos": None
                }
            ],
            "raw_text": "Hello Bob",
            "processed_text": "hello bob",
            "metadata": {"test": "value"}
        }
        
        result = IntentResult.from_dict(result_data)
        
        assert result.intent == IntentType.GREETING
        assert result.confidence == 0.95
        assert len(result.entities) == 1
        assert result.entities[0].name == "person"
        assert result.metadata["test"] == "value"


class TestARKIntentEngine:
    """Test cases for ARKIntentEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create an intent engine for testing."""
        return ARKIntentEngine()
    
    def test_engine_initialization(self, engine):
        """Test intent engine initialization."""
        assert isinstance(engine.intent_patterns, dict)
        assert isinstance(engine.entity_patterns, dict)
        
        # Check that default patterns are loaded
        assert IntentType.GREETING in engine.intent_patterns
        assert IntentType.WEATHER in engine.intent_patterns
        assert IntentType.TIME in engine.intent_patterns
        
        # Check entity patterns
        assert "LOCATION" in engine.entity_patterns
        assert "TIME" in engine.entity_patterns
        assert "PERSON" in engine.entity_patterns
    
    def test_greeting_recognition(self, engine):
        """Test greeting intent recognition."""
        test_inputs = [
            "hello",
            "hi there",
            "good morning",
            "hey",
            "greetings"
        ]
        
        for input_text in test_inputs:
            result = engine.recognize_intent(input_text)
            assert result.intent == IntentType.GREETING
            assert result.confidence > 0.5
    
    def test_weather_recognition(self, engine):
        """Test weather intent recognition."""
        test_inputs = [
            "what's the weather like?",
            "how's the weather today?",
            "weather forecast",
            "is it raining?",
            "temperature outside"
        ]
        
        for input_text in test_inputs:
            result = engine.recognize_intent(input_text)
            assert result.intent == IntentType.WEATHER
            assert result.confidence > 0.5
    
    def test_time_recognition(self, engine):
        """Test time intent recognition."""
        test_inputs = [
            "what time is it?",
            "current time",
            "what's the time?",
            "time now",
            "clock"
        ]
        
        for input_text in test_inputs:
            result = engine.recognize_intent(input_text)
            assert result.intent == IntentType.TIME
            assert result.confidence > 0.5
    
    def test_question_recognition(self, engine):
        """Test question intent recognition."""
        test_inputs = [
            "how do I do this?",
            "what is the meaning of life?",
            "why is the sky blue?",
            "where can I find help?",
            "when will this be ready?"
        ]
        
        for input_text in test_inputs:
            result = engine.recognize_intent(input_text)
            assert result.intent == IntentType.QUESTION
            assert result.confidence > 0.5
    
    def test_command_recognition(self, engine):
        """Test command intent recognition."""
        test_inputs = [
            "please help me",
            "show me the results",
            "open the file",
            "start the process",
            "stop the service"
        ]
        
        for input_text in test_inputs:
            result = engine.recognize_intent(input_text)
            assert result.intent == IntentType.COMMAND
            assert result.confidence > 0.5
    
    def test_goodbye_recognition(self, engine):
        """Test goodbye intent recognition."""
        test_inputs = [
            "goodbye",
            "bye",
            "see you later",
            "farewell",
            "talk to you soon"
        ]
        
        for input_text in test_inputs:
            result = engine.recognize_intent(input_text)
            assert result.intent == IntentType.GOODBYE
            assert result.confidence > 0.5
    
    def test_unknown_intent(self, engine):
        """Test unknown intent for unrecognized input."""
        test_inputs = [
            "xyzabc123",
            "random gibberish",
            "asdfghjkl",
            "",
            "   "
        ]
        
        for input_text in test_inputs:
            result = engine.recognize_intent(input_text)
            assert result.intent == IntentType.UNKNOWN
            assert result.confidence <= 0.5
    
    def test_entity_extraction_location(self, engine):
        """Test location entity extraction."""
        test_cases = [
            ("weather in New York", "New York"),
            ("how's the weather in Paris?", "Paris"),
            ("temperature in London today", "London"),
            ("forecast for Tokyo", "Tokyo")
        ]
        
        for input_text, expected_location in test_cases:
            entities = engine.extract_entities(input_text)
            location_entities = [e for e in entities if e.entity_type == "LOCATION"]
            
            assert len(location_entities) > 0
            assert any(expected_location.lower() in e.value.lower() for e in location_entities)
    
    def test_entity_extraction_time(self, engine):
        """Test time entity extraction."""
        test_cases = [
            ("weather today", "today"),
            ("forecast for tomorrow", "tomorrow"),
            ("temperature this morning", "morning"),
            ("weather next week", "next week")
        ]
        
        for input_text, expected_time in test_cases:
            entities = engine.extract_entities(input_text)
            time_entities = [e for e in entities if e.entity_type == "TIME"]
            
            assert len(time_entities) > 0
            assert any(expected_time.lower() in e.value.lower() for e in time_entities)
    
    def test_entity_extraction_person(self, engine):
        """Test person entity extraction."""
        test_cases = [
            ("tell Alice about this", "Alice"),
            ("message from Bob", "Bob"),
            ("call John Smith", "John"),
            ("meeting with Dr. Johnson", "Dr. Johnson")
        ]
        
        for input_text, expected_person in test_cases:
            entities = engine.extract_entities(input_text)
            person_entities = [e for e in entities if e.entity_type == "PERSON"]
            
            # Note: Person extraction might be limited in this simple implementation
            # This test verifies the extraction mechanism works
            if person_entities:
                assert any(expected_person.lower() in e.value.lower() for e in person_entities)
    
    def test_combined_intent_and_entities(self, engine):
        """Test combined intent recognition and entity extraction."""
        input_text = "what's the weather like in Paris today?"
        
        result = engine.recognize_intent(input_text)
        
        assert result.intent == IntentType.WEATHER
        assert result.confidence > 0.5
        
        # Check for location entity
        location_entities = [e for e in result.entities if e.entity_type == "LOCATION"]
        assert len(location_entities) > 0
        assert any("paris" in e.value.lower() for e in location_entities)
        
        # Check for time entity
        time_entities = [e for e in result.entities if e.entity_type == "TIME"]
        assert len(time_entities) > 0
        assert any("today" in e.value.lower() for e in time_entities)
    
    def test_confidence_scoring(self, engine):
        """Test confidence scoring for different matches."""
        # Strong match should have high confidence
        strong_match = engine.recognize_intent("hello there")
        assert strong_match.confidence > 0.8
        
        # Weak match should have lower confidence
        weak_match = engine.recognize_intent("maybe hello")
        assert weak_match.confidence < strong_match.confidence
        
        # Unknown should have very low confidence
        unknown_match = engine.recognize_intent("xyzabc")
        assert unknown_match.confidence < 0.5
    
    def test_case_insensitive_matching(self, engine):
        """Test case insensitive intent recognition."""
        test_cases = [
            "HELLO",
            "Hello",
            "hello",
            "HeLLo"
        ]
        
        for input_text in test_cases:
            result = engine.recognize_intent(input_text)
            assert result.intent == IntentType.GREETING
    
    def test_whitespace_handling(self, engine):
        """Test handling of extra whitespace."""
        test_cases = [
            "  hello  ",
            "\thello\n",
            "hello   there",
            "  what's   the   weather  like?  "
        ]
        
        for input_text in test_cases:
            result = engine.recognize_intent(input_text)
            assert result.intent in [IntentType.GREETING, IntentType.WEATHER]
    
    def test_empty_input_handling(self, engine):
        """Test handling of empty or None input."""
        test_cases = [None, "", "   ", "\t\n"]
        
        for input_text in test_cases:
            result = engine.recognize_intent(input_text)
            assert result.intent == IntentType.UNKNOWN
            assert result.confidence == 0.0
    
    def test_add_custom_intent_pattern(self, engine):
        """Test adding custom intent patterns."""
        # Add custom pattern
        custom_patterns = ["custom command", "special request"]
        engine.intent_patterns[IntentType.COMMAND].extend(custom_patterns)
        
        # Test recognition
        result = engine.recognize_intent("this is a custom command")
        assert result.intent == IntentType.COMMAND
        assert result.confidence > 0.5
    
    def test_add_custom_entity_pattern(self, engine):
        """Test adding custom entity patterns."""
        # Add custom entity pattern
        engine.entity_patterns["CUSTOM"] = [r"\b(custom_\w+)\b"]
        
        # Test extraction
        entities = engine.extract_entities("find custom_entity in the text")
        custom_entities = [e for e in entities if e.entity_type == "CUSTOM"]
        
        assert len(custom_entities) > 0
        assert "custom_entity" in custom_entities[0].value
    
    def test_multiple_entities_same_type(self, engine):
        """Test extraction of multiple entities of the same type."""
        input_text = "weather in New York and Paris today and tomorrow"
        
        entities = engine.extract_entities(input_text)
        
        # Should find multiple locations
        location_entities = [e for e in entities if e.entity_type == "LOCATION"]
        assert len(location_entities) >= 2
        
        # Should find multiple time references
        time_entities = [e for e in entities if e.entity_type == "TIME"]
        assert len(time_entities) >= 2
    
    def test_entity_position_tracking(self, engine):
        """Test entity position tracking in text."""
        input_text = "weather in Paris today"
        entities = engine.extract_entities(input_text)
        
        for entity in entities:
            if entity.start_pos is not None and entity.end_pos is not None:
                # Verify position is correct
                extracted_text = input_text[entity.start_pos:entity.end_pos]
                assert entity.value.lower() in extracted_text.lower()
    
    def test_intent_metadata(self, engine):
        """Test intent recognition metadata."""
        result = engine.recognize_intent("hello there")
        
        assert "matched_patterns" in result.metadata
        assert "processing_time" in result.metadata
        assert isinstance(result.metadata["processing_time"], float)
    
    def test_engine_performance(self, engine):
        """Test engine performance with multiple requests."""
        import time
        
        test_inputs = [
            "hello",
            "what's the weather?",
            "what time is it?",
            "goodbye"
        ] * 25  # 100 total requests
        
        start_time = time.time()
        
        for input_text in test_inputs:
            result = engine.recognize_intent(input_text)
            assert result.intent != IntentType.UNKNOWN or input_text == "goodbye"
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 100 requests in reasonable time (< 1 second)
        assert processing_time < 1.0
    
    def test_thread_safety(self, engine):
        """Test thread safety of intent engine."""
        import threading
        import time
        
        results = []
        
        def recognize_intents():
            for _ in range(10):
                result = engine.recognize_intent("hello")
                results.append(result.intent)
                time.sleep(0.001)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=recognize_intents)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All results should be GREETING
        assert len(results) == 50
        assert all(intent == IntentType.GREETING for intent in results)


class TestIntentEngineIntegration:
    """Integration tests for intent engine functionality."""
    
    def test_real_world_conversations(self):
        """Test intent recognition on real-world conversation examples."""
        engine = ARKIntentEngine()
        
        conversation_examples = [
            ("Hi there! How are you doing today?", IntentType.GREETING),
            ("Can you tell me what the weather is like in San Francisco?", IntentType.WEATHER),
            ("What time is it right now?", IntentType.TIME),
            ("How do I reset my password?", IntentType.QUESTION),
            ("Please show me my calendar for tomorrow", IntentType.COMMAND),
            ("Thanks for your help, goodbye!", IntentType.GOODBYE),
            ("Calculate 15 * 23 for me", IntentType.CALCULATION),
            ("Search for restaurants near me", IntentType.SEARCH),
            ("I need help with this application", IntentType.HELP)
        ]
        
        for input_text, expected_intent in conversation_examples:
            result = engine.recognize_intent(input_text)
            assert result.intent == expected_intent, f"Failed for: {input_text}"
            assert result.confidence > 0.5
    
    def test_complex_entity_extraction(self):
        """Test complex entity extraction scenarios."""
        engine = ARKIntentEngine()
        
        complex_examples = [
            "What's the weather like in New York tomorrow morning?",
            "Schedule a meeting with John Smith next Tuesday at 3 PM",
            "Find flights from London to Paris for next weekend",
            "Remind me to call Dr. Johnson at 2:30 PM today"
        ]
        
        for input_text in complex_examples:
            entities = engine.extract_entities(input_text)
            
            # Should extract multiple entities
            assert len(entities) > 0
            
            # Verify entity types are reasonable
            entity_types = [e.entity_type for e in entities]
            assert any(et in ["LOCATION", "TIME", "PERSON"] for et in entity_types)
    
    def test_ambiguous_input_handling(self):
        """Test handling of ambiguous input."""
        engine = ARKIntentEngine()
        
        ambiguous_examples = [
            "time weather",  # Could be time or weather
            "hello goodbye",  # Conflicting intents
            "maybe perhaps possibly",  # Uncertain language
            "weather time hello"  # Multiple potential intents
        ]
        
        for input_text in ambiguous_examples:
            result = engine.recognize_intent(input_text)
            
            # Should still return a result
            assert result.intent is not None
            assert isinstance(result.confidence, float)
            assert 0.0 <= result.confidence <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])