"""
Tests for the ARK Engine module.

This module contains comprehensive tests for the core ARK reasoning engine,
including decision making, tool execution, response generation, and
integration with other components.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

from src.core.ark_engine import ARKState, ARKDecision, ARKEngine
from src.core.intent_engine import IntentType, IntentResult, Entity
from src.core.context_manager import ConversationContext, ConversationTurn
from src.core.mcp_client import ARKMCPClient
from src.core.server_config import SimpleMCPServerConfig


class TestARKState:
    """Test cases for ARKState enum."""
    
    def test_ark_state_values(self):
        """Test that all expected ARK states are defined."""
        expected_states = [
            "INITIALIZING", "READY", "PROCESSING", "TOOL_EXECUTION", 
            "ERROR", "SHUTDOWN"
        ]
        
        for state in expected_states:
            assert hasattr(ARKState, state)
        
        # Test specific values
        assert ARKState.INITIALIZING.value == "initializing"
        assert ARKState.READY.value == "ready"
        assert ARKState.PROCESSING.value == "processing"
        assert ARKState.TOOL_EXECUTION.value == "tool_execution"
        assert ARKState.ERROR.value == "error"
        assert ARKState.SHUTDOWN.value == "shutdown"


class TestARKDecision:
    """Test cases for ARKDecision class."""
    
    def test_decision_creation(self):
        """Test basic ARK decision creation."""
        decision = ARKDecision(
            action_type="use_tools",
            tools_to_use=["weather_tool"],
            parameters={"location": "New York"},
            confidence=0.85,
            reasoning="User requested weather information for New York",
            metadata={"timestamp": "2024-01-01T00:00:00"}
        )
        
        assert decision.action_type == "use_tools"
        assert decision.tools_to_use == ["weather_tool"]
        assert decision.parameters["location"] == "New York"
        assert decision.confidence == 0.85
        assert "weather information" in decision.reasoning
        assert decision.metadata["timestamp"] == "2024-01-01T00:00:00"
    
    def test_decision_validation(self):
        """Test ARK decision validation."""
        # Test valid decision
        decision = ARKDecision(
            action_type="respond",
            tools_to_use=[],
            parameters={},
            confidence=0.9,
            reasoning="Direct response appropriate",
            metadata={}
        )
        
        assert decision.action_type == "respond"
        assert decision.tools_to_use == []
        assert decision.parameters == {}
        assert decision.confidence == 0.9
        assert decision.reasoning == "Direct response appropriate"
        assert decision.metadata == {}
    
    def test_decision_string_representation(self):
        """Test ARK decision string representation."""
        decision = ARKDecision(
            action_type="test_action",
            tools_to_use=["test_tool"],
            parameters={"param": "value"},
            confidence=0.8,
            reasoning="Test reasoning",
            metadata={}
        )
        repr_str = repr(decision)
        
        assert "ARKDecision" in repr_str
        assert "test_action" in repr_str
        assert "0.8" in repr_str


class TestARKEngine:
    """Test cases for ARKEngine class."""
    
    @pytest.fixture
    def mock_mcp_client(self):
        """Create a mock MCP client."""
        client = Mock(spec=ARKMCPClient)
        client.is_connected = Mock(return_value=True)
        client.get_available_tools = Mock(return_value=[
            {"name": "weather_tool", "description": "Get weather information"},
            {"name": "time_tool", "description": "Get current time"},
            {"name": "search_tool", "description": "Search for information"}
        ])
        client.execute_tool = AsyncMock(return_value={
            "success": True,
            "result": "Tool executed successfully"
        })
        return client
    
    @pytest.fixture
    def mock_context(self):
        """Create a mock conversation context."""
        context = Mock(spec=ConversationContext)
        context.session_id = "test_session"
        context.get_recent_turns = Mock(return_value=[])
        context.get_memory = Mock(return_value=None)
        context.add_turn = Mock()
        context.add_exchange = Mock()
        context.update_memory = Mock()
        context.get_context_summary = Mock(return_value="")
        context.find_similar_exchanges = Mock(return_value=[])
        return context
    
    @pytest.fixture
    def engine(self, mock_mcp_client, mock_context):
        """Create an ARK engine for testing."""
        config = {
            'max_conversation_history': 100,
            'max_tool_chain_length': 5,
            'confidence_threshold': 0.7,
            'enable_tool_chaining': True
        }
        engine = ARKEngine(config=config)
        # Replace the auto-created components with our mocks
        engine.mcp_client = mock_mcp_client
        engine.context_manager = mock_context
        
        # Mock intent engine
        mock_intent_engine = Mock()
        mock_intent_engine.analyze_intent = AsyncMock()
        engine.intent_engine = mock_intent_engine
        
        # Set state to READY for testing
        engine.state = ARKState.READY
        # Set up available tools for testing
        engine.available_tools = {
            'weather_tool': {'name': 'weather_tool', 'description': 'Get weather information'},
            'time_tool': {'name': 'time_tool', 'description': 'Get current time'},
            'search_tool': {'name': 'search_tool', 'description': 'Search for information'},
            'calculator_tool': {'name': 'calculator_tool', 'description': 'Perform calculations'}
        }
        return engine
    
    def test_engine_initialization(self, engine, mock_mcp_client, mock_context):
        """Test ARK engine initialization."""
        assert engine.mcp_client == mock_mcp_client
        assert engine.context_manager == mock_context
        assert engine.state == ARKState.READY
        assert engine.intent_engine is not None
        assert engine.decision_history == []
    
    def test_engine_state_management(self, engine):
        """Test ARK engine state management."""
        # Initial state
        assert engine.state == ARKState.READY
        
        # Change state
        engine.state = ARKState.PROCESSING
        assert engine.state == ARKState.PROCESSING
        
        # State transition
        engine.state = ARKState.READY
        assert engine.state == ARKState.READY
    
    @pytest.mark.asyncio
    async def test_process_input_greeting(self, engine):
        """Test processing greeting input."""
        user_input = "Hello there!"
        
        # Mock intent engine to return greeting intent
        engine.intent_engine.recognize_intent.return_value = IntentResult(
            intent=IntentType.GREETING,
            confidence=0.9,
            entities=[],
            raw_text=user_input,
            processed_text=user_input.lower()
        )
        
        response = await engine.process_input(user_input)
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert engine.state == ARKState.READY  # Should return to ready
        
        # Verify context was updated
        engine.context_manager.add_exchange.assert_called_once()
        call_args = engine.context_manager.add_exchange.call_args
        # Check keyword arguments
        assert call_args.kwargs['user_input'] == user_input
        assert call_args.kwargs['agent_response'] == response
    
    @pytest.mark.asyncio
    async def test_process_input_weather_request(self, engine):
        """Test processing weather request."""
        user_input = "What's the weather like in New York?"
        
        # Mock intent engine to return weather request intent
        engine.intent_engine.recognize_intent.return_value = IntentResult(
            intent=IntentType.TOOL_USE,
            confidence=0.9,
            entities=[],
            raw_text=user_input,
            processed_text=user_input.lower()
        )
        
        # Mock tool execution
        engine.mcp_client.execute_tool.return_value = {
            "success": True,
            "result": "Sunny, 75°F in New York"
        }
        
        response = await engine.process_input(user_input)
        
        assert isinstance(response, str)
        assert "New York" in response or "weather" in response.lower()
        
        # Verify tool was called
        engine.mcp_client.execute_tool.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_input_time_request(self, engine):
        """Test processing time request."""
        user_input = "What time is it?"
        
        # Mock intent engine to return time request intent
        engine.intent_engine.recognize_intent.return_value = IntentResult(
            intent=IntentType.QUESTION,
            confidence=0.9,
            entities=[],
            raw_text=user_input,
            processed_text=user_input.lower()
        )
        
        # Mock tool execution
        engine.mcp_client.execute_tool.return_value = {
            "success": True,
            "result": "Current time: 2:30 PM"
        }
        
        response = await engine.process_input(user_input)
        
        assert isinstance(response, str)
        assert "time" in response.lower() or "2:30" in response
    
    @pytest.mark.asyncio
    async def test_process_input_unknown_intent(self, engine):
        """Test processing unknown intent."""
        user_input = "xyzabc random gibberish"
        
        # Mock intent engine to return UNKNOWN intent
        engine.intent_engine.recognize_intent.return_value = IntentResult(
            intent=IntentType.UNKNOWN,
            confidence=0.3,
            entities=[],
            raw_text=user_input,
            processed_text=user_input.lower()
        )
        
        response = await engine.process_input(user_input)
        
        assert isinstance(response, str)
        assert len(response) > 0
        # Should provide helpful response even for unknown input
        assert any(word in response.lower() for word in ["help", "understand", "clarify", "sorry"])
    
    @pytest.mark.asyncio
    async def test_make_decision_greeting(self, engine):
        """Test decision making for greeting intent."""
        intent_result = IntentResult(
            intent=IntentType.GREETING,
            confidence=0.9,
            entities=[],
            raw_text="Hello!",
            processed_text="greeting",
            metadata={}
        )
        
        decision = await engine._make_decision(intent_result, "Hello!", [])
        
        assert decision.action_type == "respond_greeting"
        assert decision.tools_to_use == []
        assert decision.confidence > 0.5
        assert "greeting" in decision.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_make_decision_weather(self, engine):
        """Test decision making for weather intent."""
        entities = [Entity("location", "Paris", "LOCATION", 0.8)]
        intent_result = IntentResult(
            intent=IntentType.TOOL_USE,
            confidence=0.85,
            entities=entities,
            raw_text="Weather in Paris?",
            processed_text="weather query for Paris",
            metadata={}
        )
        
        decision = await engine._make_decision(intent_result, "Weather in Paris?", [])
        
        assert decision.action_type == "use_tools"
        assert "weather_tool" in decision.tools_to_use
        assert "location" in decision.parameters
        assert decision.parameters["location"] == "Paris"
        assert decision.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_make_decision_time(self, engine):
        """Test decision making for time intent."""
        intent_result = IntentResult(
            intent=IntentType.TOOL_USE,
            confidence=0.9,
            entities=[],
            raw_text="What time is it?",
            processed_text="time query",
            metadata={}
        )
        
        decision = await engine._make_decision(intent_result, "What time is it?", [])
        
        assert decision.action_type == "use_tools"
        assert "time_tool" in decision.tools_to_use
        assert decision.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_make_decision_search(self, engine):
        """Test decision making for search intent."""
        intent_result = IntentResult(
            intent=IntentType.TOOL_USE,
            confidence=0.8,
            entities=[],
            raw_text="Search for restaurants",
            processed_text="search query for restaurants",
            metadata={}
        )
        
        decision = await engine._make_decision(intent_result, "Search for restaurants", [])
        
        assert decision.action_type == "use_tools"
        assert "search_tool" in decision.tools_to_use
        assert "query" in decision.parameters
        assert decision.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_make_decision_unknown(self, engine):
        """Test decision making for unknown intent."""
        intent_result = IntentResult(
            intent=IntentType.UNKNOWN,
            confidence=0.2,
            entities=[],
            raw_text="Unknown input",
            processed_text="unknown",
            metadata={}
        )
        
        decision = await engine._make_decision(intent_result, "random input", [])
        
        assert decision.action_type == "clarify_intent"
        assert decision.tools_to_use == []
        assert decision.confidence <= 0.5
    
    @pytest.mark.asyncio
    async def test_execute_tool_success(self, engine):
        """Test successful tool execution."""
        decision = ARKDecision(
            action_type="use_tools",
            tools_to_use=["weather_tool"],
            parameters={"location": "Tokyo"},
            confidence=0.8,
            reasoning="Weather tool execution",
            metadata={}
        )
        
        # Mock successful execution
        engine.mcp_client.execute_tool.return_value = {
            "success": True,
            "result": "Cloudy, 20°C in Tokyo"
        }
        
        result = await engine._execute_tools(decision.tools_to_use, decision.parameters)
        
        assert "weather_tool" in result
        assert result["weather_tool"]["success"] is True
        assert "Tokyo" in result["weather_tool"]["result"]
        engine.mcp_client.execute_tool.assert_called_once_with(
            "weather_tool", {"location": "Tokyo"}
        )
    
    @pytest.mark.asyncio
    async def test_execute_tool_failure(self, engine):
        """Test tool execution failure."""
        decision = ARKDecision(
            action_type="use_tools",
            tools_to_use=["broken_tool"],
            parameters={},
            confidence=0.8,
            reasoning="Testing broken tool",
            metadata={}
        )
        
        # Mock failed execution
        engine.mcp_client.execute_tool.return_value = {
            "success": False,
            "error": "Tool not found"
        }
        
        result = await engine._execute_tools(decision.tools_to_use, decision.parameters)
        
        assert "broken_tool" in result
        assert result["broken_tool"]["success"] is False
        assert "error" in result["broken_tool"]
    
    @pytest.mark.asyncio
    async def test_execute_tool_exception(self, engine):
        """Test tool execution with exception."""
        decision = ARKDecision(
            action_type="use_tools",
            tools_to_use=["error_tool"],
            parameters={},
            confidence=0.8,
            reasoning="Testing error tool",
            metadata={}
        )
        
        # Mock exception
        engine.mcp_client.execute_tool.side_effect = Exception("Connection error")
        
        result = await engine._execute_tools(decision.tools_to_use, decision.parameters)
        
        assert "error_tool" in result
        assert "error" in result["error_tool"]
        assert "Connection error" in result["error_tool"]["error"]
    
    @pytest.mark.asyncio
    async def test_generate_response_greeting(self, engine):
        """Test response generation for greeting."""
        intent_result = IntentResult(
            intent=IntentType.GREETING,
            confidence=0.9,
            entities=[],
            raw_text="Hello",
            processed_text="Hello",
            metadata={}
        )
        
        # Create a mock decision for the response generation
        mock_decision = ARKDecision(
            action_type="respond_greeting",
            confidence=0.9,
            reasoning="User is greeting",
            tools_to_use=[],
            parameters={},
            metadata={}
        )
        
        response = await engine._generate_response(intent_result, mock_decision, {})
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert any(word in response.lower() for word in ["hello", "hi", "greetings"])
    
    @pytest.mark.asyncio
    async def test_generate_response_with_tool_result(self, engine):
        """Test response generation with tool result."""
        intent_result = IntentResult(
            intent=IntentType.TOOL_USE,
            confidence=0.8,
            entities=[Entity("location", "Berlin", "LOCATION", 0.9)],
            raw_text="Weather in Berlin?",
            processed_text="Weather in Berlin?",
            metadata={}
        )
        
        tool_result = {
            "success": True,
            "result": "Rainy, 15°C in Berlin"
        }
        
        # Create a mock decision for the response generation
        mock_decision = ARKDecision(
            action_type="use_tools",
            confidence=0.8,
            reasoning="User wants weather information for Berlin",
            tools_to_use=["weather_tool"],
            parameters={"location": "Berlin"},
            metadata={}
        )
        
        response = await engine._generate_response(intent_result, mock_decision, {"weather_tool": tool_result})
        
        assert isinstance(response, str)
        assert "Berlin" in response
        assert "15°C" in response or "rainy" in response.lower()
    
    @pytest.mark.asyncio
    async def test_generate_response_tool_error(self, engine):
        """Test response generation with tool error."""
        intent_result = IntentResult(
            intent=IntentType.TOOL_USE,
            confidence=0.8,
            entities=[],
            raw_text="What's the weather?",
            processed_text="What's the weather?",
            metadata={}
        )
        
        tool_result = {
            "success": False,
            "error": "Service unavailable"
        }
        
        # Create a mock decision for the response generation
        mock_decision = ARKDecision(
            action_type="use_tools",
            confidence=0.8,
            reasoning="User wants weather information",
            tools_to_use=["weather_tool"],
            parameters={"location": "current"},
            metadata={}
        )
        
        response = await engine._generate_response(intent_result, mock_decision, {"weather_tool": tool_result})
        
        assert isinstance(response, str)
        assert any(word in response.lower() for word in ["sorry", "error", "unavailable", "problem"])
    
    def test_decision_history_tracking(self, engine):
        """Test decision history tracking."""
        # Make several decisions
        decisions = [
            ARKDecision(
                action_type="respond",
                confidence=0.9,
                reasoning="Simple response",
                tools_to_use=[],
                parameters={},
                metadata={}
            ),
            ARKDecision(
                action_type="use_tools",
                confidence=0.8,
                reasoning="Weather tool needed",
                tools_to_use=["weather_tool"],
                parameters={"location": "default"},
                metadata={}
            ),
            ARKDecision(
                action_type="respond",
                confidence=0.7,
                reasoning="Follow-up response",
                tools_to_use=[],
                parameters={},
                metadata={}
            )
        ]
        
        for decision in decisions:
            engine.decision_history.append(decision)
        
        assert len(engine.decision_history) == 3
        assert engine.decision_history[0].action_type == "respond"
        assert "weather_tool" in engine.decision_history[1].tools_to_use
        assert engine.decision_history[2].confidence == 0.7
    
    @pytest.mark.asyncio
    async def test_context_integration(self, engine):
        """Test integration with conversation context."""
        # Mock context with memory
        engine.context_manager.get_memory.return_value = "Alice"
        
        # Process input that might use memory
        intent_result = IntentResult(
            intent=IntentType.GREETING,
            confidence=0.9,
            entities=[],
            raw_text="Hello again!",
            processed_text="Hello again!",
            metadata={}
        )
        
        # Create a mock decision for the response generation
        mock_decision = ARKDecision(
            action_type="respond_greeting",
            confidence=0.9,
            reasoning="User is greeting again",
            tools_to_use=[],
            parameters={},
            metadata={}
        )
        
        response = await engine._generate_response(intent_result, mock_decision, {})
        
        # Should generate a valid response
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, engine):
        """Test error handling and recovery."""
        # Set engine to READY state first
        engine.state = ARKState.READY
        
        # Mock intent engine to return weather request intent
        engine.intent_engine.recognize_intent.return_value = IntentResult(
            intent=IntentType.REQUEST,
            confidence=0.9,
            entities=[],
            raw_text="What's the weather?",
            processed_text="what's the weather?"
        )
        
        # Mock MCP client to raise an exception
        engine.mcp_client.call_tool = Mock(side_effect=Exception("Tool error"))
        
        # Process input that would trigger tool use
        response = await engine.process_input("What's the weather?")
        
        # Should handle error gracefully and return to READY state
        assert engine.state in [ARKState.READY, ARKState.ERROR]
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_state_transitions(self, engine):
        """Test state transitions during processing."""
        # Engine starts in READY state (set by fixture)
        assert engine.state == ARKState.READY
        
        # Process input should change state to PROCESSING then back to READY
        user_input = "What's the weather?"
        
        # Mock intent engine to return weather request intent
        engine.intent_engine.recognize_intent.return_value = IntentResult(
            intent=IntentType.REQUEST,
            confidence=0.9,
            entities=[],
            raw_text=user_input,
            processed_text=user_input.lower()
        )
        
        response = await engine.process_input(user_input)
        
        # Should return to READY state after processing
        assert isinstance(response, str)
        assert len(response) > 0
        assert engine.state == ARKState.READY
    
    @pytest.mark.asyncio
    async def test_tool_selection_logic(self, engine):
        """Test tool selection logic."""
        # Test weather tool selection
        weather_intent = IntentResult(
            intent=IntentType.TOOL_USE,
            confidence=0.8,
            entities=[],
            raw_text="weather query",
            processed_text="weather query",
            metadata={}
        )
        weather_decision = await engine._make_decision(weather_intent, "weather query", [])
        assert weather_decision.action_type == "use_tools"
        
        # Test time tool selection
        time_intent = IntentResult(
            intent=IntentType.TOOL_USE,
            confidence=0.9,
            entities=[],
            raw_text="time query",
            processed_text="time query",
            metadata={}
        )
        time_decision = await engine._make_decision(time_intent, "time query", [])
        assert time_decision.action_type == "use_tools"
        
        # Test search tool selection
        search_intent = IntentResult(
            intent=IntentType.TOOL_USE,
            confidence=0.7,
            entities=[],
            raw_text="search query",
            processed_text="search query",
            metadata={}
        )
        search_decision = await engine._make_decision(search_intent, "search query", [])
        assert search_decision.action_type == "use_tools"
    
    @pytest.mark.asyncio
    async def test_parameter_extraction(self, engine):
        """Test parameter extraction for tool calls."""
        # Weather with location
        entities = [Entity("location", "London", "LOCATION", 0.9)]
        weather_intent = IntentResult(
            intent=IntentType.TOOL_USE,
            confidence=0.8,
            entities=entities,
            raw_text="weather in London",
            processed_text="weather in London",
            metadata={}
        )
        decision = await engine._make_decision(weather_intent, "weather in London", [])
        
        assert decision.action_type == "use_tools"
        
        # Search with query
        search_intent = IntentResult(
            intent=IntentType.TOOL_USE,
            confidence=0.7,
            entities=[],
            raw_text="search for restaurants",
            processed_text="search for restaurants",
            metadata={}
        )
        search_decision = await engine._make_decision(search_intent, "search for restaurants", [])
        
        assert search_decision.action_type == "use_tools"
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, engine):
        """Test concurrent input processing."""
        inputs = [
            "Hello",
            "What's the weather?",
            "What time is it?",
            "Goodbye"
        ]
        
        # Process inputs concurrently
        tasks = [engine.process_input(inp) for inp in inputs]
        responses = await asyncio.gather(*tasks)
        
        # All should return valid responses
        assert len(responses) == 4
        for response in responses:
            assert isinstance(response, str)
            assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_confidence_thresholds(self, engine):
        """Test confidence threshold handling."""
        # High confidence intent
        high_confidence = IntentResult(
            intent=IntentType.TOOL_USE,
            confidence=0.95,
            entities=[],
            raw_text="weather",
            processed_text="weather",
            metadata={}
        )
        high_decision = await engine._make_decision(high_confidence, "weather", [])
        assert high_decision.confidence >= 0.0
        
        # Low confidence intent
        low_confidence = IntentResult(
            intent=IntentType.UNKNOWN,
            confidence=0.1,
            entities=[],
            raw_text="unclear input",
            processed_text="unclear input",
            metadata={}
        )
        low_decision = await engine._make_decision(low_confidence, "unclear input", [])
        assert low_decision.confidence < 0.5
        assert low_decision.action_type == "clarify_intent"  # Should clarify for low confidence
    
    @pytest.mark.asyncio
    async def test_memory_updates(self, engine):
        """Test memory updates during processing."""
        # Set engine to READY state first
        engine.state = ARKState.READY
        
        # Process input that might update memory
        user_input = "My name is Bob"
        
        # Mock intent engine to return information intent
        engine.intent_engine.recognize_intent.return_value = IntentResult(
            intent=IntentType.INFORMATION,
            confidence=0.8,
            entities=[],
            raw_text=user_input,
            processed_text=user_input.lower()
        )
        
        await engine.process_input(user_input)
        
        # Should have attempted to update memory
        # (Implementation would need to extract name and store it)
        engine.context_manager.add_exchange.assert_called_once()
    
    def test_engine_shutdown(self, engine):
        """Test engine shutdown."""
        engine.state = ARKState.SHUTDOWN
        assert engine.state == ARKState.SHUTDOWN
        
        # Engine should handle shutdown state gracefully
        assert engine.state == ARKState.SHUTDOWN


class TestARKEngineIntegration:
    """Integration tests for ARK engine functionality."""
    
    @pytest.mark.asyncio
    async def test_full_conversation_flow(self):
        """Test complete conversation flow."""
        # Create real components (not mocked)
        from src.core.intent_engine import ARKIntentEngine
        from src.core.context_manager import ConversationContext
        
        context = ConversationContext()
        intent_engine = ARKIntentEngine()
        
        # Mock MCP client
        mcp_client = Mock(spec=ARKMCPClient)
        mcp_client.is_connected = Mock(return_value=True)
        mcp_client.get_available_tools = Mock(return_value=[
            {"name": "weather_tool", "description": "Get weather"}
        ])
        mcp_client.execute_tool = AsyncMock(return_value={
            "success": True,
            "result": "Sunny, 25°C"
        })
        
        engine = ARKEngine()
        engine.mcp_client = mcp_client
        engine.context_manager = context
        engine.intent_engine = intent_engine
        engine.state = ARKState.READY
        
        # Simulate conversation
        conversation = [
            "Hello there!",
            "What's the weather like?",
            "Thank you!",
            "Goodbye"
        ]
        
        responses = []
        for user_input in conversation:
            response = await engine.process_input(user_input)
            responses.append(response)
        
        # Verify responses
        assert len(responses) == 4
        for response in responses:
            assert isinstance(response, str)
            assert len(response) > 0
        
        # Verify context was updated
        assert len(context.conversation_history) == 4
        
        # Verify decision history
        assert len(engine.decision_history) == 4
    
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery in conversation flow."""
        from src.core.context_manager import ConversationContext
        from src.core.intent_engine import ARKIntentEngine
        
        context = ConversationContext()
        intent_engine = ARKIntentEngine()
        
        # Mock MCP client that fails
        mcp_client = Mock(spec=ARKMCPClient)
        mcp_client.is_connected = Mock(return_value=False)
        mcp_client.execute_tool = AsyncMock(side_effect=Exception("Connection failed"))
        
        engine = ARKEngine()
        engine.mcp_client = mcp_client
        engine.context_manager = context
        engine.intent_engine = intent_engine
        engine.state = ARKState.READY
        
        # Try to process input that would require tools
        response = await engine.process_input("What's the weather?")
        
        # Should handle error gracefully
        assert isinstance(response, str)
        assert len(response) > 0
        assert engine.state in [ARKState.READY, ARKState.ERROR]
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self):
        """Test performance benchmarks."""
        import time
        from src.core.context_manager import ConversationContext
        from src.core.intent_engine import ARKIntentEngine
        
        context = ConversationContext()
        intent_engine = ARKIntentEngine()
        mcp_client = Mock(spec=ARKMCPClient)
        mcp_client.is_connected = Mock(return_value=True)
        mcp_client.get_available_tools = Mock(return_value=[])
        
        engine = ARKEngine()
        engine.mcp_client = mcp_client
        engine.context_manager = context
        engine.intent_engine = intent_engine
        
        # Test intent recognition performance
        start_time = time.time()
        
        for i in range(100):
            intent_result = engine.intent_engine.recognize_intent(f"test input {i}")
            decision = await engine._make_decision(intent_result, f"test input {i}", [])
            
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 100 decisions quickly
        assert processing_time < 1.0  # Less than 1 second
        
        # Average time per decision should be reasonable
        avg_time = processing_time / 100
        assert avg_time < 0.01  # Less than 10ms per decision


if __name__ == "__main__":
    pytest.main([__file__, "-v"])