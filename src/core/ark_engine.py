"""
ARK (Autonomous Reasoning Kernel) Engine

This is the core reasoning engine that powers the Jarvis AI agent.
ARK integrates MCP tools, intent recognition, context management,
and decision-making capabilities.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .mcp_client import ARKMCPClient
from .server_config import SimpleMCPServerConfig
from .context_manager import ConversationContext
from .intent_engine import ARKIntentEngine, IntentType, IntentResult


class ARKState(Enum):
    """ARK engine operational states."""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    TOOL_EXECUTION = "tool_execution"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class ARKDecision:
    """Represents a decision made by the ARK engine."""
    action_type: str
    confidence: float
    reasoning: str
    tools_to_use: List[str]
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]


class ARKEngine:
    """
    Autonomous Reasoning Kernel (ARK) Engine
    
    The core reasoning engine that orchestrates all AI agent capabilities
    including intent recognition, tool selection, context management,
    and response generation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ARK engine.
        
        Args:
            config: Configuration dictionary for ARK engine
        """
        self.config = config or {}
        self.state = ARKState.INITIALIZING
        self.ark_logger = logging.getLogger('ark.engine')
        
        # Initialize core components
        self.mcp_client = ARKMCPClient()
        self.context_manager = ConversationContext(
            max_history=self.config.get('max_conversation_history', 100)
        )
        self.intent_engine = ARKIntentEngine()
        
        # ARK-specific attributes
        self.available_tools: Dict[str, Any] = {}
        self.tool_usage_stats: Dict[str, int] = {}
        self.decision_history: List[ARKDecision] = []
        
        # Configuration
        self.max_tool_chain_length = self.config.get('max_tool_chain_length', 5)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.enable_tool_chaining = self.config.get('enable_tool_chaining', True)
        
        # Initialize performance metrics
        self._initialize_performance_metrics()
        
        self.ark_logger.info("ARK engine initialized - Autonomous Reasoning Kernel ready")
    
    async def initialize(self, mcp_servers: Optional[List[SimpleMCPServerConfig]] = None) -> bool:
        """
        Initialize ARK engine with MCP servers and capabilities.
        
        Args:
            mcp_servers: List of MCP server configurations
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.state = ARKState.INITIALIZING
            self.ark_logger.info("ARK: Starting initialization sequence")
            
            # Initialize MCP client with servers
            if mcp_servers:
                for server_config in mcp_servers:
                    success = await self.mcp_client.connect_to_server(server_config)
                    if success:
                        self.ark_logger.info(f"ARK: Connected to MCP server '{server_config.name}'")
                    else:
                        self.ark_logger.warning(f"ARK: Failed to connect to MCP server '{server_config.name}'")
            
            # Discover available tools
            await self._discover_tools()
            
            # Initialize performance tracking
            self._initialize_performance_metrics()
            
            self.state = ARKState.READY
            self.ark_logger.info(f"ARK: Initialization complete - {len(self.available_tools)} tools available")
            return True
            
        except Exception as e:
            self.state = ARKState.ERROR
            self.ark_logger.error(f"ARK: Initialization failed: {e}")
            return False
    
    async def process_input(self, user_input: str) -> str:
        """
        Process user input through the ARK reasoning pipeline.
        
        Args:
            user_input: User's natural language input
            
        Returns:
            Generated response
        """
        if self.state != ARKState.READY:
            return "ARK engine is not ready. Please wait for initialization to complete."
        
        try:
            self.state = ARKState.PROCESSING
            self.ark_logger.info(f"ARK: Processing input: '{user_input[:50]}...'")
            
            # Step 1: Intent Recognition and Entity Extraction
            intent_result = self.intent_engine.recognize_intent(user_input)
            self.ark_logger.debug(f"ARK: Recognized intent '{intent_result.intent.value}' with confidence {intent_result.confidence:.2f}")
            
            # Step 2: Context Analysis
            context_summary = self.context_manager.get_context_summary()
            recent_turns = self.context_manager.get_recent_turns()
            similar_exchanges = self.context_manager.find_similar_exchanges(user_input)
            # Check for relevant memory information
            user_memory = self.context_manager.get_memory("user_preferences", {})
            
            # Step 3: Decision Making
            decision = await self._make_decision(intent_result, context_summary, similar_exchanges)
            self.decision_history.append(decision)
            
            # Step 4: Tool Execution (if needed)
            tool_results = {}
            if decision.tools_to_use:
                self.state = ARKState.TOOL_EXECUTION
                tool_results = await self._execute_tools(decision.tools_to_use, decision.parameters)
            
            # Step 5: Response Generation
            response = await self._generate_response(intent_result, decision, tool_results)
            
            # Step 6: Update Context
            self.context_manager.add_exchange(
                user_input=user_input,
                agent_response=response,
                intent=intent_result.intent.value,
                tools_used=decision.tools_to_use,
                metadata={
                    "decision_confidence": decision.confidence,
                    "entities_extracted": len(intent_result.entities),
                    "tool_results": bool(tool_results)
                }
            )
            
            # Update performance metrics
            self._update_performance_metrics(intent_result, decision, tool_results)
            
            self.state = ARKState.READY
            self.ark_logger.info("ARK: Input processing complete")
            return response
            
        except Exception as e:
            self.state = ARKState.ERROR
            self.ark_logger.error(f"ARK: Error processing input: {e}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    async def _discover_tools(self) -> None:
        """Discover and catalog available tools from MCP servers."""
        try:
            self.available_tools = {}
            
            # Discover tools from all connected servers
            for server_name in self.mcp_client.sessions.keys():
                try:
                    server_tools = await self.mcp_client.discover_tools(server_name)
                    for tool in server_tools:
                        tool_name = tool.get('name', f"unknown_tool_{len(self.available_tools)}")
                        self.available_tools[tool_name] = tool
                except Exception as e:
                    self.ark_logger.warning(f"ARK: Failed to discover tools from server '{server_name}': {e}")
            
            self.ark_logger.info(f"ARK: Discovered {len(self.available_tools)} tools")
            
            # Initialize usage stats for all tools
            for tool_name in self.available_tools.keys():
                self.tool_usage_stats[tool_name] = 0
                
        except Exception as e:
            self.ark_logger.error(f"ARK: Tool discovery failed: {e}")
            self.available_tools = {}
    
    async def _make_decision(
        self, 
        intent_result: IntentResult, 
        context_summary: str,
        similar_exchanges: List[Any]
    ) -> ARKDecision:
        """
        Make a decision about how to respond to the user input.
        
        Args:
            intent_result: Result from intent recognition
            context_summary: Summary of conversation context
            similar_exchanges: Similar past exchanges
            
        Returns:
            ARK decision object
        """
        # Analyze intent and determine action type
        action_type = self._determine_action_type(intent_result.intent)
        
        # Select appropriate tools based on intent and entities
        tools_to_use = self._select_tools(intent_result)
        
        # Calculate confidence based on intent confidence and tool availability
        confidence = self._calculate_decision_confidence(intent_result, tools_to_use)
        
        # Generate reasoning explanation
        reasoning = self._generate_reasoning(intent_result, tools_to_use, similar_exchanges)
        
        # Prepare parameters for tool execution
        parameters = self._prepare_tool_parameters(intent_result, tools_to_use)
        
        decision = ARKDecision(
            action_type=action_type,
            confidence=confidence,
            reasoning=reasoning,
            tools_to_use=tools_to_use,
            parameters=parameters,
            metadata={
                "intent": intent_result.intent.value,
                "entity_count": len(intent_result.entities),
                "similar_exchanges_count": len(similar_exchanges),
                "decision_timestamp": self._get_timestamp()
            }
        )
        
        self.ark_logger.debug(f"ARK: Decision made - Action: {action_type}, Tools: {tools_to_use}, Confidence: {confidence:.2f}")
        return decision
    
    def _determine_action_type(self, intent: IntentType) -> str:
        """
        Determine the action type based on recognized intent.
        
        Args:
            intent: Recognized intent type
            
        Returns:
            Action type string
        """
        action_mapping = {
            IntentType.GREETING: "respond_greeting",
            IntentType.GOODBYE: "respond_goodbye",
            IntentType.QUESTION: "answer_question",
            IntentType.COMMAND: "execute_command",
            IntentType.REQUEST: "fulfill_request",
            IntentType.TOOL_USE: "use_tools",
            IntentType.HELP: "provide_help",
            IntentType.INFORMATION: "provide_information",
            IntentType.CONVERSATION: "engage_conversation",
            IntentType.UNKNOWN: "clarify_intent"
        }
        
        return action_mapping.get(intent, "engage_conversation")
    
    def _select_tools(self, intent_result: IntentResult) -> List[str]:
        """
        Select appropriate tools based on intent and entities.
        
        Args:
            intent_result: Result from intent recognition
            
        Returns:
            List of tool names to use
        """
        selected_tools = []
        
        # Tool selection based on intent type
        if intent_result.intent == IntentType.TOOL_USE:
            # Analyze entities and text for specific tool requirements
            text_lower = intent_result.raw_text.lower()
            
            # Weather-related tools
            if any(word in text_lower for word in ['weather', 'temperature', 'forecast', 'rain', 'sunny']):
                if 'weather_tool' in self.available_tools:
                    selected_tools.append('weather_tool')
            
            # Time-related tools
            if any(word in text_lower for word in ['time', 'date', 'calendar', 'schedule']):
                if 'time_tool' in self.available_tools:
                    selected_tools.append('time_tool')
            
            # Search-related tools
            if any(word in text_lower for word in ['search', 'find', 'lookup', 'google']):
                if 'search_tool' in self.available_tools:
                    selected_tools.append('search_tool')
            
            # Calculation tools
            if any(word in text_lower for word in ['calculate', 'compute', 'math', 'convert']):
                if 'calculator_tool' in self.available_tools:
                    selected_tools.append('calculator_tool')
        
        # Handle specific intent types
        elif intent_result.intent == IntentType.TIME:
            if 'time_tool' in self.available_tools:
                selected_tools.append('time_tool')
        
        elif intent_result.intent == IntentType.WEATHER:
            if 'weather_tool' in self.available_tools:
                selected_tools.append('weather_tool')
        
        elif intent_result.intent == IntentType.CALCULATION:
            if 'calculator_tool' in self.available_tools:
                selected_tools.append('calculator_tool')
        
        elif intent_result.intent == IntentType.SEARCH:
            if 'search_tool' in self.available_tools:
                selected_tools.append('search_tool')
        
        # Question answering might need search tools
        elif intent_result.intent == IntentType.QUESTION:
            if 'search_tool' in self.available_tools and not selected_tools:
                selected_tools.append('search_tool')
        
        # Limit tool chain length
        selected_tools = selected_tools[:self.max_tool_chain_length]
        
        return selected_tools
    
    def _calculate_decision_confidence(self, intent_result: IntentResult, tools_to_use: List[str]) -> float:
        """
        Calculate confidence in the decision.
        
        Args:
            intent_result: Result from intent recognition
            tools_to_use: Selected tools
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence from intent recognition
        base_confidence = intent_result.confidence
        
        # Boost confidence if we have appropriate tools
        tool_boost = 0.0
        if tools_to_use:
            available_tool_count = sum(1 for tool in tools_to_use if tool in self.available_tools)
            tool_boost = (available_tool_count / len(tools_to_use)) * 0.2
        
        # Reduce confidence for unknown intents
        if intent_result.intent == IntentType.UNKNOWN:
            base_confidence *= 0.5
        
        # Entity extraction boost
        entity_boost = min(len(intent_result.entities) * 0.05, 0.15)
        
        final_confidence = min(base_confidence + tool_boost + entity_boost, 1.0)
        return final_confidence
    
    def _generate_reasoning(
        self, 
        intent_result: IntentResult, 
        tools_to_use: List[str],
        similar_exchanges: List[Any]
    ) -> str:
        """
        Generate reasoning explanation for the decision.
        
        Args:
            intent_result: Result from intent recognition
            tools_to_use: Selected tools
            similar_exchanges: Similar past exchanges
            
        Returns:
            Reasoning explanation string
        """
        reasoning_parts = []
        
        # Intent-based reasoning
        reasoning_parts.append(f"Recognized intent: {intent_result.intent.value}")
        
        if intent_result.entities:
            entity_types = [entity.type for entity in intent_result.entities]
            reasoning_parts.append(f"Extracted entities: {', '.join(set(entity_types))}")
        
        # Tool selection reasoning
        if tools_to_use:
            reasoning_parts.append(f"Selected tools: {', '.join(tools_to_use)}")
        else:
            reasoning_parts.append("No tools required for this request")
        
        # Context reasoning
        if similar_exchanges:
            reasoning_parts.append(f"Found {len(similar_exchanges)} similar past exchanges")
        
        return "; ".join(reasoning_parts)
    
    def _prepare_tool_parameters(self, intent_result: IntentResult, tools_to_use: List[str]) -> Dict[str, Any]:
        """
        Prepare parameters for tool execution.
        
        Args:
            intent_result: Result from intent recognition
            tools_to_use: Selected tools
            
        Returns:
            Dictionary of flattened parameters
        """
        parameters = {}
        
        for tool_name in tools_to_use:
            # Extract relevant entities for each tool
            for entity in intent_result.entities:
                if tool_name == 'weather_tool' and entity.type.upper() == 'LOCATION':
                    parameters['location'] = entity.value
                elif tool_name == 'time_tool' and entity.type.upper() == 'LOCATION':
                    parameters['timezone'] = entity.value
                elif tool_name == 'search_tool':
                    parameters['query'] = intent_result.raw_text
                elif tool_name == 'calculator_tool' and entity.type.upper() == 'MATH':
                    parameters['expression'] = entity.value
                elif tool_name == 'calculator_tool' and entity.type.upper() == 'NUMBER':
                    parameters['expression'] = intent_result.raw_text
            
            # Default parameters if none extracted for this tool
            if tool_name == 'search_tool' and 'query' not in parameters:
                parameters['query'] = intent_result.raw_text
            elif tool_name == 'weather_tool' and 'location' not in parameters:
                parameters['location'] = 'current'
            elif tool_name == 'time_tool' and 'timezone' not in parameters:
                parameters['timezone'] = 'local'
        
        return parameters
    
    async def _execute_tools(self, tools_to_use: List[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute selected tools with given parameters.
        
        Args:
            tools_to_use: List of tool names to execute
            parameters: Flattened parameters dictionary
            
        Returns:
            Dictionary of tool results
        """
        results = {}
        
        for tool_name in tools_to_use:
            try:
                # Use the flattened parameters directly
                result = await self.mcp_client.execute_tool(tool_name, parameters)
                results[tool_name] = result
                
                # Update usage statistics
                self.tool_usage_stats[tool_name] = self.tool_usage_stats.get(tool_name, 0) + 1
                
                self.ark_logger.debug(f"ARK: Tool '{tool_name}' executed successfully")
                
            except Exception as e:
                self.ark_logger.error(f"ARK: Tool '{tool_name}' execution failed: {e}")
                results[tool_name] = {"error": str(e)}
        
        return results
    
    async def _generate_response(
        self, 
        intent_result: IntentResult, 
        decision: ARKDecision, 
        tool_results: Dict[str, Any]
    ) -> str:
        """
        Generate response based on intent, decision, and tool results.
        
        Args:
            intent_result: Result from intent recognition
            decision: ARK decision
            tool_results: Results from tool execution
            
        Returns:
            Generated response string
        """
        # Handle different action types
        if decision.action_type == "respond_greeting":
            return self._generate_greeting_response()
        
        elif decision.action_type == "respond_goodbye":
            return self._generate_goodbye_response()
        
        elif decision.action_type == "provide_help":
            return self._generate_help_response()
        
        elif decision.action_type == "use_tools" and tool_results:
            return self._generate_tool_response(tool_results)
        
        elif decision.action_type == "answer_question":
            if tool_results:
                return self._generate_tool_response(tool_results)
            else:
                return self._generate_conversational_response(intent_result)
        
        elif decision.action_type == "clarify_intent":
            return self._generate_clarification_response(intent_result)
        
        else:
            return self._generate_conversational_response(intent_result)
    
    def _generate_greeting_response(self) -> str:
        """Generate a greeting response."""
        greetings = [
            "Hello! I'm Jarvis, your AI assistant. How can I help you today?",
            "Hi there! I'm ready to assist you with any questions or tasks.",
            "Greetings! What can I do for you today?",
            "Hello! I'm here to help. What would you like to know or do?"
        ]
        import random
        return random.choice(greetings)
    
    def _generate_goodbye_response(self) -> str:
        """Generate a goodbye response."""
        goodbyes = [
            "Goodbye! Feel free to come back anytime you need assistance.",
            "Take care! I'll be here whenever you need help.",
            "Farewell! It was great helping you today.",
            "See you later! Don't hesitate to ask if you need anything else."
        ]
        import random
        return random.choice(goodbyes)
    
    def _generate_help_response(self) -> str:
        """Generate a help response."""
        available_tools_list = list(self.available_tools.keys())
        
        help_text = "I'm Jarvis, your AI assistant powered by the ARK (Autonomous Reasoning Kernel) engine. "
        help_text += "I can help you with various tasks including:\n\n"
        
        if available_tools_list:
            help_text += "Available capabilities:\n"
            for tool in available_tools_list:
                help_text += f"• {tool.title()} operations\n"
        
        help_text += "\nJust ask me questions or tell me what you'd like to do, and I'll do my best to help!"
        
        return help_text
    
    def _generate_tool_response(self, tool_results: Dict[str, Any]) -> str:
        """Generate response based on tool results."""
        if not tool_results:
            return "I wasn't able to get any results from the tools."
        
        response_parts = []
        
        for tool_name, result in tool_results.items():
            if isinstance(result, dict) and "error" in result:
                response_parts.append(f"I encountered an error with {tool_name}: {result['error']}")
            else:
                response_parts.append(f"Here's what I found using {tool_name}: {result}")
        
        return "\n\n".join(response_parts)
    
    def _generate_conversational_response(self, intent_result: IntentResult) -> str:
        """Generate a conversational response."""
        responses = [
            "I understand you're asking about that. Let me think about how I can help.",
            "That's an interesting question. Based on what you've told me, I'd say...",
            "I see what you're getting at. Here's my perspective on that:",
            "Thanks for sharing that with me. Let me provide some thoughts:"
        ]
        
        import random
        base_response = random.choice(responses)
        
        # Add context if entities were found
        if intent_result.entities:
            entity_info = ", ".join([f"{e.type}: {e.value}" for e in intent_result.entities[:3]])
            base_response += f" I noticed you mentioned: {entity_info}."
        
        return base_response
    
    def _generate_clarification_response(self, intent_result: IntentResult) -> str:
        """Generate a clarification response for unclear intents."""
        return ("I'm not entirely sure what you're asking for. Could you please rephrase your question "
                "or provide more details? I'm here to help with information, tools, and various tasks.")
    
    def _initialize_performance_metrics(self) -> None:
        """Initialize performance tracking metrics."""
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "tool_executions": 0,
            "average_response_time": 0.0,
            "intent_accuracy": 0.0,
            "tool_success_rate": 0.0,
            "start_time": self._get_timestamp()
        }
    
    def _update_performance_metrics(
        self, 
        intent_result: IntentResult, 
        decision: ARKDecision, 
        tool_results: Dict[str, Any]
    ) -> None:
        """Update performance metrics after processing a request."""
        self.performance_metrics["total_requests"] += 1
        
        if decision.confidence >= self.confidence_threshold:
            self.performance_metrics["successful_requests"] += 1
        
        if tool_results:
            self.performance_metrics["tool_executions"] += len(tool_results)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get basic engine status for testing and monitoring.
        
        Returns:
            Dictionary containing basic engine status
        """
        return {
            "state": self.state.value,
            "available_tools": len(self.available_tools),
            "decision_history_count": len(self.decision_history)
        }
    
    def get_engine_status(self) -> Dict[str, Any]:
        """
        Get current ARK engine status and metrics.
        
        Returns:
            Dictionary with engine status information
        """
        return {
            "state": self.state.value,
            "available_tools": list(self.available_tools.keys()),
            "tool_usage_stats": self.tool_usage_stats,
            "performance_metrics": self.performance_metrics,
            "conversation_stats": self.context_manager.get_session_stats(),
            "decision_history_count": len(self.decision_history),
            "configuration": {
                "max_tool_chain_length": self.max_tool_chain_length,
                "confidence_threshold": self.confidence_threshold,
                "enable_tool_chaining": self.enable_tool_chaining
            }
        }
    
    async def close(self) -> None:
        """
        Close the ARK engine and clean up resources.
        
        Alias for shutdown() method for compatibility.
        """
        await self.shutdown()
    
    async def shutdown(self) -> None:
        """Shutdown the ARK engine gracefully."""
        self.state = ARKState.SHUTDOWN
        self.ark_logger.info("ARK: Initiating shutdown sequence")
        
        # Close MCP client connections
        try:
            if hasattr(self.mcp_client, 'close'):
                await self.mcp_client.close()
            elif hasattr(self.mcp_client, 'disconnect_all'):
                await self.mcp_client.disconnect_all()
        except Exception as e:
            self.ark_logger.warning(f"ARK: Error during MCP client shutdown: {e}")
        
        # Export conversation if needed
        try:
            conversation_export = self.context_manager.export_conversation()
            self.ark_logger.debug("ARK: Conversation exported for archival")
        except Exception as e:
            self.ark_logger.warning(f"ARK: Error exporting conversation: {e}")
        
        self.ark_logger.info("ARK: Shutdown complete")