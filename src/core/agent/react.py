"""
ReAct Agent Implementation
"""
import json
import re
import logging
from typing import List, Dict, Any, Optional, Union, Callable

from .base import BaseAgent
from .types import AgentState, AgentStep
from ..llm.client import LLMMessage
from ..llm.converters import convert_langchain_to_llm_messages
from ..prompt.manager import PromptManager
from ..llm.parsers import JSONOutputParser

from ..llm.stream_handler import StreamTokenHandler

class ReActAgent(BaseAgent):
    """
    Agent implementing the ReAct (Reasoning and Acting) framework.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_steps = self.config.get('max_steps', 15)
        self.logger = logging.getLogger('agent.react')
        self.prompt_manager = PromptManager()
        self.parser = JSONOutputParser()
        
    async def process_input(self, user_input: str, callbacks: Optional[Dict[str, Callable]] = None, **kwargs) -> str:
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
            
    async def _run_loop(self, user_input: str, callbacks: Optional[Dict[str, Callable]] = None) -> str:
        """Execute the ReAct loop."""
        steps: List[AgentStep] = []
        
        # Initial message
        messages = self._build_initial_messages(user_input)
        
        for i in range(self.max_steps):
            self.logger.info(f"ReAct Step {i+1}/{self.max_steps}")
            
            # 1. Get LLM Response
            try:
                response_text = await self._generate_and_stream_response(messages, callbacks)
            except Exception as e:
                return f"Error communicating with LLM: {e}"
                
            self.logger.debug(f"LLM Response: {response_text}")
            
            # 2. Parse JSON Response
            try:
                parsed_response = self.parser.parse(response_text)
            except ValueError as e:
                self.logger.error(f"Failed to parse JSON response: {e}")
                messages.append(LLMMessage(role="assistant", content=response_text))
                messages.append(LLMMessage(role="user", content=f"Error: Invalid JSON output. Please output valid JSON matching the schema. Error: {e}"))
                continue

            thought = parsed_response.get("thought", "")
            msg_type = parsed_response.get("type", "answer")
            content = parsed_response.get("content", "")
            
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
                    
                    messages.append(LLMMessage(role="user", content=f"Observation: {result}"))
                    
                    steps.append(AgentStep(
                        thought=thought,
                        action=tool_name,
                        action_input=tool_args,
                        observation=str(result)
                    ))
                else:
                    messages.append(LLMMessage(role="user", content="Error: Tool call content must be a JSON object."))
            else:
                self.logger.warning(f"Unknown message type: {msg_type}")
                # Treat as continue?
                pass
        
        return "Reached maximum steps without finding a final answer."

    async def _generate_and_stream_response(self, messages: List[LLMMessage], callbacks: Optional[Dict[str, Callable]] = None) -> str:
        """
        Generate response with streaming and callbacks using StreamTokenHandler.
        Parses JSON during streaming.
        """
        handler = StreamTokenHandler(callbacks)
        return await handler.process_stream(self.llm_manager.stream_response(messages))

    def _build_initial_messages(self, user_input: str) -> List[LLMMessage]:
        """Build the initial prompt messages."""
        system_messages = self._get_system_prompt()
        llm_system_messages = convert_langchain_to_llm_messages(system_messages)
        
        return llm_system_messages + [LLMMessage(role="user", content=user_input)]

    async def execute_tool(self, name: str, params: Any) -> Any:
        """Execute a tool. Override in subclasses."""
        return f"Error: Tool execution not implemented for '{name}'"

    def _get_system_prompt(self) -> List[Any]:
        """Get the system prompt. Override in subclasses."""
        messages = self.prompt_manager.render_template("react_system")
        return messages

