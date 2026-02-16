"""
ReAct Agent Implementation
"""
import json
import re
import logging
from typing import List, Dict, Any, Optional, Union

from .base import BaseAgent
from .types import AgentState, AgentStep
from ..llm.client import LLMMessage

class ReActAgent(BaseAgent):
    """
    Agent implementing the ReAct (Reasoning and Acting) framework.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_steps = self.config.get('max_steps', 15)
        self.logger = logging.getLogger('agent.react')
        
    async def process_input(self, user_input: str) -> str:
        """
        Process user input using the ReAct loop.
        """
        if self.state != AgentState.READY:
            return "Agent is not ready. Please wait for initialization."
            
        try:
            self.state = AgentState.PROCESSING
            self.logger.info(f"Processing input: '{user_input[:50]}...'")
            
            response = await self._run_loop(user_input)
            
            self.state = AgentState.READY
            return response
            
        except Exception as e:
            self.state = AgentState.ERROR
            self.logger.error(f"Error processing input: {e}")
            return f"Error: {str(e)}"
            
    async def _run_loop(self, user_input: str) -> str:
        """Execute the ReAct loop."""
        steps: List[AgentStep] = []
        
        # Initial message
        messages = self._build_initial_messages(user_input)
        
        for i in range(self.max_steps):
            self.logger.info(f"ReAct Step {i+1}/{self.max_steps}")
            
            # 1. Get LLM Response
            try:
                llm_response = await self.llm_manager.generate_response(messages)
                response_text = llm_response.content
            except Exception as e:
                return f"Error communicating with LLM: {e}"
                
            self.logger.debug(f"LLM Response: {response_text}")
            
            # Clean up response (remove any potential previous tool response if duplicated)
            if '<tool_response>' in response_text:
                response_text = response_text.split('<tool_response>')[0]
                
            messages.append(LLMMessage(role="assistant", content=response_text))
            
            # 2. Parse Response
            # Check for Final Answer
            if '<answer>' in response_text and '</answer>' in response_text:
                answer = response_text.split('<answer>')[1].split('</answer>')[0]
                return answer.strip()
            
            # Check for Tool Calls
            tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', response_text, re.DOTALL)
            
            if tool_calls:
                self.logger.info(f"DEBUG: Found {len(tool_calls)} tool calls")
                # 3. Execute Actions (Parallel)
                import asyncio
                tasks = []
                for tc in tool_calls:
                    self.logger.info(f"DEBUG: Processing tool call: {tc}")
                    tasks.append(self._execute_single_tool_call(tc))
                
                results = await asyncio.gather(*tasks)
                self.logger.info(f"DEBUG: Tool results: {results}")
                
                # 4. Append Observation
                combined_response = "\n".join(results)
                tool_response_msg = f"<tool_response>\n{combined_response}\n</tool_response>"
                messages.append(LLMMessage(role="user", content=tool_response_msg))
                
                # Also track in steps for compatibility/logging if needed, though message history is primary now
                steps.append(AgentStep(
                    thought="Processed tool calls",
                    action="Multiple Tools",
                    action_input=tool_calls,
                    observation=combined_response
                ))
            else:
                # No tool call and no answer?
                # Check for just thought
                if '<thought>' in response_text and '</thought>' in response_text:
                    # Just thinking, let it continue or prompt it?
                    # Ideally the model should output thought AND tool_call OR answer.
                    # If it only outputs thought, we might need to nudge it.
                    # For now, we'll assume it might be a multi-step thought process or error.
                    # Let's add a system reminder if it seems stuck, or just continue.
                    pass
                else:
                    self.logger.warning("No valid XML tags found in response.")
                    # Optional: Add a user message prompting to use correct format?
        
        return "Reached maximum steps without finding a final answer."

    def _build_initial_messages(self, user_input: str) -> List[LLMMessage]:
        """Build the initial prompt messages."""
        system_prompt = self._get_system_prompt()
        return [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_input)
        ]

    async def _execute_single_tool_call(self, tool_call_str: str) -> str:
        """Parse and execute a single tool call string."""
        try:
            # Handle PythonInterpreter special format (JSON + <code>)
            if "<code>" in tool_call_str:
                json_part = tool_call_str.split("<code>")[0].strip()
                code_part = tool_call_str.split("<code>")[1].split("</code>")[0].strip()
                try:
                    tool_info = json.loads(json_part)
                    tool_name = tool_info.get("name")
                    # Special handling for PythonInterpreter if supported, 
                    # or pass code as argument if the tool expects it
                    return await self.execute_tool(tool_name, {"code": code_part, **tool_info.get("arguments", {})})
                except json.JSONDecodeError:
                    return "Error: Invalid JSON in tool call."
            
            # Standard JSON
            tool_call = json.loads(tool_call_str)
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments", {})
            
            return await self.execute_tool(tool_name, tool_args)
            
        except json.JSONDecodeError:
            return f"Error: Tool call is not valid JSON: {tool_call_str}"
        except Exception as e:
            return f"Error executing tool: {str(e)}"

    def _get_system_prompt(self) -> str:
        """Get the system prompt. Override in subclasses."""
        return """You are an AI agent using the ReAct framework.
Use the available tools to answer the user's request.

IMPORTANT: You must use the following XML format for your response.

1. To think (optional but recommended):
<thought>
Your reasoning here...
</thought>

2. To use a tool (you can use multiple tools in parallel):
<tool_call>
{"name": "tool_name", "arguments": {"arg1": "value1"}}
</tool_call>

3. To provide the final answer:
<answer>
Your final answer here...
</answer>
"""
    
    # Deprecated/Unused methods kept for compatibility if needed, or remove
    def _build_prompt(self, user_input: str, steps: List[AgentStep]) -> List[LLMMessage]:
         # This is replaced by message history maintenance in _run_loop
         return []

    def _parse_response(self, response_text: str) -> AgentStep:
        # This is replaced by inline parsing in _run_loop
        return AgentStep(thought="", action="", action_input={})

    async def execute_tool(self, name: str, params: Any) -> Any:
        """Execute a tool. Override in subclasses."""
        return f"Error: Tool execution not implemented for '{name}'"
