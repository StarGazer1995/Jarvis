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
        
        for i in range(self.max_steps):
            self.logger.info(f"ReAct Step {i+1}/{self.max_steps}")
            
            # 1. Build Prompt
            messages = self._build_prompt(user_input, steps)
            
            # 2. Get LLM Response
            try:
                llm_response = await self.llm_manager.generate_response(messages)
                response_text = llm_response.content
            except Exception as e:
                return f"Error communicating with LLM: {e}"
                
            self.logger.debug(f"LLM Response: {response_text}")
            
            # 3. Parse Response
            step = self._parse_response(response_text)
            
            if step.action == "Final Answer":
                steps.append(step)
                return step.observation or "No final answer content provided."
                
            if step.action:
                # 4. Execute Action
                observation = await self.execute_tool(step.action, step.action_input)
                step.observation = str(observation)
                steps.append(step)
            else:
                # No action, just thought or error
                if not step.thought:
                     step.observation = "Error: Invalid format. Please provide 'Action:' and 'Action Input:' or 'Final Answer:'."
                steps.append(step)
                
        return "Reached maximum steps without finding a final answer."

    def _build_prompt(self, user_input: str, steps: List[AgentStep]) -> List[LLMMessage]:
        """Build the prompt messages."""
        system_prompt = self._get_system_prompt()
        
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=f"User Request: {user_input}")
        ]
        
        # Add history
        history_text = ""
        for step in steps:
            history_text += f"Thought: {step.thought}\n"
            if step.action:
                history_text += f"Action: {step.action}\n"
                input_str = json.dumps(step.action_input, ensure_ascii=False) if isinstance(step.action_input, (dict, list)) else str(step.action_input)
                history_text += f"Action Input: {input_str}\n"
                history_text += f"Observation: {step.observation}\n\n"
            else:
                history_text += f"Observation: {step.observation}\n\n"
                
        if history_text:
            messages.append(LLMMessage(role="assistant", content=history_text))
            
        return messages

    def _parse_response(self, response_text: str) -> AgentStep:
        """Parse LLM response into an AgentStep."""
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction|\nFinal Answer|$)", response_text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        
        final_answer_match = re.search(r"Final Answer:\s*(.*)", response_text, re.DOTALL)
        if final_answer_match:
            return AgentStep(
                thought=thought,
                action="Final Answer",
                observation=final_answer_match.group(1).strip()
            )
            
        action_match = re.search(r"Action:\s*(.*?)\n", response_text)
        action_input_match = re.search(r"Action Input:\s*(.*)", response_text, re.DOTALL)
        
        if action_match and action_input_match:
            action = action_match.group(1).strip()
            input_str = action_input_match.group(1).strip()
            
            # Clean up input
            if input_str.startswith("```"):
                input_str = re.sub(r"^```\w*\n|```$", "", input_str).strip()
            elif input_str.startswith("`"):
                input_str = input_str.strip("`")
                
            try:
                action_input = json.loads(input_str)
            except json.JSONDecodeError:
                action_input = input_str
                
            return AgentStep(thought=thought, action=action, action_input=action_input)
            
        return AgentStep(thought=thought, observation="Error: Could not parse Action and Action Input.")

    def _get_system_prompt(self) -> str:
        """Get the system prompt. Override in subclasses."""
        return """You are an AI agent using the ReAct framework.
Use the available tools to answer the user's request.

Format:
Thought: ...
Action: ...
Action Input: ...
Observation: ...

OR

Thought: ...
Final Answer: ...
"""

    async def execute_tool(self, name: str, params: Any) -> Any:
        """Execute a tool. Override in subclasses."""
        return f"Error: Tool execution not implemented for '{name}'"
