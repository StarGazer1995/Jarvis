import json
import re
import logging
from typing import Dict, Any, List, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.runnables import RunnableConfig

from ..state import JarvisState
from ...llm.client import LLMManager, LLMMessage
from ...agent.types import AgentStep
from ..utils import clean_llm_response, AgentSpec

logger = logging.getLogger("ark.nodes.master")

class MasterNode:
    """
    The main reasoning node (Agent) for Jarvis.
    Merges responsibilities of Supervisor (Scheduling) and Executor (Tools).
    """
    def __init__(self, llm_manager: LLMManager, agents: Optional[List[AgentSpec]] = None):
        self.llm_manager = llm_manager
        self.agents = agents or []
        self.agent_map = {a.name: a for a in self.agents}

    async def __call__(self, state: JarvisState, config: RunnableConfig) -> Dict[str, Any]:
        """
        Execute the master agent logic.
        """
        # 1. Convert State Messages to LLM Messages
        messages = self._convert_messages(state["messages"])
        
        # 2. Dynamic System Prompt Injection
        # We construct the system prompt based on the current state (e.g. todo_list)
        system_prompt = self._get_system_prompt(state)
        
        # Check if the first message is system; if so, update it; otherwise insert it
        if messages and messages[0].role == "system":
            messages[0].content = system_prompt
        else:
            messages.insert(0, LLMMessage(role="system", content=system_prompt))
            
        # 3. Call LLM
        logger.debug("MasterNode calling LLM...")
        try:
            response = await self.llm_manager.generate_response(messages)
            raw_content = response.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raw_content = f"Error: Failed to generate response from LLM: {e}"
        
        logger.debug(f"LLM Response (Raw): {raw_content[:100]}...")
        
        # 4. Clean Response (<think> tags)
        content = clean_llm_response(raw_content)
        
        # 5. Parse Response for Tools OR Agents
        tool_call = self._parse_tool_call(content)
        
        if tool_call:
            # Ensure args is a dictionary for Pydantic validation if possible
            tool_args = tool_call["action_input"]
            if not isinstance(tool_args, dict):
                # Try to force it into a dict if it's a string, or wrap it
                try:
                    if isinstance(tool_args, str) and tool_args.strip().startswith("{"):
                         tool_args = json.loads(tool_args)
                except:
                    pass
            
            # Construct AIMessage with tool_calls
            lc_tool_call = {
                "name": tool_call["action"],
                "args": tool_args,
                "id": f"call_{len(state['messages'])}" # Simple ID generation
            }
            
            return {
                "messages": [AIMessage(content=content, tool_calls=[lc_tool_call])],
                "sender": "master"
            }
        else:
            # Normal response (Thought or Final Answer)
            return {
                "messages": [AIMessage(content=content)],
                "sender": "master"
            }

    def _convert_messages(self, lc_messages: List[BaseMessage]) -> List[LLMMessage]:
        """Convert LangChain messages to internal LLMMessage format."""
        out = []
        for m in lc_messages:
            role = "user"
            if isinstance(m, AIMessage):
                role = "assistant"
            elif isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, HumanMessage):
                role = "user"
            # Skip ToolMessages here as they should be incorporated into history 
            # differently depending on LLM provider support.
            # For ReAct (text-based), we append them as Observation text.
            # But here we are building a list of messages.
            # If the previous message was an AI message with tool_calls, 
            # the LLM expects the tool output next.
            # Since our LLMManager is generic text-based (mostly), we might need to 
            # flatten ToolMessages into the previous User/Assistant message or 
            # send them as "user" role with "Observation: ..." prefix.
            
            # For simplicity in this ReAct implementation:
            # We treat ToolMessage as User message saying "Observation: ..."
            
            content = m.content
            if m.type == "tool":
                role = "user"
                content = f"Observation: {content}"
            elif m.name and m.name in self.agent_map:
                 # Worker Output
                 content = f"[WORKER OUTPUT from {m.name}]:\n{content}"
                 role = "user"
            
            out.append(LLMMessage(role=role, content=str(content)))
        return out

    def _get_system_prompt(self, state: JarvisState) -> str:
        """Generate the system prompt based on state."""
        todo_list = state.get("todo_list", [])
        
        # 1. Todo List Status
        todo_status = "No tasks in todo list."
        if todo_list:
            todo_status = "Current Todo List:\n"
            for task in todo_list:
                status = task.get("status", "pending")
                desc = task.get("description", "")
                result = task.get("result", "")
                task_id = task.get("id", "")
                
                todo_status += f"- [{task_id}] {status}: {desc}"
                if result:
                    todo_status += f" (Result: {result})"
                todo_status += "\n"
        
        # 2. Available Tools
        tools_desc = "Available Tools:\n"
        available_tools = state.get("available_tools", {})
        if available_tools:
            for name, info in available_tools.items():
                desc = info.get("description", "No description")
                schema = json.dumps(info.get("input_schema", {}), indent=2)
                tools_desc += f"- {name}: {desc}\n  Schema: {schema}\n"
        else:
            tools_desc += "No tools available.\n"
            
        # 3. Available Agents (Workers)
        agents_desc = ""
        if self.agents:
            agents_desc = "Available Worker Agents:\n"
            for agent in self.agents:
                agents_desc += f"- {agent.name}: {agent.description}\n"
        
        return f"""You are Jarvis, an intelligent agent acting as an Orchestrator.

{todo_status}

{tools_desc}

{agents_desc}

Instructions:
1. Analyze the user's request.
2. Break it down into a list of tasks using 'manage_tasks' if needed.
3. Schedule execution by delegating to Worker Agents or using Tools.
   - To delegate to a Worker Agent, use 'Action: <AgentName>' with appropriate input.
4. Execute tasks one by one.
5. Update task status as you progress.
6. When finished, provide a Final Answer.

Format your response as follows:

Thought: <your reasoning>
Action: <tool_name_or_agent_name>
Action Input: <json_or_string_input>

OR

Thought: <your reasoning>
Final Answer: <your final response to the user>
"""

    def _parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response for ReAct pattern."""
        # Regex adapted from ReActAgent
        action_match = re.search(r"Action:\s*(.*?)\n", text)
        action_input_match = re.search(r"Action Input:\s*(.*)", text, re.DOTALL)
        
        if action_match and action_input_match:
            action = action_match.group(1).strip()
            # Clean up action name (take first word, remove trailing punctuation)
            action = action.split()[0].rstrip(",.:")
            
            input_str = action_input_match.group(1).strip()
            
            # Clean up input
            if input_str.startswith("```"):
                input_str = re.sub(r"^```\w*\n|```$", "", input_str).strip()
            elif input_str.startswith("`"):
                input_str = input_str.strip("`")
                
            try:
                action_input = json.loads(input_str)
            except json.JSONDecodeError:
                # Attempt to find JSON substring if parsing failed
                try:
                    # Find first {
                    start = input_str.find("{")
                    if start != -1:
                        # Try to decode just the JSON object part
                        try:
                            decoder = json.JSONDecoder()
                            action_input, _ = decoder.raw_decode(input_str[start:])
                        except json.JSONDecodeError:
                            # Fallback to finding the last } if raw_decode fails
                            end = input_str.rfind("}")
                            if end != -1 and end > start:
                                json_str = input_str[start:end+1]
                                action_input = json.loads(json_str)
                            else:
                                raise
                    else:
                        # Fallback to error dict or string
                        # If action is an agent, input might be string (task description)
                        # Let's handle plain string input if JSON fail
                        action_input = input_str
                except:
                     action_input = input_str
            
            # If action_input is just a string, wrap it if needed or return as is?
            # ReActAgent logic returned dict or string.
            # MasterNode logic expects dict usually for tools.
            # If we allow string for agents, we should return it.
                 
            return {
                "action": action,
                "action_input": action_input
            }
        return None
