"""
ARK (Autonomous Reasoning Kernel) Engine

This is the core reasoning engine that powers the Jarvis AI agent.
ARK integrates MCP tools, intent recognition, context management,
and decision-making capabilities.
"""

import logging
import asyncio
import json
import re
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from ..mcp.client import ARKMCPClient
from ..config.server import SimpleMCPServerConfig
from ..context.manager import ConversationContext
from ..llm.client import LLMManager, LLMMessage
from ..llm.config import LLMConfig, LLMProvider, load_llm_config
from ..prompt.manager import PromptManager


class ARKState(Enum):
    """ARK engine operational states."""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    TOOL_EXECUTION = "tool_execution"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class TaskStatus(Enum):
    """Status of a task in the todo list."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Represents a unit of work to be done."""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "result": self.result
        }


@dataclass
class ReActStep:
    """Represents a single step in the ReAct loop."""
    thought: str
    action: Optional[str] = None
    action_input: Optional[Union[Dict[str, Any], str]] = None
    observation: Optional[str] = None


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
        
        # Initialize LLM components
        llm_config_dict = self.config.get('llm', {})
        self.llm_config = load_llm_config(llm_config_dict) if llm_config_dict else load_llm_config()
        self.llm_manager = LLMManager(self.llm_config)
        self.prompt_manager = PromptManager()
        self.llm_enabled = self.config.get('enable_llm', True)
        
        # ARK-specific attributes
        self.available_tools: Dict[str, Any] = {}
        self.tool_usage_stats: Dict[str, int] = {}
        self.todo_list: List[Task] = []
        
        # Configuration
        self.max_tool_chain_length = self.config.get('max_tool_chain_length', 5)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.enable_tool_chaining = self.config.get('enable_tool_chaining', True)
        
        # Initialize performance tracking
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
            
            # Initialize LLM if enabled
            if self.llm_enabled:
                await self._initialize_llm()
            
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
        Process user input using the ReAct framework.
        
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
            
            # Use ReAct loop for processing
            response = await self._run_react_loop(user_input)
            
            # Update Context (simplified)
            self.context_manager.add_exchange(
                user_input=user_input,
                agent_response=response,
                intent="react_execution",
                tools_used=[t.description for t in self.todo_list], # simplified tracking
                metadata={"todo_count": len(self.todo_list)}
            )
            
            self.state = ARKState.READY
            self.ark_logger.info("ARK: Input processing complete")
            return response
            
        except Exception as e:
            self.state = ARKState.ERROR
            self.ark_logger.error(f"ARK: Error processing input: {e}")
            import traceback
            self.ark_logger.error(traceback.format_exc())
            return f"I encountered an error while processing your request: {str(e)}"

    async def _run_react_loop(self, user_input: str) -> str:
        """
        Execute the ReAct (Reason+Act) loop.
        """
        max_steps = 15
        steps: List[ReActStep] = []
        
        # If todo list is empty, we might want to clear it or keep it?
        # For now, we keep it across turns as requested by the user ("todo list to record tasks").
        
        for i in range(max_steps):
            self.ark_logger.info(f"ARK: ReAct Step {i+1}/{max_steps}")
            
            # 1. Build Prompt
            prompt_messages = self._build_react_prompt(user_input, steps)
            
            # 2. Get LLM Response
            try:
                llm_response = await self.llm_manager.generate_response(prompt_messages)
                response_text = llm_response.content
            except Exception as e:
                return f"Error communicating with LLM: {e}"
            
            self.ark_logger.debug(f"ARK: LLM Response: {response_text}")
            
            # 3. Parse Response
            # Expected format:
            # Thought: ...
            # Action: ...
            # Action Input: ...
            # OR
            # Final Answer: ...
            
            thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction|\nFinal Answer|$)", response_text, re.DOTALL)
            thought = thought_match.group(1).strip() if thought_match else "No thought provided."
            
            final_answer_match = re.search(r"Final Answer:\s*(.*)", response_text, re.DOTALL)
            if final_answer_match:
                final_answer = final_answer_match.group(1).strip()
                steps.append(ReActStep(thought=thought, action="Final Answer", observation=final_answer))
                return final_answer
            
            action_match = re.search(r"Action:\s*(.*?)\n", response_text)
            action_input_match = re.search(r"Action Input:\s*(.*)", response_text, re.DOTALL)
            
            if action_match and action_input_match:
                action_name = action_match.group(1).strip()
                action_input_str = action_input_match.group(1).strip()
                
                # Clean up action input (remove code blocks if present)
                if action_input_str.startswith("```"):
                    action_input_str = re.sub(r"^```\w*\n|```$", "", action_input_str).strip()
                elif action_input_str.startswith("`"):
                    action_input_str = action_input_str.strip("`")
                
                try:
                    # Try to parse JSON
                    action_input = json.loads(action_input_str)
                except json.JSONDecodeError:
                    # Fallback to string
                    action_input = action_input_str
                
                # 4. Execute Action
                observation = await self._execute_react_action(action_name, action_input)
                
                # Record Step
                steps.append(ReActStep(
                    thought=thought,
                    action=action_name,
                    action_input=action_input,
                    observation=str(observation)
                ))
                
            else:
                # If no action found but also no final answer, treat whole text as answer or ask for clarification?
                # Usually ReAct agents should output Action or Final Answer.
                # If it fails to follow format, we can append an observation telling it to format correctly.
                steps.append(ReActStep(
                    thought=response_text,
                    observation="Error: You must provide either an 'Action:' and 'Action Input:' or a 'Final Answer:'."
                ))
        
        return "I'm sorry, I reached the maximum number of steps without finding a final answer."

    def _build_react_prompt(self, user_input: str, steps: List[ReActStep]) -> List[LLMMessage]:
        """Build the prompt for the ReAct loop."""
        
        # System Prompt
        system_prompt = self._get_react_system_prompt()
        
        messages = [
            LLMMessage(role="system", content=system_prompt)
        ]
        
        # Add conversation history (simplified)
        # We could pull from context_manager, but for ReAct, the current loop history is more critical.
        # We can add recent user-assistant turns if needed.
        recent_history = self.context_manager.get_recent_turns(num_turns=5)
        for turn in reversed(recent_history):
            messages.append(LLMMessage(role="user", content=turn.user_input))
            messages.append(LLMMessage(role="assistant", content=turn.agent_response))
        
        # User Input
        messages.append(LLMMessage(role="user", content=f"User Request: {user_input}"))
        
        # ReAct History (Steps)
        history_text = ""
        for step in steps:
            history_text += f"Thought: {step.thought}\n"
            if step.action:
                history_text += f"Action: {step.action}\n"
                history_text += f"Action Input: {json.dumps(step.action_input, ensure_ascii=False) if isinstance(step.action_input, (dict, list)) else step.action_input}\n"
                history_text += f"Observation: {step.observation}\n\n"
            else:
                history_text += f"Observation: {step.observation}\n\n"
        
        if history_text:
            messages.append(LLMMessage(role="assistant", content=history_text))
            
        return messages

    def _get_react_system_prompt(self) -> str:
        """Generate the system prompt for ReAct agent."""
        
        # 1. Todo List Status
        todo_status = "No tasks in todo list."
        if self.todo_list:
            todo_status = "Current Todo List:\n"
            for task in self.todo_list:
                todo_status += f"- [{task.id}] {task.status.value}: {task.description}"
                if task.result:
                    todo_status += f" (Result: {task.result})"
                todo_status += "\n"
        
        # 2. Available Tools
        tools_desc = "Available Tools:\n"
        
        # Internal Tools
        tools_desc += "- manage_tasks: Manage the todo list. Input: {\"action\": \"add\"|\"update\"|\"complete\", ...}\n"
        
        # External Tools
        for name, info in self.available_tools.items():
            desc = info.get('description', 'No description')
            tools_desc += f"- {name}: {desc}\n"
            
        return f"""You are Jarvis, an intelligent agent using the ReAct framework.

{todo_status}

{tools_desc}

Instructions:
1. Analyze the user's request.
2. Break it down into a list of tasks using 'manage_tasks' if needed.
3. Execute tasks one by one.
4. Use available tools to gather information or perform actions.
5. Update task status as you progress.
6. When finished, provide a Final Answer.

Format your response as follows:

Thought: <your reasoning>
Action: <tool_name>
Action Input: <json_or_string_input>

OR

Thought: <your reasoning>
Final Answer: <your final response to the user>
"""

    async def _execute_react_action(self, action_name: str, action_input: Any) -> Any:
        """Execute an action (tool or internal)."""
        
        if action_name == "manage_tasks":
            if isinstance(action_input, str):
                try:
                    action_input = json.loads(action_input)
                except:
                    return "Error: Action Input for manage_tasks must be valid JSON."
            return self._manage_tasks(**action_input)
            
        # Check external tools
        if action_name in self.available_tools:
            # Need to ensure parameters is a dict
            params = action_input
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except:
                    # If string, maybe assume it's the primary arg? depends on tool.
                    # For now, return error if not dict
                    return f"Error: Tool arguments for {action_name} must be a JSON object."
            
            try:
                result = await self.mcp_client.execute_tool(action_name, params)
                return result
            except Exception as e:
                return f"Error executing tool {action_name}: {e}"
                
        return f"Error: Unknown tool '{action_name}'."

    def _manage_tasks(self, action: str, **kwargs) -> str:
        """
        Manage the todo list.
        
        Args:
            action: add, update, complete
            kwargs: task details
        """
        if action == "add":
            description = kwargs.get("description")
            if not description:
                return "Error: Description required for adding task."
            task_id = str(len(self.todo_list) + 1) # Simple ID
            task = Task(id=task_id, description=description)
            self.todo_list.append(task)
            return f"Task added: [{task_id}] {description}"
            
        elif action == "update":
            task_id = kwargs.get("id") or kwargs.get("task_id")
            status = kwargs.get("status")
            result = kwargs.get("result")
            
            task = next((t for t in self.todo_list if t.id == str(task_id)), None)
            if not task:
                return f"Error: Task {task_id} not found."
                
            if status:
                try:
                    task.status = TaskStatus(status)
                except ValueError:
                    return f"Error: Invalid status {status}."
            
            if result:
                task.result = result
                
            return f"Task {task_id} updated."
            
        elif action == "complete":
            task_id = kwargs.get("id") or kwargs.get("task_id")
            result = kwargs.get("result")
            
            task = next((t for t in self.todo_list if t.id == str(task_id)), None)
            if not task:
                return f"Error: Task {task_id} not found."
                
            task.status = TaskStatus.COMPLETED
            if result:
                task.result = result
                
            return f"Task {task_id} completed."
            
        return f"Error: Unknown action {action}."
    
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
    
    
    async def _initialize_llm(self) -> None:
        """Initialize LLM client and prompt manager."""
        try:
            # Initialize default LLM client
            success = await self.llm_manager.initialize_default_client()
            if success:
                self.ark_logger.info("ARK: LLM client initialized successfully")
            else:
                self.ark_logger.warning("ARK: LLM client initialization failed, using template fallback")
                self.llm_enabled = False
            
            # Load default prompts
            self.prompt_manager.load_default_templates()
            self.ark_logger.info("ARK: Prompt templates loaded successfully")
            
        except Exception as e:
            self.ark_logger.error(f"ARK: Error initializing LLM: {e}")
            self.llm_enabled = False
    
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
            "todo_count": len(self.todo_list)
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
            "todo_list": [t.to_dict() for t in self.todo_list],
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