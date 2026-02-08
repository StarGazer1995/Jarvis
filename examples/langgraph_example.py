"""
LangGraph ARK Engine Example (Real Agent)

This script demonstrates how to initialize and run the LangGraph-based ARK Engine
with a REAL LLM configuration loaded from config/llm_config.yaml.

It replaces the previous Mock example to provide a genuine execution flow.
"""

import asyncio
import logging
import os
import sys
from typing import Any, Dict

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.core.ark.engine import ARKEngine, Task, TaskStatus
from src.core.config.loader import load_llm_config as load_yaml_config
from src.core.config.loader import LLMConfig as YamlLLMConfig, ProviderConfig, ModelConfig
from src.core.llm.types import LLMConfig as ClientLLMConfig, LLMProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("example.langgraph")

def convert_to_client_config(yaml_config: YamlLLMConfig) -> ClientLLMConfig:
    """
    Convert the loaded YAML configuration into the Client LLMConfig format.
    """
    # 1. Determine active provider
    provider_name = yaml_config.global_config.default_provider
    logger.info(f"Selected Provider: {provider_name}")
    
    if provider_name not in yaml_config.providers:
        raise ValueError(f"Provider '{provider_name}' not found in configuration")
        
    provider_cfg: ProviderConfig = yaml_config.providers[provider_name]
    
    if not provider_cfg.enabled:
        raise ValueError(f"Provider '{provider_name}' is disabled in configuration")

    # 2. Determine model
    model_name = provider_cfg.default_model
    logger.info(f"Selected Model: {model_name}")
    
    model_cfg: ModelConfig = provider_cfg.models.get(model_name)
    if not model_cfg:
        logger.warning(f"Model config for '{model_name}' not found, using defaults")
        model_cfg = ModelConfig()

    # 3. Construct Client LLMConfig
    try:
        provider_type = LLMProvider(provider_cfg.type)
    except ValueError:
        provider_type = provider_cfg.type
        
    client_config = ClientLLMConfig(
        provider=provider_type,
        model=model_name,
        api_key=provider_cfg.api_key,
        base_url=provider_cfg.base_url,
        temperature=model_cfg.temperature,
        max_tokens=model_cfg.max_tokens,
        timeout=provider_cfg.timeout.total if provider_cfg.timeout else 30.0,
        retry_attempts=provider_cfg.retry.max_attempts if provider_cfg.retry else 3,
        stream=yaml_config.features.streaming_enabled,
        extra_params={
            "top_p": model_cfg.top_p,
            "frequency_penalty": model_cfg.frequency_penalty,
            "presence_penalty": model_cfg.presence_penalty
        },
        provider_name=provider_name  # Important for correct error messages
    )
    
    return client_config

async def main():
    logger.info("Starting LangGraph ARK Example (Real Agent)...")
    
    try:
        # Load configuration (Default to production to use nvidia_nim as per recent fixes)
        # We respect JARVIS_ENV if set, otherwise default to production
        env = os.getenv("JARVIS_ENV", "production")
        logger.info(f"Loading configuration for environment: {env}")
        
        yaml_config = load_yaml_config(environment=env)
        client_config = convert_to_client_config(yaml_config)
        
        logger.info(f"Config loaded for: {client_config.provider_name} ({client_config.model})")
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    # Initialize Engine
    # Note: ARKEngine init might log errors for other missing providers in default config,
    # but as long as we inject our valid client_config, it should work for the main task.
    engine = ARKEngine()
    
    # Configure the LLM Manager with our loaded config
    # We must update the _default_config so that engine.initialize() uses OUR config,
    # not the default one loaded by BaseAgent.
    engine.llm_manager._default_config = client_config
    
    # Also add it explicitly (optional, but good for clarity)
    await engine.llm_manager.add_client("default", client_config)
    
    # Manually register the 'manage_tasks' tool so the LLM knows about it
    # ARKEngine usually discovers tools from MCP, but for this standalone example,
    # we inject the tool definition manually.
    engine.available_tools["manage_tasks"] = {
        "name": "manage_tasks",
        "description": "Manage the todo list. Actions: 'add' (requires description), 'update' (requires id, optional status/result), 'complete' (requires id).",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "update", "complete"]},
                "description": {"type": "string"},
                "id": {"type": "string"},
                "status": {"type": "string"},
                "result": {"type": "string"}
            },
            "required": ["action"]
        }
    }
    
    # Initialize Engine (Tools, Graph)
    # We must initialize FIRST to create the ToolsNode
    if not await engine.initialize():
        logger.error("Engine initialization failed")
        return

    # ---------------------------------------------------------
    # NEW: Register a local 'buy_item' tool to complete the task
    # ---------------------------------------------------------
    
    # 1. Define the actual python function
    def buy_item(item_name: str, quantity: int = 1):
        """Simulate buying an item."""
        logger.info(f"*** MOCK SHOPPING: Buying {quantity} x {item_name} ***")
        return f"Successfully purchased {quantity} {item_name}."

    # 2. Register the function logic in the Engine's ToolsNode
    # Note: We rely on the fact that we modified ARKEngine to expose tools_node
    if hasattr(engine, 'tools_node'):
        engine.tools_node.register_tool("buy_item", buy_item)
    else:
        logger.warning("Could not register local tool function: tools_node not found on engine")

    # 4. Implement and register manage_tasks local tool
    def manage_tasks(action: str, description: str = None, id: str = None, status: str = None, result: str = None):
        """Manage the todo list."""
        logger.info(f"*** MOCK MANAGE TASKS: {action} ***")
        
        if action == "add":
            if not description:
                return "Error: Description required for adding task."
            task_id = str(len(engine.todo_list) + 1)
            task = Task(
                id=task_id,
                description=description,
                status=TaskStatus.PENDING
            )
            engine.todo_list.append(task)
            logger.info(f"Task added: [{task_id}] {description}")
            return f"Task added: [{task_id}] {description}"
            
        elif action == "update":
            task_id = str(id)
            task = next((t for t in engine.todo_list if t.id == task_id), None)
            if not task:
                logger.error(f"Error: Task {task_id} not found.")
                return f"Error: Task {task_id} not found."
            
            if status:
                try:
                    task.status = TaskStatus(status)
                except ValueError:
                    pass
            if result:
                task.result = result
                logger.info(f"Task {task_id} updated.")
            return f"Task {task_id} updated."
            
        elif action == "complete":
            task_id = str(id)
            task = next((t for t in engine.todo_list if t.id == task_id), None)
            if not task:
                logger.error(f"Error: Task {task_id} not found.")
                return f"Error: Task {task_id} not found."
            task.status = TaskStatus.COMPLETED
            if result:
                task.result = result
                logger.info(f"Task {task_id} completed.")
            return f"Task {task_id} completed."
        logger.error(f"Error: Unknown action {action}.")
        return f"Error: Unknown action {action}."

    if hasattr(engine, 'tools_node'):
        engine.tools_node.register_tool("manage_tasks", manage_tasks)

    # 3. Register the tool definition (Metadata) so LLM knows it exists
    # Note: engine.initialize() resets available_tools, so we must add this AFTER initialize
    engine.available_tools["buy_item"] = {
        "name": "buy_item",
        "description": "Buy an item from the store.",
        "input_schema": {
            "type": "object",
            "properties": {
                "item_name": {"type": "string", "description": "The name of the item to buy"},
                "quantity": {"type": "integer", "description": "Number of items to buy", "default": 1}
            },
            "required": ["item_name"]
        }
    }
    
    # Re-register manage_tasks because initialize() wiped it
    engine.available_tools["manage_tasks"] = {
        "name": "manage_tasks",
        "description": "Manage the todo list. Actions: 'add' (requires description), 'update' (requires id, optional status/result), 'complete' (requires id).",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "update", "complete"]},
                "description": {"type": "string"},
                "id": {"type": "string"},
                "status": {"type": "string"},
                "result": {"type": "string"}
            },
            "required": ["action"]
        }
    }
    
    logger.info("Registered local tool: manage_tasks")
    logger.info("Registered local tool: buy_item")

    # Run Real Interaction
    # We ask a question that requires both planning (adding a task) and execution (doing it).
    # This demonstrates the full "Plan -> Execute -> Complete" cycle.
    user_input = "Please add a task to buy 5 cartons of milk and then execute it using the 'buy_item' tool."
    print(f"\nUser: {user_input}")
    print("-" * 50)
    
    response = await engine.process_input(user_input)
    
    print("-" * 50)
    print(f"Agent: {response}")
    
    # Verify State
    print("\n--- Engine Status ---")
    status = engine.get_engine_status()
    todos = status["todo_list"]
    print(f"Todos: {todos}")
    
    if len(todos) > 0:
        print(f"\nTask Status: {todos[-1]['description']} - {todos[-1]['status']}")
    else:
        print("\nNo tasks were added (Agent might have just replied textually).")

    await engine.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
