"""
LangGraph ARK Engine - Real LLM Example

This script demonstrates how to run the ARK Engine with a REAL LLM (OpenAI, Anthropic, etc.)
via environment variables.

Usage:
    export LLM_PROVIDER=openai  # or anthropic, litellm
    export LLM_MODEL=gpt-4o     # or claude-3-5-sonnet-20240620
    export OPENAI_API_KEY=sk-...

    python examples/langgraph_real.py
"""

import asyncio
import logging
import os
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.core.ark.engine import ARKEngine
from src.core.llm.types import LLMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("example.real")


async def main():
    # 1. Load Config from Environment
    try:
        llm_config = LLMConfig.from_env()
    except Exception as e:
        logger.error(f"Failed to load config from env: {e}")
        return

    logger.info(f"Initializing Engine with Provider: {llm_config.provider}")
    logger.info(f"Model: {llm_config.model}")

    if llm_config.provider.value == "mock":
        logger.warning(
            "WARNING: LLM_PROVIDER not set or set to 'mock'. This will use the Mock client!"
        )
        logger.warning(
            "To use a real LLM, set LLM_PROVIDER=openai (or litellm) and OPENAI_API_KEY."
        )

    # 2. Initialize Engine
    # We pass the LLM config directly to the engine's LLMManager during init
    # Note: ARKEngine initializes its own LLMManager with default_config=None.
    # We need to manually set it up before initialize() or rely on env vars if ARKEngine supported it.
    # ARKEngine's __init__ doesn't take llm_config, but its parent BaseAgent does?
    # Let's check BaseAgent... actually ARKEngine inherits ReActAgent -> BaseAgent.
    # BaseAgent initializes LLMManager.

    # Let's construct the engine
    engine = ARKEngine()

    # 3. Configure the LLM Manager with our real config
    # We add it as the 'default' client
    await engine.llm_manager.add_client("default", llm_config)

    # 4. Initialize Engine (Tools, Graph)
    if not await engine.initialize():
        logger.error("Engine initialization failed")
        return

    # 5. Run Interaction
    user_input = "Please add a task to 'Research LangGraph integration' and then tell me it's done."
    print(f"\nUser: {user_input}")
    print("-" * 50)

    response = await engine.process_input(user_input)

    print("-" * 50)
    print(f"Agent: {response}")

    # 6. Verify State
    status = engine.get_engine_status()
    todos = status["todo_list"]
    print(f"\nFinal Todos: {todos}")

    await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
