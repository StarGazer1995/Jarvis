# Example: Real LLM Streaming with ARKEngine
#
# This script demonstrates how to use the ARKEngine to stream responses from a real LLM provider (e.g., OpenAI, NVIDIA NIM).
# It sets up the engine, configures the LLM client, and processes a user request with real-time streaming output to the console.

import asyncio
import sys
import os
import logging

# Add project root to path
sys.path.append(os.getcwd())

from src.core.ark.engine import ARKEngine
from src.core.config.loader import load_llm_config as load_yaml_config
from src.core.llm.config import convert_to_client_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# Suppress noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


async def main():
    print("\n=== ARKEngine Real LLM Streaming Demo ===\n")

    # 1. Configuration
    # You can load from env vars or define manually.
    # Ensure OPENAI_API_KEY or NVIDIA_API_KEY is set in your environment.

    # Load configuration from YAML file
    try:
        env = os.getenv("JARVIS_ENV", "production")
        yaml_config = load_yaml_config(environment=env)
        client_config = convert_to_client_config(yaml_config)
        logging.info(
            f"Loaded config for: {client_config.provider_name}/{client_config.model}"
        )
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # 2. Initialize Engine
    print(f"Initializing Engine with model: {client_config.model}...")
    engine = ARKEngine()

    # Initialize LLM Manager with our config
    # We explicitly add the client from the loaded config
    await engine.llm_manager.add_client("default", client_config)
    engine.llm_manager.default_client = "default"

    # Initialize the rest of the engine (MCP, Graph, etc.)
    # We pass empty list for mcp_servers to skip connecting to real tools for this simple demo
    if not await engine.initialize(mcp_servers=[]):
        print("Failed to initialize engine.")
        return

    print("Engine initialized. Ready for input.\n")

    # 3. Define Streaming Callbacks
    # These functions will be called by the engine as tokens arrive

    def on_token(token: str):
        # Print regular response tokens immediately
        print(token, end="", flush=True)

    def on_thought_start():
        # Visual indicator for start of reasoning
        print(
            "\n\033[90m[Thinking process started]\033[0m\n\033[90m", end="", flush=True
        )

    def on_thought_token(token: str):
        # Print reasoning tokens in gray
        print(token, end="", flush=True)

    def on_thought_end():
        # Visual indicator for end of reasoning
        print("\033[0m\n\033[90m[Thinking process ended]\033[0m\n", flush=True)

    callbacks = {
        "on_token": on_token,
        "on_thought_start": on_thought_start,
        "on_thought_token": on_thought_token,
        "on_thought_end": on_thought_end,
    }

    # 4. Process Input
    user_query = "Explain the concept of 'Recursion' in programming with a simple Python example."
    print(f"User: {user_query}\n")
    print("Assistant: ", end="", flush=True)

    try:
        # The engine will stream output via callbacks and return the final complete string
        await engine.process_input(user_query, callbacks=callbacks)
        print("\n\n[Stream Completed]")

    except Exception as e:
        print(f"\nError during processing: {e}")

    # 5. Cleanup
    await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
