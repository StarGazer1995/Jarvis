"""
Deep Research Demo
"""

import asyncio
import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.agent.deep_research import DeepResearchAgent

# Configure logging
logging.basicConfig(level=logging.INFO)


async def main():
    print("Initializing Deep Research Agent...")

    # Configuration
    config = {
        "llm": {
            # Use environment variables for provider/model if not set here
            # Defaulting to OpenAI/GPT-4o logic if env vars present
            "temperature": 0.5,
            "max_tokens": 4096,
        },
        "max_steps": 15,
    }

    agent = DeepResearchAgent(config)

    if not await agent.initialize():
        print("Failed to initialize agent.")
        return

    query = "What are the key differences between the DeepResearch paper (Tongyi) and OpenAI's Deep Research in terms of architecture?"
    print(f"\nUser Query: {query}\n")
    print("-" * 50)

    response = await agent.process_input(query)

    print("-" * 50)
    print("\nFinal Answer:\n")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
