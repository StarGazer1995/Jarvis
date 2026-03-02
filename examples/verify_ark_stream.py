import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from unittest.mock import MagicMock
from src.core.ark.engine import ARKEngine, ARKState


class MockLLMManager:
    async def stream_response(self, messages, **kwargs):
        print("Mock LLM streaming...")
        tokens = [
            '{"thought": "Thinking',
            ' process", ',
            '"type": "answer", ',
            '"content": "Hello',
            ' World"}',
        ]
        for token in tokens:
            yield token
            await asyncio.sleep(0.01)  # Simulate delay


async def main():
    print("Initializing ARKEngine with mocks...")
    engine = ARKEngine()

    # Mock LLM Manager
    engine.llm_manager = MockLLMManager()

    # Mock MCP Client
    engine.mcp_client = MagicMock()
    engine.mcp_client.tools = []
    engine.mcp_client.sessions = {}  # For discover_tools if needed

    # Manually initialize graph to avoid full engine initialization overhead
    from src.core.ark.graph import create_ark_graph
    from src.core.ark.nodes.tools import ToolsNode

    tools_node = ToolsNode(engine.mcp_client)
    # Re-create graph with the mock LLM manager
    engine.graph = create_ark_graph(
        engine.llm_manager, engine.mcp_client, tools_node_instance=tools_node
    )

    # Set ready state
    engine.state = ARKState.READY

    print("Testing process_input with callbacks...")

    captured_thoughts = []
    captured_tokens = []

    def on_thought_token(token):
        print(f"[THOUGHT]: {token}")
        captured_thoughts.append(token)

    def on_token(token):
        print(f"[TOKEN]: {token}")
        captured_tokens.append(token)

    callbacks = {"on_thought_token": on_thought_token, "on_token": on_token}

    # Note: process_input calls graph.ainvoke, which calls MasterNode, which calls llm_manager.stream_response
    response = await engine.process_input("Test input", callbacks=callbacks)

    print(f"\nFinal Response: {response}")

    # Verification
    full_thought = "".join(captured_thoughts)
    full_token = "".join(captured_tokens)

    print(f"Captured Thoughts: '{full_thought}'")
    print(f"Captured Tokens: '{full_token}'")

    assert "Thinking process" == full_thought
    assert "Hello World" == full_token
    # response is now just the answer string
    assert response == "Hello World"

    print(
        "\nVerification SUCCESS: Callbacks were invoked correctly and CoT was separated."
    )


if __name__ == "__main__":
    asyncio.run(main())
