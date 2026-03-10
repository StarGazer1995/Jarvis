import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.core.agent.react import ReActAgent, AgentState


class MockLLMManager:
    async def stream_response(self, messages, **kwargs):
        tokens = [
            "## Reasoning",
            "\n",
            "Reasoning",
            "\n",
            "## Response",
            "\n",
            "Answer",
        ]
        for token in tokens:
            yield token
            await asyncio.sleep(0.01)


async def main():
    agent = ReActAgent()
    agent.llm_manager = MockLLMManager()
    agent.state = AgentState.READY

    captured_thoughts = []
    captured_tokens = []

    callbacks = {
        "on_thought_token": lambda t: captured_thoughts.append(t),
        "on_token": lambda t: captured_tokens.append(t),
    }

    response = await agent.process_input("test", callbacks=callbacks)

    print(f"Thoughts: {captured_thoughts}")
    print(f"Tokens: {captured_tokens}")
    print(f"Response: {response}")

    assert "Reasoning" in "".join(captured_thoughts)
    assert "Answer" in "".join(captured_tokens)
    print("ReActAgent Verification SUCCESS")


if __name__ == "__main__":
    asyncio.run(main())
