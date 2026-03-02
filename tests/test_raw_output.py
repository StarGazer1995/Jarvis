import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from langchain_core.messages import HumanMessage, AIMessage

from src.core.ark.engine import ARKEngine, ARKState


class TestARKEngineRawOutput(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Patch dependencies
        self.mcp_client_patcher = patch("src.core.ark.engine.ARKMCPClient")
        self.context_manager_patcher = patch("src.core.ark.engine.ConversationContext")
        self.prompt_manager_patcher = patch("src.core.ark.engine.PromptManager")

        self.mock_mcp_client_cls = self.mcp_client_patcher.start()
        self.mock_context_manager_cls = self.context_manager_patcher.start()
        self.mock_prompt_manager_cls = self.prompt_manager_patcher.start()

        self.mock_mcp_client = self.mock_mcp_client_cls.return_value
        self.mock_context_manager = self.mock_context_manager_cls.return_value
        self.mock_prompt_manager = self.mock_prompt_manager_cls.return_value

        # Patch create_ark_graph and ToolsNode
        self.create_graph_patcher = patch("src.core.ark.engine.create_ark_graph")
        self.tools_node_patcher = patch("src.core.ark.engine.ToolsNode")

        self.mock_create_graph = self.create_graph_patcher.start()
        self.mock_tools_node_cls = self.tools_node_patcher.start()

        self.engine = ARKEngine(config={"enable_llm": True})
        self.engine.state = ARKState.READY

    async def asyncTearDown(self):
        self.mcp_client_patcher.stop()
        self.context_manager_patcher.stop()
        self.prompt_manager_patcher.stop()
        self.create_graph_patcher.stop()
        self.tools_node_patcher.stop()

    async def test_process_input_returns_raw_output(self):
        """Test process_input returns raw content."""
        mock_graph = AsyncMock()
        self.engine.graph = mock_graph

        # Mock graph response with Markdown headers
        raw_content = "## Reasoning\nReasoning...\n## Response\nFinal Answer"
        mock_graph.ainvoke.return_value = {
            "messages": [
                HumanMessage(content="Question"),
                AIMessage(content=raw_content),
            ],
            "todo_list": [],
            "sender": "master",
        }

        response = await self.engine.process_input("Question")

        # Verify that response is EXACTLY the raw content (no cleaning)
        self.assertEqual(response, raw_content)
