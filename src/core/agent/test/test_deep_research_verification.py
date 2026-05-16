"""
Verify Deep Research Implementation
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.agent.deep_research import DeepResearchAgent
from src.core.agent.types import AgentState


@pytest.fixture
def mock_llm_manager():
    with patch("src.core.agent.base.LLMManager") as mock:
        manager = mock.return_value
        manager.initialize_default_client = AsyncMock(return_value=True)
        manager.generate_response = AsyncMock()
        yield manager


@pytest.mark.asyncio
async def test_parallel_tool_execution(mock_llm_manager):
    """Verify that multiple tool calls are executed in parallel."""

    # Setup Agent
    agent = DeepResearchAgent({"llm": {"provider": "mock"}})
    agent.llm_manager = mock_llm_manager
    agent.state = AgentState.READY

    # Mock LLM response with multiple tool calls using JSON Protocol
    async def async_gen(content):
        yield content

    # Mock stream_response to return JSON strings
    mock_llm_manager.stream_response.side_effect = [
        # First response: Tool Call 1
        async_gen(
            '{"thought": "Searching 1", "type": "tool_call", "content": {"name": "search", "arguments": {"query": ["topic 1"]}}}'
        ),
        # Second response: Tool Call 2
        async_gen(
            '{"thought": "Searching 2", "type": "tool_call", "content": {"name": "search", "arguments": {"query": ["topic 2"]}}}'
        ),
        # Third response: Answer
        async_gen('{"thought": "Done", "type": "answer", "content": "Done"}'),
    ]

    # Mock Tools
    agent.tools = MagicMock()
    # Ensure AsyncMock is used for async methods
    agent.tools.search = AsyncMock(side_effect=["Result 1", "Result 2"])

    # Run
    await agent.process_input("test")

    # Verify tools were called
    # In the new sequential loop (ReAct style), tools might be called sequentially if the LLM outputs them one by one.
    # The original test assumed parallel execution from a single response containing multiple tool calls.
    # If the LLM outputs one tool call per step, we will have 2 calls.
    assert agent.tools.search.call_count == 2


@pytest.mark.asyncio
async def test_file_parser_routing():
    """Verify file parser routes to correct library based on extension."""
    from src.capabilities.deep_research_tools import DeepResearchTools

    tools = DeepResearchTools()

    # We need to mock pypdf.PdfReader and docx.Document
    with (
        patch("src.capabilities.deep_research_tools.os.path.exists", return_value=True),
        patch("builtins.open", MagicMock()),
        patch.dict("sys.modules", {"pypdf": MagicMock(), "docx": MagicMock()}),
    ):
        # Setup mocks
        import docx
        import pypdf

        # Mock PDF
        mock_pdf_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PDF Content"
        mock_pdf_reader.pages = [mock_page]
        pypdf.PdfReader = MagicMock(return_value=mock_pdf_reader)

        # Mock DOCX
        mock_doc = MagicMock()
        mock_para = MagicMock()
        mock_para.text = "DOCX Content"
        mock_doc.paragraphs = [mock_para]
        docx.Document = MagicMock(return_value=mock_doc)

        # Test PDF
        result_pdf = await tools.parse_file(["test.pdf"])
        assert "PDF Content" in result_pdf
        pypdf.PdfReader.assert_called_with("test.pdf")

        # Test DOCX
        result_docx = await tools.parse_file(["test.docx"])
        assert "DOCX Content" in result_docx
        docx.Document.assert_called_with("test.docx")


class TestDeepResearchExecuteTool:
    """测试 DeepResearchAgent.execute_tool 的边界情况"""

    @pytest.mark.asyncio
    async def test_execute_tool_unknown(self):
        """测试执行未知工具"""
        agent = DeepResearchAgent({"llm": {"provider": "mock"}})
        agent.tools = MagicMock()
        result = await agent.execute_tool("unknown_tool", {})
        assert "Error: Unknown tool" in result

    @pytest.mark.asyncio
    async def test_execute_tool_exception(self):
        """测试工具执行抛出异常"""
        agent = DeepResearchAgent({"llm": {"provider": "mock"}})
        agent.tools = MagicMock()
        agent.tools.search = AsyncMock(side_effect=Exception("Search failed"))
        result = await agent.execute_tool("search", {"query": ["test"]})
        assert "Error executing tool" in result

    @pytest.mark.asyncio
    async def test_execute_tool_python_interpreter_no_code(self):
        """测试 PythonInterpreter 缺少 code 参数"""
        agent = DeepResearchAgent({"llm": {"provider": "mock"}})
        agent.tools = MagicMock()
        result = await agent.execute_tool("PythonInterpreter", {})
        assert "must provide 'code' argument" in result

    @pytest.mark.asyncio
    async def test_execute_tool_python_interpreter_success(self):
        """测试 PythonInterpreter 执行成功"""
        agent = DeepResearchAgent({"llm": {"provider": "mock"}})
        agent.tools = MagicMock()
        agent.tools.python_interpreter = AsyncMock(return_value="Execution result")
        result = await agent.execute_tool(
            "PythonInterpreter", {"code": "print('hello')"}
        )
        assert "Execution result" in result
        agent.tools.python_interpreter.assert_called_once_with("print('hello')")

    @pytest.mark.asyncio
    async def test_execute_tool_visit(self):
        """测试 visit 工具"""
        agent = DeepResearchAgent({"llm": {"provider": "mock"}})
        agent.tools = MagicMock()
        agent.tools.visit = AsyncMock(return_value="Page content")
        result = await agent.execute_tool(
            "visit", {"url": ["http://example.com"], "goal": "test"}
        )
        assert "Page content" in result
        agent.tools.visit.assert_called_once_with(["http://example.com"], "test")

    @pytest.mark.asyncio
    async def test_execute_tool_google_scholar(self):
        """测试 google_scholar 工具"""
        agent = DeepResearchAgent({"llm": {"provider": "mock"}})
        agent.tools = MagicMock()
        agent.tools.google_scholar = AsyncMock(return_value="Scholar results")
        result = await agent.execute_tool("google_scholar", {"query": ["AI"]})
        assert "Scholar results" in result

    @pytest.mark.asyncio
    async def test_execute_tool_parse_file(self):
        """测试 parse_file 工具"""
        agent = DeepResearchAgent({"llm": {"provider": "mock"}})
        agent.tools = MagicMock()
        agent.tools.parse_file = AsyncMock(return_value="File content")
        result = await agent.execute_tool("parse_file", {"files": ["test.pdf"]})
        assert "File content" in result
