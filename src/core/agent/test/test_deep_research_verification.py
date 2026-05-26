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
    """Verify that a single tool_calls response executes multiple tools."""

    # Setup Agent
    agent = DeepResearchAgent({"llm": {"provider": "mock"}})
    agent.llm_manager = mock_llm_manager
    agent.state = AgentState.READY

    # Mock LLM response with multiple tool calls using JSON Protocol
    async def async_gen(content):
        yield content

    # Mock stream_response to return JSON strings
    mock_llm_manager.stream_response.side_effect = [
        # First response: parallel tool_calls
        async_gen(
            '{"thought": "Search in parallel", "type": "tool_calls", "content": [{"name": "search", "arguments": {"query": ["topic 1"]}}, {"name": "search", "arguments": {"query": ["topic 2"]}}]}'
        ),
        # Second response: Answer
        async_gen('{"thought": "Done", "type": "answer", "content": "Done"}'),
    ]

    # Mock Tools
    agent.tools = MagicMock()
    # Ensure AsyncMock is used for async methods
    agent.tools.search = AsyncMock(side_effect=["Result 1", "Result 2"])

    # Run
    await agent.process_input("test")

    # Verify both tool calls executed from a single protocol response.
    assert agent.tools.search.call_count == 2
    assert (
        agent.last_raw_response
        == '{"thought": "Done", "type": "answer", "content": "Done"}'
    )
    assert len(agent.raw_response_history) == 2
    assert agent.parsed_response_history[0]["type"] == "tool_calls"
    assert agent.parsed_response_history[1]["type"] == "answer"


@pytest.mark.asyncio
async def test_error_protocol_response_short_circuits(mock_llm_manager):
    """Verify formal error responses are surfaced consistently."""

    agent = DeepResearchAgent({"llm": {"provider": "mock"}})
    agent.llm_manager = mock_llm_manager
    agent.state = AgentState.READY

    async def async_gen(content):
        yield content

    mock_llm_manager.stream_response.return_value = async_gen(
        '{"thought": "Invalid request", "type": "error", "content": {"code": "INVALID_TOOL_ARGUMENTS", "message": "bad args"}}'
    )

    response = await agent.process_input("test")

    assert response == "Error (INVALID_TOOL_ARGUMENTS): bad args"
    assert agent.last_parsed_response is not None
    assert agent.last_parsed_response["type"] == "error"
    assert agent.raw_response_history[-1].startswith('{"thought": "Invalid request"')


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
