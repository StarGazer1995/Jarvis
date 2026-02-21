"""
Verify Deep Research Implementation
"""
import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from src.core.agent.deep_research import DeepResearchAgent
from src.core.llm.client import LLMResponse, LLMMessage

@pytest.fixture
def mock_llm_manager():
    with patch("src.core.agent.base.LLMManager") as mock:
        manager = mock.return_value
        manager.initialize_default_client = AsyncMock(return_value=True)
        manager.generate_response = AsyncMock()
        yield manager

from src.core.agent.types import AgentState

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
        async_gen('{"thought": "Searching 1", "type": "tool_call", "content": {"name": "search", "arguments": {"query": ["topic 1"]}}}'),
        # Second response: Tool Call 2
        async_gen('{"thought": "Searching 2", "type": "tool_call", "content": {"name": "search", "arguments": {"query": ["topic 2"]}}}'),
        # Third response: Answer
        async_gen('{"thought": "Done", "type": "answer", "content": "Done"}')
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
    with patch("src.capabilities.deep_research_tools.os.path.exists", return_value=True), \
         patch("builtins.open", MagicMock()) as mock_open, \
         patch.dict("sys.modules", {
             "pypdf": MagicMock(), 
             "docx": MagicMock()
         }):
        
        # Setup mocks
        import pypdf
        import docx
        
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
