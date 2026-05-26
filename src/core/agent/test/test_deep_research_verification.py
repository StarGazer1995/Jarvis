"""
Verify Deep Research Implementation
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.agent.deep_research import DeepResearchAgent
from src.core.agent.types import AgentState, AgentStep


@pytest.fixture
def mock_llm_manager():
    with patch("src.core.agent.base.LLMManager") as mock:
        manager = mock.return_value
        manager.initialize_default_client = AsyncMock(return_value=True)
        manager.generate_response = AsyncMock(
            return_value=MagicMock(
                content=(
                    '{"supported": true, "issues": [], "per_source": '
                    '[{"url": "https://example.com/topic", "supported": true, '
                    '"reason": "Observed evidence supports the cited excerpt."}]}'
                )
            )
        )
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
        async_gen(
            '{"thought": "Done", "type": "answer", "content": {"summary": "Done", "claims": [{"statement": "Topic searches completed.", "source_urls": ["https://example.com/topic"]}], "sources": [{"url": "https://example.com/topic", "title": "Topic source", "evidence": "The topic source documents the completed topic searches."}], "insufficient_evidence": false}}'
        ),
    ]

    # Mock Tools
    agent.tools = MagicMock()
    # Ensure AsyncMock is used for async methods
    agent.tools.search = AsyncMock(
        side_effect=[
            {
                "tool": "search",
                "results": [
                    {
                        "query": "topic 1",
                        "title": "Topic source",
                        "url": "https://example.com/topic",
                        "snippet": "The topic source documents the completed topic searches.",
                    }
                ],
            },
            {
                "tool": "search",
                "results": [
                    {
                        "query": "topic 2",
                        "title": "Topic source",
                        "url": "https://example.com/topic",
                        "snippet": "The topic source documents the completed topic searches.",
                    }
                ],
            },
        ]
    )

    # Run
    await agent.process_input("test")

    # Verify both tool calls executed from a single protocol response.
    assert agent.tools.search.call_count == 2
    assert (
        agent.last_raw_response
        == '{"thought": "Done", "type": "answer", "content": {"summary": "Done", "claims": [{"statement": "Topic searches completed.", "source_urls": ["https://example.com/topic"]}], "sources": [{"url": "https://example.com/topic", "title": "Topic source", "evidence": "The topic source documents the completed topic searches."}], "insufficient_evidence": false}}'
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
async def test_final_answer_requires_evidence_before_return(mock_llm_manager):
    """Verify unsupported answers are retried until evidence handling is explicit."""
    agent = DeepResearchAgent({"llm": {"provider": "mock"}})
    agent.llm_manager = mock_llm_manager
    agent.state = AgentState.READY

    async def async_gen(content):
        yield content

    mock_llm_manager.stream_response.side_effect = [
        async_gen(
            '{"thought": "I can answer now", "type": "answer", "content": {"summary": "OpenAI o1 costs $999.", "claims": [{"statement": "OpenAI o1 costs $999.", "source_urls": ["https://example.com/pricing"]}], "sources": [{"url": "https://example.com/pricing", "title": "Pricing", "evidence": "The pricing page lists OpenAI o1 at $999."}], "insufficient_evidence": false}}'
        ),
        async_gen(
            '{"thought": "Not enough verified evidence", "type": "answer", "content": {"summary": "Insufficient evidence to answer reliably.", "claims": [], "sources": [], "insufficient_evidence": true}}'
        ),
    ]

    response = await agent.process_input("price question")

    assert response == "Insufficient evidence to answer reliably."
    assert mock_llm_manager.stream_response.call_count == 2


@pytest.mark.asyncio
async def test_final_answer_requires_sources_for_factual_claims(mock_llm_manager):
    """Verify factual answers need a Sources section after tool-backed research."""
    agent = DeepResearchAgent({"llm": {"provider": "mock"}})
    agent.llm_manager = mock_llm_manager
    agent.state = AgentState.READY

    async def async_gen(content):
        yield content

    mock_llm_manager.stream_response.side_effect = [
        async_gen(
            '{"thought": "Need evidence", "type": "tool_call", "content": {"name": "search", "arguments": {"query": ["OpenAI o1 price"]}}}'
        ),
        async_gen(
            '{"thought": "I can answer now", "type": "answer", "content": {"summary": "OpenAI o1 costs $20.", "claims": [{"statement": "OpenAI o1 costs $20.", "source_urls": ["https://example.com/pricing"]}], "sources": [], "insufficient_evidence": false}}'
        ),
        async_gen(
            '{"thought": "Grounded answer", "type": "answer", "content": {"summary": "OpenAI o1 costs $20.", "claims": [{"statement": "OpenAI o1 costs $20.", "source_urls": ["https://example.com/pricing"]}], "sources": [{"url": "https://example.com/pricing", "title": "Pricing page", "evidence": "The pricing page states that OpenAI o1 costs $20."}], "insufficient_evidence": false}}'
        ),
    ]

    agent.tools = MagicMock()
    agent.tools.search = AsyncMock(
        return_value={
            "tool": "search",
            "results": [
                {
                    "query": "OpenAI o1 price",
                    "title": "Pricing",
                    "url": "https://example.com/pricing",
                    "snippet": "The pricing page states that OpenAI o1 costs $20.",
                }
            ],
        }
    )

    response = await agent.process_input("price question")

    assert "Sources:" in response
    assert "https://example.com/pricing" in response
    assert "Claims:" in response
    assert mock_llm_manager.stream_response.call_count == 3


@pytest.mark.asyncio
async def test_structured_answer_renders_summary_claims_and_sources(mock_llm_manager):
    """Verify structured answers are rendered into readable text."""
    agent = DeepResearchAgent({"llm": {"provider": "mock"}})
    agent.llm_manager = mock_llm_manager
    agent.state = AgentState.READY

    async def async_gen(content):
        yield content

    mock_llm_manager.stream_response.side_effect = [
        async_gen(
            '{"thought": "Need evidence", "type": "tool_call", "content": {"name": "search", "arguments": {"query": ["OpenAI o1 price"]}}}'
        ),
        async_gen(
            '{"thought": "Grounded answer", "type": "answer", "content": {"summary": "OpenAI o1 costs $20.", "claims": [{"statement": "OpenAI o1 costs $20.", "source_urls": ["https://example.com/pricing"]}], "sources": [{"url": "https://example.com/pricing", "title": "Pricing page", "evidence": "The pricing page states that OpenAI o1 costs $20."}], "insufficient_evidence": false}}'
        ),
    ]

    agent.tools = MagicMock()
    agent.tools.search = AsyncMock(
        return_value={
            "tool": "search",
            "results": [
                {
                    "query": "OpenAI o1 price",
                    "title": "Pricing",
                    "url": "https://example.com/pricing",
                    "snippet": "The pricing page states that OpenAI o1 costs $20.",
                }
            ],
        }
    )

    response = await agent.process_input("price question")

    assert "OpenAI o1 costs $20." in response
    assert "Claims:" in response
    assert "Sources:" in response
    assert "[Sources: https://example.com/pricing]" in response
    assert "Pricing page: https://example.com/pricing" in response
    assert "Evidence: The pricing page states that OpenAI o1 costs $20." in response


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


@pytest.mark.asyncio
async def test_search_excludes_generated_quick_answer():
    """Verify Tavily quick answers are not injected into observations."""
    from src.capabilities.deep_research_tools import DeepResearchTools

    tools = DeepResearchTools()
    tools.tavily_client = MagicMock()
    tools.tavily_client.search.return_value = {
        "answer": "Generated summary that should not be surfaced.",
        "results": [
            {
                "title": "Example Result",
                "url": "https://example.com",
                "content": "Evidence snippet.",
            }
        ],
    }

    result = await tools.search(["test topic"])

    assert result["tool"] == "search"
    assert result["results"][0]["url"] == "https://example.com"
    assert "answer" not in result["results"][0]
    tools.tavily_client.search.assert_called_once_with(
        query="test topic",
        search_depth="advanced",
        max_results=5,
        include_answer=False,
    )


@pytest.mark.asyncio
async def test_extract_info_returns_validated_structured_output():
    """Verify extractor output is parsed into validated structured evidence."""
    from src.capabilities.deep_research_tools import DeepResearchTools

    llm_manager = MagicMock()
    llm_manager.generate_response = AsyncMock(
        return_value=MagicMock(
            content='{"rational": "Matched pricing section", "evidence": "Price is $10.", "summary": "The page lists the current price."}'
        )
    )
    tools = DeepResearchTools(llm_manager=llm_manager)

    result = await tools._extract_info("webpage content", "find pricing")

    assert result["rational"] == "Matched pricing section"
    assert result["evidence"] == "Price is $10."
    assert result["summary"] == "The page lists the current price."


@pytest.mark.asyncio
async def test_search_returns_configuration_error_without_tavily():
    """Verify search returns a structured configuration error without Tavily."""
    from src.capabilities.deep_research_tools import DeepResearchTools

    tools = DeepResearchTools()
    tools.tavily_client = None

    result = await tools.search(["test topic"])

    assert result["tool"] == "search"
    assert result["results"] == []
    assert "not configured" in result["error"]


@pytest.mark.asyncio
async def test_search_captures_tavily_exceptions_per_query():
    """Verify search records per-query errors instead of failing the whole call."""
    from src.capabilities.deep_research_tools import DeepResearchTools

    tools = DeepResearchTools()
    tools.tavily_client = MagicMock()
    tools.tavily_client.search.side_effect = RuntimeError("search boom")

    result = await tools.search(["test topic"])

    assert result["results"][0]["query"] == "test topic"
    assert result["results"][0]["snippet"] == "Error: search boom"


@pytest.mark.asyncio
async def test_extract_info_rejects_non_dict_parser_output():
    """Verify extractor rejects non-dict parser outputs explicitly."""
    from src.capabilities.deep_research_tools import DeepResearchTools

    llm_manager = MagicMock()
    llm_manager.generate_response = AsyncMock(return_value=MagicMock(content="[]"))
    tools = DeepResearchTools(llm_manager=llm_manager)
    tools.extractor_parser.parse = MagicMock(return_value=["bad"])

    result = await tools._extract_info("webpage content", "find pricing")

    assert result["error"].startswith("Error extracting info:")
    assert "JSON object" in result["error"]


@pytest.mark.asyncio
async def test_extract_info_rejects_empty_fields():
    """Verify extractor rejects parsed payloads with empty required fields."""
    from src.capabilities.deep_research_tools import DeepResearchTools

    llm_manager = MagicMock()
    llm_manager.generate_response = AsyncMock(
        return_value=MagicMock(
            content='{"rational": "", "evidence": "Price is $10.", "summary": "Found price."}'
        )
    )
    tools = DeepResearchTools(llm_manager=llm_manager)

    result = await tools._extract_info("webpage content", "find pricing")

    assert result["error"].startswith("Error extracting info:")
    assert "non-empty rational" in result["error"]


@pytest.mark.asyncio
async def test_extract_info_rejects_malformed_output():
    """Verify malformed extractor output is surfaced as an extraction error."""
    from src.capabilities.deep_research_tools import DeepResearchTools

    llm_manager = MagicMock()
    llm_manager.generate_response = AsyncMock(
        return_value=MagicMock(content="not valid extractor json")
    )
    tools = DeepResearchTools(llm_manager=llm_manager)

    result = await tools._extract_info("webpage content", "find pricing")

    assert result["error"].startswith("Error extracting info:")


@pytest.mark.asyncio
async def test_visit_returns_structured_evidence_objects():
    """Verify visit returns structured evidence objects that preserve URL metadata."""
    from src.capabilities.deep_research_tools import DeepResearchTools

    tools = DeepResearchTools()
    tools._fetch_page_content = AsyncMock(return_value="page content")
    tools._extract_info = AsyncMock(
        return_value={
            "rational": "Matched pricing section",
            "evidence": "The page states that OpenAI o1 costs $20.",
            "summary": "Current price found.",
        }
    )

    result = await tools.visit(["https://example.com/pricing"], "find pricing")

    assert result["tool"] == "visit"
    assert result["results"][0]["url"] == "https://example.com/pricing"
    assert (
        result["results"][0]["evidence"] == "The page states that OpenAI o1 costs $20."
    )


@pytest.mark.asyncio
async def test_visit_returns_error_entry_when_fetch_fails():
    """Verify visit records a structured error result when page fetch fails."""
    from src.capabilities.deep_research_tools import DeepResearchTools

    tools = DeepResearchTools()
    tools._fetch_page_content = AsyncMock(
        return_value="Error: Could not fetch page content."
    )

    result = await tools.visit(["https://example.com/pricing"], "find pricing")

    assert result["results"][0]["error"] == "Error: Could not fetch page content."
    assert result["results"][0]["summary"] == "Error: Could not fetch page content."


@pytest.mark.asyncio
async def test_visit_returns_error_entry_when_fetch_raises():
    """Verify visit records a structured error result when fetching raises."""
    from src.capabilities.deep_research_tools import DeepResearchTools

    tools = DeepResearchTools()
    tools._fetch_page_content = AsyncMock(side_effect=RuntimeError("fetch boom"))

    result = await tools.visit(["https://example.com/pricing"], "find pricing")

    assert result["results"][0]["error"] == "Error: fetch boom"


@pytest.mark.asyncio
async def test_google_scholar_returns_configuration_error_without_tavily():
    """Verify google_scholar returns a structured configuration error without Tavily."""
    from src.capabilities.deep_research_tools import DeepResearchTools

    tools = DeepResearchTools()
    tools.tavily_client = None

    result = await tools.google_scholar(["paper topic"])

    assert result["tool"] == "google_scholar"
    assert result["results"] == []
    assert "not configured" in result["error"]


@pytest.mark.asyncio
async def test_google_scholar_captures_search_exceptions():
    """Verify google_scholar records per-query errors instead of failing the call."""
    from src.capabilities.deep_research_tools import DeepResearchTools

    tools = DeepResearchTools()
    tools.tavily_client = MagicMock()
    tools.tavily_client.search.side_effect = RuntimeError("scholar boom")

    result = await tools.google_scholar(["paper topic"])

    assert result["results"][0]["query"] == "paper topic"
    assert result["results"][0]["snippet"] == "Error: scholar boom"


@pytest.mark.asyncio
async def test_google_scholar_returns_structured_results():
    """Verify google_scholar returns structured query/title/url/snippet entries."""
    from src.capabilities.deep_research_tools import DeepResearchTools

    tools = DeepResearchTools()
    tools.tavily_client = MagicMock()
    tools.tavily_client.search.return_value = {
        "results": [
            {
                "title": "Scholar Result",
                "url": "https://example.com/paper",
                "content": "Paper summary.",
            }
        ]
    }

    result = await tools.google_scholar(["paper topic"])

    assert result["tool"] == "google_scholar"
    assert result["results"][0] == {
        "query": "paper topic",
        "title": "Scholar Result",
        "url": "https://example.com/paper",
        "snippet": "Paper summary.",
    }


@pytest.mark.asyncio
async def test_final_answer_rejects_untraceable_source_evidence(mock_llm_manager):
    """Verify grounded answers must trace source evidence back to collected observations."""
    agent = DeepResearchAgent({"llm": {"provider": "mock"}})
    agent.llm_manager = mock_llm_manager
    agent.state = AgentState.READY

    async def async_gen(content):
        yield content

    mock_llm_manager.stream_response.side_effect = [
        async_gen(
            '{"thought": "Need evidence", "type": "tool_call", "content": {"name": "search", "arguments": {"query": ["OpenAI o1 price"]}}}'
        ),
        async_gen(
            '{"thought": "Grounded answer", "type": "answer", "content": {"summary": "OpenAI o1 costs $20.", "claims": [{"statement": "OpenAI o1 costs $20.", "source_urls": ["https://example.com/pricing"]}], "sources": [{"url": "https://example.com/pricing", "title": "Pricing page", "evidence": "The pricing page states that OpenAI o1 costs $20."}], "insufficient_evidence": false}}'
        ),
        async_gen(
            '{"thought": "Insufficient traceable evidence", "type": "answer", "content": {"summary": "Insufficient evidence to answer reliably.", "claims": [], "sources": [], "insufficient_evidence": true}}'
        ),
    ]

    agent.tools = MagicMock()
    agent.tools.search = AsyncMock(
        return_value={
            "tool": "search",
            "results": [
                {
                    "query": "OpenAI o1 price",
                    "title": "Pricing",
                    "url": "https://example.com/pricing",
                    "snippet": "Price is $30.",
                }
            ],
        }
    )
    mock_llm_manager.generate_response.return_value = MagicMock(
        content=(
            '{"supported": false, "issues": '
            '["The cited evidence says $20, but the observed evidence for the same URL says $30."], '
            '"per_source": [{"url": "https://example.com/pricing", "supported": false, '
            '"reason": "Observed evidence contradicts the final evidence excerpt."}]}'
        )
    )

    response = await agent.process_input("price question")

    assert response == "Insufficient evidence to answer reliably."
    assert mock_llm_manager.stream_response.call_count == 3


@pytest.mark.asyncio
async def test_final_answer_rejects_uncollected_source_url(mock_llm_manager):
    """Verify grounded answers cannot cite source URLs never seen in tool observations."""
    agent = DeepResearchAgent({"llm": {"provider": "mock"}})
    agent.llm_manager = mock_llm_manager
    agent.state = AgentState.READY

    async def async_gen(content):
        yield content

    mock_llm_manager.stream_response.side_effect = [
        async_gen(
            '{"thought": "Need evidence", "type": "tool_call", "content": {"name": "search", "arguments": {"query": ["OpenAI o1 price"]}}}'
        ),
        async_gen(
            '{"thought": "Grounded answer", "type": "answer", "content": {"summary": "OpenAI o1 costs $20.", "claims": [{"statement": "OpenAI o1 costs $20.", "source_urls": ["https://other.example.com/pricing"]}], "sources": [{"url": "https://other.example.com/pricing", "title": "Other pricing page", "evidence": "OpenAI o1 costs $20."}], "insufficient_evidence": false}}'
        ),
        async_gen(
            '{"thought": "Insufficient traceable evidence", "type": "answer", "content": {"summary": "Insufficient evidence to answer reliably.", "claims": [], "sources": [], "insufficient_evidence": true}}'
        ),
    ]

    agent.tools = MagicMock()
    agent.tools.search = AsyncMock(
        return_value={
            "tool": "search",
            "results": [
                {
                    "query": "OpenAI o1 price",
                    "title": "Pricing",
                    "url": "https://example.com/pricing",
                    "snippet": "Price is $20.",
                }
            ],
        }
    )

    response = await agent.process_input("price question")

    assert response == "Insufficient evidence to answer reliably."
    assert mock_llm_manager.stream_response.call_count == 3


@pytest.mark.asyncio
async def test_final_answer_uses_shared_auditor_feedback(mock_llm_manager):
    """Verify final answer validation honors shared auditor review feedback."""
    agent = DeepResearchAgent({"llm": {"provider": "mock"}})
    agent.llm_manager = mock_llm_manager
    agent.state = AgentState.READY

    async def async_gen(content):
        yield content

    mock_llm_manager.stream_response.side_effect = [
        async_gen(
            '{"thought": "Need evidence", "type": "tool_call", "content": {"name": "search", "arguments": {"query": ["OpenAI o1 price"]}}}'
        ),
        async_gen(
            '{"thought": "Grounded answer", "type": "answer", "content": {"summary": "OpenAI o1 costs $20.", "claims": [{"statement": "OpenAI o1 costs $20.", "source_urls": ["https://example.com/pricing"]}], "sources": [{"url": "https://example.com/pricing", "title": "Pricing page", "evidence": "OpenAI o1 costs $20 on the pricing page."}], "insufficient_evidence": false}}'
        ),
        async_gen(
            '{"thought": "Fallback", "type": "answer", "content": {"summary": "Insufficient evidence to answer reliably.", "claims": [], "sources": [], "insufficient_evidence": true}}'
        ),
    ]

    agent.tools = MagicMock()
    agent.tools.search = AsyncMock(
        return_value={
            "tool": "search",
            "results": [
                {
                    "query": "OpenAI o1 price",
                    "title": "Pricing",
                    "url": "https://example.com/pricing",
                    "snippet": "OpenAI o1 costs $20 on the pricing page.",
                }
            ],
        }
    )
    agent.auditor.review_with_observations = AsyncMock(
        return_value=(
            "RETRY: Add a more direct supporting excerpt for claim "
            "'OpenAI o1 costs $20.'."
        )
    )

    response = await agent.process_input("price question")

    assert response == "Insufficient evidence to answer reliably."
    assert agent.auditor.review_with_observations.call_count == 1


@pytest.mark.asyncio
async def test_final_answer_requires_sources_section_text_when_steps_exist(
    mock_llm_manager,
):
    """Verify factual answers without a rendered Sources section are rejected early."""
    agent = DeepResearchAgent({"llm": {"provider": "mock"}})
    agent.llm_manager = mock_llm_manager

    result = await agent._validate_final_answer(
        content={"insufficient_evidence": False},
        rendered_answer="OpenAI o1 costs $20.",
        steps=[
            AgentStep(
                thought="search",
                action="search",
                action_input={"query": ["OpenAI o1 price"]},
                observation="obs",
                observation_data={},
            )
        ],
    )

    assert "Sources:" in result


def test_render_final_answer_returns_string_content():
    """Verify string answer content is returned unchanged."""
    agent = DeepResearchAgent({"llm": {"provider": "mock"}})

    assert agent._render_final_answer("plain answer") == "plain answer"


def test_render_final_answer_stringifies_non_dict_content():
    """Verify non-dict answer payloads are stringified."""
    agent = DeepResearchAgent({"llm": {"provider": "mock"}})

    assert agent._render_final_answer(["a", "b"]) == "['a', 'b']"


def test_render_final_answer_appends_insufficient_evidence_status_without_summary_phrase():
    """Verify insufficient-evidence answers append an explicit evidence-status line."""
    agent = DeepResearchAgent({"llm": {"provider": "mock"}})

    rendered = agent._render_final_answer(
        {
            "summary": "Need more evidence before concluding.",
            "claims": [],
            "sources": [],
            "insufficient_evidence": True,
        }
    )

    assert "Evidence status: insufficient evidence" in rendered


def test_render_tool_observation_stringifies_non_dict_result():
    """Verify non-dict tool results are stringified for observations."""
    agent = DeepResearchAgent({"llm": {"provider": "mock"}})

    assert agent._render_tool_observation("search", {}, "raw text") == "raw text"


def test_render_tool_observation_renders_search_entries():
    """Verify search-like tool observations include query, title, URL, and snippet."""
    agent = DeepResearchAgent({"llm": {"provider": "mock"}})

    rendered = agent._render_tool_observation(
        "search",
        {},
        {
            "tool": "search",
            "results": [
                {
                    "query": "OpenAI o1 price",
                    "title": "Pricing page",
                    "url": "https://example.com/pricing",
                    "snippet": "OpenAI o1 costs $20.",
                }
            ],
        },
    )

    assert "Query: OpenAI o1 price" in rendered
    assert "Title: Pricing page" in rendered
    assert "URL: https://example.com/pricing" in rendered
    assert "Snippet: OpenAI o1 costs $20." in rendered


def test_render_tool_observation_skips_non_dict_search_entries():
    """Verify search observation rendering ignores malformed result entries."""
    agent = DeepResearchAgent({"llm": {"provider": "mock"}})

    rendered = agent._render_tool_observation(
        "search",
        {},
        {
            "tool": "search",
            "results": [
                "bad-entry",
                {
                    "query": "OpenAI o1 price",
                    "title": "Pricing page",
                    "url": "https://example.com/pricing",
                    "snippet": "OpenAI o1 costs $20.",
                },
            ],
        },
    )

    assert "bad-entry" not in rendered
    assert "Pricing page" in rendered


def test_render_tool_observation_renders_visit_error_entry():
    """Verify visit observations render explicit error lines when extraction fails."""
    agent = DeepResearchAgent({"llm": {"provider": "mock"}})

    rendered = agent._render_tool_observation(
        "visit",
        {},
        {
            "tool": "visit",
            "results": [
                {
                    "url": "https://example.com/pricing",
                    "error": "Error extracting info: boom",
                }
            ],
        },
    )

    assert "URL: https://example.com/pricing" in rendered
    assert "Error: Error extracting info: boom" in rendered


def test_render_tool_observation_renders_visit_evidence_fields():
    """Verify visit observations render rationale, evidence, and summary fields."""
    agent = DeepResearchAgent({"llm": {"provider": "mock"}})

    rendered = agent._render_tool_observation(
        "visit",
        {},
        {
            "tool": "visit",
            "results": [
                {
                    "url": "https://example.com/pricing",
                    "rational": "Matched pricing section",
                    "evidence": "The page states that OpenAI o1 costs $20.",
                    "summary": "Current price found.",
                }
            ],
        },
    )

    assert "Rationale: Matched pricing section" in rendered
    assert "Evidence: The page states that OpenAI o1 costs $20." in rendered
    assert "Summary: Current price found." in rendered


def test_render_tool_observation_skips_non_dict_visit_entries_and_unknown_tool_falls_back():
    """Verify visit skips malformed entries and unknown structured tools fall back to str()."""
    agent = DeepResearchAgent({"llm": {"provider": "mock"}})

    rendered_visit = agent._render_tool_observation(
        "visit",
        {},
        {
            "tool": "visit",
            "results": [
                "bad-entry",
                {
                    "url": "https://example.com/pricing",
                    "summary": "Current price found.",
                },
            ],
        },
    )
    rendered_other = agent._render_tool_observation(
        "other",
        {},
        {"tool": "other", "results": ["raw"]},
    )

    assert "bad-entry" not in rendered_visit
    assert "Summary: Current price found." in rendered_visit
    assert rendered_other == "{'tool': 'other', 'results': ['raw']}"


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
