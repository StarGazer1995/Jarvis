"""Tests for the structured Deep Research reviewer auditor."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.capabilities.refinement.auditor import DeepResearchAuditor
from src.core.agent.types import AgentStep


def test_deep_research_auditor_passes_grounded_report():
    """Verify grounded reports with valid claim/source linkage pass audit."""
    auditor = DeepResearchAuditor()

    verdict = auditor.review(
        "OpenAI o1 costs $20.\n\n"
        "Claims:\n"
        "1. OpenAI o1 costs $20. [Sources: https://example.com/pricing]\n\n"
        "Sources:\n"
        "- Pricing page: https://example.com/pricing\n"
        "  Evidence: The pricing page states that OpenAI o1 costs $20."
    )

    assert verdict == "PASS"


def test_deep_research_auditor_rejects_missing_claims():
    """Verify grounded-looking reports without claims are rejected."""
    auditor = DeepResearchAuditor()

    verdict = auditor.review(
        "OpenAI o1 costs $20.\n\nSources:\n- Pricing page: https://example.com/pricing"
    )

    assert verdict.startswith("RETRY:")
    assert "Claims:" in verdict


def test_deep_research_auditor_rejects_missing_sources():
    """Verify reports with claims but no source section are rejected."""
    auditor = DeepResearchAuditor()

    verdict = auditor.review(
        "OpenAI o1 costs $20.\n\n"
        "Claims:\n"
        "1. OpenAI o1 costs $20. [Sources: https://example.com/pricing]"
    )

    assert verdict.startswith("RETRY:")
    assert "Sources:" in verdict


def test_deep_research_auditor_rejects_undeclared_claim_source():
    """Verify claim source URLs must appear in the top-level source list."""
    auditor = DeepResearchAuditor()

    verdict = auditor.review(
        "OpenAI o1 costs $20.\n\n"
        "Claims:\n"
        "1. OpenAI o1 costs $20. [Sources: https://missing.example.com]\n\n"
        "Sources:\n"
        "- Pricing page: https://example.com/pricing\n"
        "  Evidence: The pricing page states that OpenAI o1 costs $20."
    )

    assert verdict.startswith("RETRY:")
    assert "top-level" in verdict


def test_deep_research_auditor_passes_insufficient_evidence_report():
    """Verify explicit insufficient-evidence reports pass without claims."""
    auditor = DeepResearchAuditor()

    verdict = auditor.review("Insufficient evidence to answer reliably.")

    assert verdict == "PASS"


def test_deep_research_auditor_rejects_missing_evidence_excerpt():
    """Verify claims fail audit when mapped sources omit evidence excerpts."""
    auditor = DeepResearchAuditor()

    verdict = auditor.review(
        "OpenAI o1 costs $20.\n\n"
        "Claims:\n"
        "1. OpenAI o1 costs $20. [Sources: https://example.com/pricing]\n\n"
        "Sources:\n"
        "- Pricing page: https://example.com/pricing\n"
        "  Evidence: "
    )

    assert verdict.startswith("RETRY:")
    assert "Evidence:" in verdict


def test_deep_research_auditor_rejects_unsupported_numeric_claim():
    """Verify claim numbers must appear in the evidence excerpts."""
    auditor = DeepResearchAuditor()

    verdict = auditor.review(
        "OpenAI o1 costs $20.\n\n"
        "Claims:\n"
        "1. OpenAI o1 costs $20. [Sources: https://example.com/pricing]\n\n"
        "Sources:\n"
        "- Pricing page: https://example.com/pricing\n"
        "  Evidence: The pricing page states that OpenAI o1 costs $30."
    )

    assert verdict.startswith("RETRY:")
    assert "Missing numbers" in verdict


def test_deep_research_auditor_rejects_unsupported_claim_terms():
    """Verify evidence excerpts must overlap with the substantive claim terms."""
    auditor = DeepResearchAuditor()

    verdict = auditor.review(
        "Claude Sonnet supports multimodal document analysis.\n\n"
        "Claims:\n"
        "1. Claude Sonnet supports multimodal document analysis. [Sources: https://example.com/model]\n\n"
        "Sources:\n"
        "- Model page: https://example.com/model\n"
        "  Evidence: The model page describes a text-only coding assistant."
    )

    assert verdict.startswith("RETRY:")
    assert "support the claim text" in verdict


@pytest.mark.asyncio
async def test_deep_research_auditor_passes_traceable_raw_observations():
    """Verify LLM-backed traceability passes when raw observations support the report."""
    llm_manager = MagicMock()
    llm_manager.generate_response = AsyncMock(
        return_value=MagicMock(
            content=(
                '{"supported": true, "issues": [], "per_source": '
                '[{"url": "https://example.com/pricing", "supported": true, '
                '"reason": "The observed evidence directly supports the cited excerpt."}]}'
            )
        )
    )
    auditor = DeepResearchAuditor(llm_manager=llm_manager)
    steps = [
        AgentStep(
            thought="search",
            action="search",
            action_input={"query": ["OpenAI o1 price"]},
            observation="search result",
            observation_data={
                "tool": "search",
                "results": [
                    {
                        "query": "OpenAI o1 price",
                        "title": "Pricing page",
                        "url": "https://example.com/pricing",
                        "snippet": "OpenAI o1 costs $20 on the pricing page.",
                    }
                ],
            },
        )
    ]

    verdict = await auditor.review_with_observations(
        "OpenAI o1 costs $20.\n\n"
        "Claims:\n"
        "1. OpenAI o1 costs $20. [Sources: https://example.com/pricing]\n\n"
        "Sources:\n"
        "- Pricing page: https://example.com/pricing\n"
        "  Evidence: OpenAI o1 costs $20 on the pricing page.",
        steps=steps,
    )

    assert verdict == "PASS"


@pytest.mark.asyncio
async def test_deep_research_auditor_rejects_unobserved_declared_source():
    """Verify declared sources must map back to collected raw observations."""
    auditor = DeepResearchAuditor()
    steps = [
        AgentStep(
            thought="search",
            action="search",
            action_input={"query": ["OpenAI o1 price"]},
            observation="search result",
            observation_data={
                "tool": "search",
                "results": [
                    {
                        "query": "OpenAI o1 price",
                        "title": "Pricing page",
                        "url": "https://example.com/pricing",
                        "snippet": "OpenAI o1 costs $20 on the pricing page.",
                    }
                ],
            },
        )
    ]

    verdict = await auditor.review_with_observations(
        "OpenAI o1 costs $20.\n\n"
        "Claims:\n"
        "1. OpenAI o1 costs $20. [Sources: https://other.example.com/pricing]\n\n"
        "Sources:\n"
        "- Other pricing page: https://other.example.com/pricing\n"
        "  Evidence: OpenAI o1 costs $20 on the pricing page.",
        steps=steps,
    )

    assert verdict.startswith("RETRY:")
    assert "real collected tool observation" in verdict


@pytest.mark.asyncio
async def test_deep_research_auditor_rejects_untraceable_observation_evidence():
    """Verify LLM-backed traceability can reject unsupported evidence excerpts."""
    llm_manager = MagicMock()
    llm_manager.generate_response = AsyncMock(
        return_value=MagicMock(
            content=(
                '{"supported": false, "issues": '
                '["The cited evidence says $30, but the observed evidence for the same URL says $20."], '
                '"per_source": [{"url": "https://example.com/pricing", "supported": false, '
                '"reason": "Observed evidence contradicts the final evidence excerpt."}]}'
            )
        )
    )
    auditor = DeepResearchAuditor(llm_manager=llm_manager)
    steps = [
        AgentStep(
            thought="visit",
            action="visit",
            action_input={
                "url": ["https://example.com/pricing"],
                "goal": "find current price",
            },
            observation="visit result",
            observation_data={
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
    ]

    verdict = await auditor.review_with_observations(
        "OpenAI o1 costs $30.\n\n"
        "Claims:\n"
        "1. OpenAI o1 costs $30. [Sources: https://example.com/pricing]\n\n"
        "Sources:\n"
        "- Pricing page: https://example.com/pricing\n"
        "  Evidence: The page states that OpenAI o1 costs $30.",
        steps=steps,
    )

    assert verdict.startswith("RETRY:")
    assert "observed evidence" in verdict


def test_deep_research_auditor_rejects_claim_without_source_urls():
    """Verify a rendered claim must include at least one source URL."""
    auditor = DeepResearchAuditor()

    verdict = auditor.review(
        "OpenAI o1 costs $20.\n\n"
        "Claims:\n"
        "1. OpenAI o1 costs $20.\n\n"
        "Sources:\n"
        "- Pricing page: https://example.com/pricing\n"
        "  Evidence: The pricing page states that OpenAI o1 costs $20."
    )

    assert verdict.startswith("RETRY:")
    assert "at least one source URL" in verdict


@pytest.mark.asyncio
async def test_deep_research_auditor_returns_deterministic_failure_before_llm():
    """Verify deterministic failures short-circuit the observation-backed path."""
    llm_manager = MagicMock()
    llm_manager.generate_response = AsyncMock()
    auditor = DeepResearchAuditor(llm_manager=llm_manager)

    verdict = await auditor.review_with_observations("Ungrounded summary only.")

    assert verdict.startswith("RETRY:")
    llm_manager.generate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_deep_research_auditor_passes_without_steps_after_structure_checks():
    """Verify observation-backed review passes when no raw observations are provided."""
    auditor = DeepResearchAuditor()

    verdict = await auditor.review_with_observations(
        "OpenAI o1 costs $20.\n\n"
        "Claims:\n"
        "1. OpenAI o1 costs $20. [Sources: https://example.com/pricing]\n\n"
        "Sources:\n"
        "- Pricing page: https://example.com/pricing\n"
        "  Evidence: The pricing page states that OpenAI o1 costs $20."
    )

    assert verdict == "PASS"


@pytest.mark.asyncio
async def test_deep_research_auditor_requires_llm_manager_for_observation_audit():
    """Verify model-backed traceability requires an LLM manager once steps exist."""
    auditor = DeepResearchAuditor()
    steps = [
        AgentStep(
            thought="search",
            action="search",
            action_input={"query": ["OpenAI o1 price"]},
            observation="search result",
            observation_data={
                "tool": "search",
                "results": [
                    {
                        "query": "OpenAI o1 price",
                        "title": "Pricing page",
                        "url": "https://example.com/pricing",
                        "snippet": "OpenAI o1 costs $20 on the pricing page.",
                    }
                ],
            },
        )
    ]

    verdict = await auditor.review_with_observations(
        "OpenAI o1 costs $20.\n\n"
        "Claims:\n"
        "1. OpenAI o1 costs $20. [Sources: https://example.com/pricing]\n\n"
        "Sources:\n"
        "- Pricing page: https://example.com/pricing\n"
        "  Evidence: OpenAI o1 costs $20 on the pricing page.",
        steps=steps,
    )

    assert verdict.startswith("RETRY:")
    assert "no LLM manager is configured" in verdict


def test_deep_research_auditor_parse_claims_skips_blank_lines():
    """Verify blank claim lines are ignored during parsing."""
    auditor = DeepResearchAuditor()

    claims = auditor._parse_claims(
        "Claims:\n\n1. OpenAI o1 costs $20. [Sources: https://example.com/pricing]\n\n"
        "Sources:\n"
        "- Pricing page: https://example.com/pricing\n"
        "  Evidence: The pricing page states that OpenAI o1 costs $20."
    )

    assert len(claims) == 1


def test_deep_research_auditor_parse_claims_skips_internal_blank_lines():
    """Verify internal blank claim lines are ignored safely."""
    auditor = DeepResearchAuditor()

    claims = auditor._parse_claims(
        "Claims:\n"
        "1. OpenAI o1 costs $20. [Sources: https://example.com/pricing]\n"
        "\n"
        "2. Claude Sonnet costs $15. [Sources: https://example.com/sonnet]\n\n"
        "Sources:\n"
        "- Pricing page: https://example.com/pricing\n"
        "  Evidence: OpenAI o1 costs $20.\n"
        "- Sonnet page: https://example.com/sonnet\n"
        "  Evidence: Claude Sonnet costs $15."
    )

    assert len(claims) == 2


def test_deep_research_auditor_parse_sources_skips_non_matching_lines():
    """Verify the source parser skips blank and malformed lines safely."""
    auditor = DeepResearchAuditor()

    sources = auditor._parse_declared_sources(
        "Sources:\n\n"
        "malformed source header\n"
        "- Pricing page: https://example.com/pricing\n"
        "  Evidence: The pricing page states that OpenAI o1 costs $20."
    )

    assert list(sources) == ["https://example.com/pricing"]


def test_deep_research_auditor_parse_sources_skips_internal_blank_lines():
    """Verify internal blank source lines are ignored safely."""
    auditor = DeepResearchAuditor()

    sources = auditor._parse_declared_sources(
        "Sources:\n"
        "- Pricing page: https://example.com/pricing\n"
        "  Evidence: The pricing page states that OpenAI o1 costs $20.\n"
        "\n"
        "- Sonnet page: https://example.com/sonnet\n"
        "  Evidence: Claude Sonnet costs $15."
    )

    assert set(sources) == {
        "https://example.com/pricing",
        "https://example.com/sonnet",
    }


def test_deep_research_auditor_validate_observed_source_urls_handles_empty_sources():
    """Verify the observed-source URL gate passes when no declared sources exist."""
    auditor = DeepResearchAuditor()

    verdict = auditor._validate_observed_source_urls({}, {})

    assert verdict is None


@pytest.mark.asyncio
async def test_deep_research_auditor_handles_non_dict_audit_parser_result():
    """Verify a non-dict parser result raises the dedicated audit error."""
    llm_manager = MagicMock()
    llm_manager.generate_response = AsyncMock(return_value=MagicMock(content="{}"))
    auditor = DeepResearchAuditor(llm_manager=llm_manager)
    auditor.audit_parser.parse = MagicMock(return_value=["bad"])

    with pytest.raises(ValueError, match="audit verdict must be a JSON object"):
        await auditor._review_traceability_with_llm(
            content="report",
            declared_sources={
                "https://example.com/pricing": auditor._parse_declared_sources(
                    "Sources:\n- Pricing page: https://example.com/pricing\n  Evidence: Price is $20."
                )["https://example.com/pricing"]
            },
            observed_evidence={"https://example.com/pricing": ["Price is $20."]},
        )


@pytest.mark.asyncio
async def test_deep_research_auditor_returns_generic_retry_when_llm_issues_missing():
    """Verify unsupported verdicts without issues still return a usable retry message."""
    llm_manager = MagicMock()
    llm_manager.generate_response = AsyncMock(
        return_value=MagicMock(
            content=(
                '{"supported": false, "issues": [], "per_source": '
                '[{"url": "https://example.com/pricing", "supported": false, '
                '"reason": "Unsupported."}]}'
            )
        )
    )
    auditor = DeepResearchAuditor(llm_manager=llm_manager)

    verdict = await auditor._review_traceability_with_llm(
        content="report",
        declared_sources={
            "https://example.com/pricing": auditor._parse_declared_sources(
                "Sources:\n- Pricing page: https://example.com/pricing\n  Evidence: Price is $20."
            )["https://example.com/pricing"]
        },
        observed_evidence={"https://example.com/pricing": ["Price is $20."]},
    )

    assert verdict.startswith("RETRY:")
    assert "did not provide a usable issue list" in verdict


def test_deep_research_auditor_collect_observed_evidence_skips_invalid_entries():
    """Verify raw observation collection ignores malformed observation payloads."""
    auditor = DeepResearchAuditor()
    steps = [
        AgentStep(thought="bad1", observation_data="not-a-dict"),
        AgentStep(
            thought="bad2", observation_data={"tool": "search", "results": "bad"}
        ),
        AgentStep(
            thought="bad3",
            observation_data={
                "tool": "search",
                "results": [
                    "bad-entry",
                    {
                        "url": "https://example.com",
                        "query": "",
                        "title": "",
                        "snippet": "",
                    },
                ],
            },
        ),
        AgentStep(
            thought="bad4",
            observation_data={
                "tool": "visit",
                "results": ["bad-entry", {"url": "", "evidence": "x", "summary": "y"}],
            },
        ),
    ]

    evidence_map = auditor._collect_observed_evidence(steps)

    assert evidence_map == {}
