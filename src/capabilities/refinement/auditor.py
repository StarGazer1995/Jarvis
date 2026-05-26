"""Structured reviewer utilities for Deep Research outputs."""

import json
import re
from dataclasses import dataclass
from typing import Any

from src.core.agent.types import AgentStep
from src.core.llm.client import LLMManager
from src.core.llm.converters import convert_langchain_to_llm_messages
from src.core.llm.parsers import JSONOutputParser
from src.core.prompt.manager import PromptManager


@dataclass
class ParsedClaim:
    """Represent a rendered claim and its referenced source URLs."""

    statement: str
    source_urls: list[str]


@dataclass
class ParsedSource:
    """Represent a rendered source and its supporting evidence excerpt."""

    url: str
    title: str
    evidence: str


class DeepResearchAuditor:
    """
    Audit rendered Deep Research reports using layered structural checks.
    """

    _CLAIMS_HEADER = "Claims:"
    _SOURCES_HEADER = "Sources:"
    _INSUFFICIENT_EVIDENCE_PATTERNS = (
        "insufficient evidence",
        "evidence is insufficient",
        "insufficient information",
    )
    _STOPWORDS = {
        "about",
        "after",
        "again",
        "also",
        "among",
        "around",
        "because",
        "before",
        "being",
        "between",
        "costs",
        "could",
        "every",
        "from",
        "have",
        "latest",
        "model",
        "models",
        "pricing",
        "research",
        "report",
        "should",
        "their",
        "there",
        "these",
        "this",
        "those",
        "through",
        "under",
        "using",
        "with",
    }

    def __init__(
        self,
        llm_manager: LLMManager | None = None,
        prompt_manager: PromptManager | None = None,
    ):
        """
        Initialize the auditor.

        Args:
            llm_manager: Optional LLM manager for model-backed traceability review.
            prompt_manager: Optional prompt manager override.
        """
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager or PromptManager()
        self.audit_parser = JSONOutputParser(protocol="deep_research_audit")

    def review(self, content: str, steps: list[AgentStep] | None = None) -> str:
        """
        Review a rendered Deep Research report and return a refinement verdict.

        Args:
            content: Rendered report content.
            steps: Unused in the synchronous path. Reserved for API compatibility.

        Returns:
            `PASS` when the report passes deterministic structural checks,
            otherwise `RETRY: <instructions>`.
        """
        _ = steps
        normalized_content = content.strip()
        lowered_content = normalized_content.lower()
        claims = self._parse_claims(normalized_content)
        declared_sources = self._parse_declared_sources(normalized_content)

        if not claims:
            if any(
                pattern in lowered_content
                for pattern in self._INSUFFICIENT_EVIDENCE_PATTERNS
            ):
                return "PASS"
            return (
                "RETRY: Add a 'Claims:' section for grounded findings, or explicitly "
                "state that the evidence is insufficient."
            )

        if not declared_sources:
            return "RETRY: Add a 'Sources:' section listing every source URL used by the claims."

        declared_source_urls = set(declared_sources)
        for claim in claims:
            if not claim.source_urls:
                return (
                    "RETRY: Every claim must include at least one source URL in the format "
                    "[Sources: https://...]."
                )

            for source_url in claim.source_urls:
                if source_url not in declared_source_urls:
                    return (
                        "RETRY: Ensure every claim source URL is also listed in the top-level "
                        "'Sources:' section."
                    )

            semantic_verdict = self._validate_claim_support(
                claim=claim, declared_sources=declared_sources
            )
            if semantic_verdict:
                return semantic_verdict

        return "PASS"

    async def review_with_observations(
        self, content: str, steps: list[AgentStep] | None = None
    ) -> str:
        """
        Review a rendered report with raw observation evidence available.

        Args:
            content: Rendered report content.
            steps: Optional Deep Research execution steps containing raw
                `observation_data` for traceability audit.

        Returns:
            `PASS` when both deterministic and model-backed checks pass,
            otherwise `RETRY: <instructions>`.
        """
        deterministic_verdict = self.review(content)
        if deterministic_verdict != "PASS":
            return deterministic_verdict

        if not steps:
            return "PASS"

        declared_sources = self._parse_declared_sources(content.strip())
        observed_evidence = self._collect_observed_evidence(steps)
        source_url_verdict = self._validate_observed_source_urls(
            declared_sources=declared_sources, observed_evidence=observed_evidence
        )
        if source_url_verdict:
            return source_url_verdict

        if not self.llm_manager:
            return (
                "RETRY: LLM-backed source traceability review is unavailable because "
                "no LLM manager is configured."
            )

        return await self._review_traceability_with_llm(
            content=content,
            declared_sources=declared_sources,
            observed_evidence=observed_evidence,
        )

    def _parse_claims(self, content: str) -> list[ParsedClaim]:
        """
        Parse rendered claim lines from a Deep Research report.

        Args:
            content: Rendered report content.

        Returns:
            Parsed claim objects extracted from the `Claims:` section.
        """
        claims_section = self._extract_section(
            content=content,
            header=self._CLAIMS_HEADER,
            next_header=self._SOURCES_HEADER,
        )
        if not claims_section:
            return []

        parsed_claims: list[ParsedClaim] = []
        for raw_line in claims_section.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            statement = re.sub(r"^\d+\.\s*", "", line).strip()
            source_urls = re.findall(r"https?://[^\s,\]]+", line)
            statement = re.sub(r"\[Sources:\s*[^\]]+\]\s*$", "", statement).strip()
            parsed_claims.append(
                ParsedClaim(statement=statement, source_urls=source_urls)
            )

        return parsed_claims

    def _parse_declared_sources(self, content: str) -> dict[str, ParsedSource]:
        """
        Parse the declared top-level sources from a rendered report.

        Args:
            content: Rendered report content.

        Returns:
            Mapping of source URL to parsed source metadata.
        """
        sources_section = self._extract_section(
            content=content, header=self._SOURCES_HEADER, next_header=None
        )
        if not sources_section:
            return {}

        parsed_sources: dict[str, ParsedSource] = {}
        lines = sources_section.splitlines()
        index = 0
        while index < len(lines):
            raw_line = lines[index]
            line = raw_line.strip()
            if not line:
                index += 1
                continue

            header_match = re.match(
                r"^-\s*(?P<title>.*?):\s*(?P<url>https?://\S+)$", line
            )
            if not header_match:
                index += 1
                continue

            title = header_match.group("title").strip()
            url = header_match.group("url").strip()
            evidence = ""

            if index + 1 < len(lines):
                next_line = lines[index + 1].strip()
                if next_line.startswith("Evidence:"):
                    evidence = next_line.split("Evidence:", 1)[1].strip()
                    index += 1

            parsed_sources[url] = ParsedSource(url=url, title=title, evidence=evidence)
            index += 1

        return parsed_sources

    def _validate_claim_support(
        self, claim: ParsedClaim, declared_sources: dict[str, ParsedSource]
    ) -> str | None:
        """
        Validate that a claim is minimally supported by the mapped source evidence.

        Args:
            claim: Parsed claim to validate.
            declared_sources: Parsed source mapping for the report.

        Returns:
            Optional retry verdict when support is insufficient.
        """
        mapped_sources = [
            declared_sources[source_url]
            for source_url in claim.source_urls
            if source_url in declared_sources
        ]
        combined_evidence = " ".join(
            source.evidence for source in mapped_sources
        ).strip()
        if not combined_evidence:
            return (
                "RETRY: Every claim source must include a non-empty 'Evidence:' excerpt in "
                "the top-level 'Sources:' section."
            )

        claim_numbers = re.findall(r"\d+(?:\.\d+)?", claim.statement)
        missing_numbers = [
            number for number in claim_numbers if number not in combined_evidence
        ]
        if missing_numbers:
            return (
                "RETRY: The evidence excerpts must explicitly contain the numeric facts used "
                f"in each claim. Missing numbers for claim '{claim.statement}': {', '.join(missing_numbers)}."
            )

        claim_terms = self._extract_salient_terms(claim.statement)
        evidence_terms = self._extract_salient_terms(combined_evidence)
        overlapping_terms = claim_terms & evidence_terms
        required_overlap = 1 if len(claim_terms) <= 2 else 2
        if claim_terms and len(overlapping_terms) < required_overlap:
            return (
                "RETRY: The evidence excerpts do not appear to support the claim text closely "
                f"enough. Add a more direct supporting excerpt for claim '{claim.statement}'."
            )

        return None

    def _validate_observed_source_urls(
        self,
        declared_sources: dict[str, ParsedSource],
        observed_evidence: dict[str, list[str]],
    ) -> str | None:
        """
        Validate that declared source URLs exist in collected observations.

        Args:
            declared_sources: Parsed source mapping for the report.
            observed_evidence: URL-indexed observed evidence fragments.

        Returns:
            Optional retry verdict when a declared source URL was never observed.
        """
        if not declared_sources:
            return None

        for source in declared_sources.values():
            if source.url not in observed_evidence:
                return (
                    "RETRY: Every declared source URL must come from a real collected "
                    "tool observation."
                )

        return None

    async def _review_traceability_with_llm(
        self,
        content: str,
        declared_sources: dict[str, ParsedSource],
        observed_evidence: dict[str, list[str]],
    ) -> str:
        """
        Use an LLM to audit source traceability against raw observations.

        Args:
            content: Rendered final report content.
            declared_sources: Parsed source mapping for the report.
            observed_evidence: URL-indexed observed evidence fragments.

        Returns:
            `PASS` when the model judges the report supported, otherwise
            `RETRY: <instructions>`.
        """
        observed_payload = self._build_observed_evidence_payload(
            declared_sources=declared_sources,
            observed_evidence=observed_evidence,
        )
        prompt_messages = self.prompt_manager.render_template(
            "deep_research_auditor",
            report_content=json.dumps(content, ensure_ascii=False),
            observed_evidence_json=json.dumps(observed_payload, ensure_ascii=False),
        )
        messages = convert_langchain_to_llm_messages(prompt_messages)
        response = await self.llm_manager.generate_response(messages)
        parsed_response = self.audit_parser.parse(str(response.content))
        if not isinstance(parsed_response, dict):
            raise ValueError("Deep Research audit verdict must be a JSON object.")

        if bool(parsed_response.get("supported")):
            return "PASS"

        issues = [
            issue.strip()
            for issue in parsed_response.get("issues", [])
            if isinstance(issue, str) and issue.strip()
        ]
        if issues:
            return "RETRY: " + " ".join(issues)

        return (
            "RETRY: The model-backed source traceability audit rejected the report, "
            "but did not provide a usable issue list."
        )

    def _build_observed_evidence_payload(
        self,
        declared_sources: dict[str, ParsedSource],
        observed_evidence: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """
        Build the URL-scoped evidence payload sent to the LLM auditor.

        Args:
            declared_sources: Parsed source mapping for the report.
            observed_evidence: URL-indexed observed evidence fragments.

        Returns:
            A filtered mapping containing only evidence for declared source URLs.
        """
        return {
            source.url: observed_evidence.get(source.url, [])
            for source in declared_sources.values()
        }

    def _collect_observed_evidence(
        self, steps: list[AgentStep]
    ) -> dict[str, list[str]]:
        """
        Collect URL-indexed evidence text from raw tool observation data.

        Args:
            steps: Deep Research execution steps containing raw observations.

        Returns:
            Mapping of URL to collected evidence text fragments.
        """
        evidence_map: dict[str, list[str]] = {}
        for step in steps:
            raw_observation = step.observation_data
            if not isinstance(raw_observation, dict):
                continue

            tool = raw_observation.get("tool")
            results = raw_observation.get("results", [])
            if not isinstance(results, list):
                continue

            if tool in {"search", "google_scholar"}:
                self._collect_search_like_evidence(
                    evidence_map=evidence_map, results=results
                )
                continue

            if tool == "visit":
                self._collect_visit_evidence(evidence_map=evidence_map, results=results)

        return evidence_map

    def _collect_search_like_evidence(
        self, evidence_map: dict[str, list[str]], results: list[dict[str, Any]]
    ) -> None:
        """
        Collect evidence from search-like tool results.

        Args:
            evidence_map: Mutable URL-to-evidence mapping.
            results: Structured tool result entries from search-like tools.
        """
        for entry in results:
            if not isinstance(entry, dict):
                continue

            url = str(entry.get("url", "")).strip()
            query = str(entry.get("query", "")).strip()
            title = str(entry.get("title", "")).strip()
            snippet = str(entry.get("snippet", "")).strip()
            observed_parts = [
                part
                for part in (query, title, snippet)
                if part and not part.startswith("Error:")
            ]
            if url and observed_parts:
                evidence_map.setdefault(url, []).append(" ".join(observed_parts))

    def _collect_visit_evidence(
        self, evidence_map: dict[str, list[str]], results: list[dict[str, Any]]
    ) -> None:
        """
        Collect evidence from visit tool results.

        Args:
            evidence_map: Mutable URL-to-evidence mapping.
            results: Structured tool result entries from the visit tool.
        """
        for entry in results:
            if not isinstance(entry, dict):
                continue

            url = str(entry.get("url", "")).strip()
            evidence = str(entry.get("evidence", "")).strip()
            summary = str(entry.get("summary", "")).strip()
            if not url:
                continue
            if evidence:
                evidence_map.setdefault(url, []).append(evidence)
            if summary and not summary.startswith("Error:"):
                evidence_map.setdefault(url, []).append(summary)

    def _extract_salient_terms(self, text: str) -> set[str]:
        """
        Extract salient lowercase terms from text for deterministic support checks.

        Args:
            text: Input text.

        Returns:
            Set of salient terms.
        """
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]{2,}", text.lower())
        return {
            token
            for token in tokens
            if token not in self._STOPWORDS and len(token) >= 4
        }

    def _extract_section(
        self, content: str, header: str, next_header: str | None
    ) -> str:
        """
        Extract a named section from a rendered report.

        Args:
            content: Rendered report content.
            header: Header to extract.
            next_header: Optional next header that terminates the section.

        Returns:
            Section body text without the header line.
        """
        pattern = rf"{re.escape(header)}\n(?P<body>[\s\S]*)"
        match = re.search(pattern, content)
        if not match:
            return ""

        body = match.group("body")
        if next_header:
            next_header_pattern = rf"\n\n{re.escape(next_header)}\n"
            next_match = re.search(next_header_pattern, body)
            if next_match:
                return body[: next_match.start()].strip()

        return body.strip()
