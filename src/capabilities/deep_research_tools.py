"""
Deep Research Tools
Implementation of tools required for the Deep Research agent.
"""

import logging
import os

import requests

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

from src.capabilities.interpreter import PythonInterpreter
from src.core.llm.client import LLMManager
from src.core.llm.config import load_llm_config
from src.core.llm.converters import convert_langchain_to_llm_messages
from src.core.llm.parsers import JSONOutputParser
from src.core.prompt.manager import PromptManager

logger = logging.getLogger(__name__)


class DeepResearchTools:
    """
    Tools for Deep Research Agent.
    """

    def __init__(self, llm_manager: LLMManager | None = None):
        """
        Initialize the tools.
        """
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.tavily_client = (
            TavilyClient(api_key=self.tavily_api_key)
            if (self.tavily_api_key and TavilyClient)
            else None
        )

        self.prompt_manager = PromptManager()
        self.llm_manager = llm_manager
        if not self.llm_manager:
            self.llm_manager = LLMManager(load_llm_config())
            # Note: We assume LLM manager is initialized by the agent or we initialize it lazily

        self.interpreter = PythonInterpreter()
        self.extractor_parser = JSONOutputParser()

    async def search(self, query: list[str]) -> dict[str, object]:
        """
        Perform web searches for multiple queries.
        """
        if not self.tavily_client:
            return {
                "tool": "search",
                "error": "Tavily API key not configured.",
                "results": [],
            }

        results: list[dict[str, str]] = []
        for q in query:
            try:
                response = self.tavily_client.search(
                    query=q,
                    search_depth="advanced",
                    max_results=5,
                    include_answer=False,
                )

                for res in response.get("results", []):
                    results.append(
                        {
                            "query": q,
                            "title": res["title"],
                            "url": res["url"],
                            "snippet": res["content"],
                        }
                    )
            except Exception as e:
                results.append(
                    {"query": q, "title": "", "url": "", "snippet": f"Error: {str(e)}"}
                )

        return {"tool": "search", "results": results}

    async def visit(self, url: list[str], goal: str) -> dict[str, object]:
        """
        Visit webpages and extract information based on the goal.
        """
        results: list[dict[str, str]] = []
        for u in url:
            try:
                content = await self._fetch_page_content(u)
                if content.startswith("Error"):
                    results.append(
                        {
                            "url": u,
                            "rational": "",
                            "evidence": "",
                            "summary": content,
                            "error": content,
                        }
                    )
                    continue

                # Summarize/Extract using LLM
                extracted = await self._extract_info(content, goal)
                extracted["url"] = u
                results.append(extracted)
            except Exception as e:
                results.append(
                    {
                        "url": u,
                        "rational": "",
                        "evidence": "",
                        "summary": f"Error: {str(e)}",
                        "error": f"Error: {str(e)}",
                    }
                )

        return {"tool": "visit", "goal": goal, "results": results}

    async def _fetch_page_content(self, url: str) -> str:
        """
        Fetch page content using Jina or Tavily.
        """
        # Try Jina first (as per DeepResearch paper preference)
        try:
            # Jina Reader API: https://r.jina.ai/
            response = requests.get(f"https://r.jina.ai/{url}", timeout=30)
            if response.status_code == 200:
                return response.text
        except Exception as e:
            logger.warning(f"Jina fetch failed for {url}: {e}")

        # Fallback to Tavily
        if self.tavily_client:
            try:
                response = self.tavily_client.extract(urls=[url])
                if response.get("results"):
                    return response["results"][0].get("raw_content") or response[
                        "results"
                    ][0].get("content")
            except Exception as e:
                logger.warning(f"Tavily extract failed for {url}: {e}")

        return "Error: Could not fetch page content."

    async def _extract_info(self, content: str, goal: str) -> dict[str, str]:
        """
        Extract relevant info using LLM.
        """
        # Truncate content to avoid token limits (rough estimate)
        max_chars = 50000
        if len(content) > max_chars:
            content = content[:max_chars] + "...(truncated)"

        prompt_messages = self.prompt_manager.render_template(
            "deep_research_extractor", webpage_content=content, goal=goal
        )
        messages = convert_langchain_to_llm_messages(prompt_messages)

        try:
            response = await self.llm_manager.generate_response(messages)
            parsed_response = self.extractor_parser.parse(str(response.content))
            if not isinstance(parsed_response, dict):
                raise ValueError("Extractor response must be a JSON object.")

            rational = parsed_response.get("rational", "")
            evidence = parsed_response.get("evidence", "")
            summary = parsed_response.get("summary", "")
            if not all(
                isinstance(value, str) and value.strip()
                for value in (rational, evidence, summary)
            ):
                raise ValueError(
                    "Extractor response must include non-empty rational, evidence, and summary fields."
                )

            return {
                "rational": rational.strip(),
                "evidence": evidence.strip(),
                "summary": summary.strip(),
            }
        except Exception as e:
            return {
                "rational": "",
                "evidence": "",
                "summary": f"Error extracting info: {str(e)}",
                "error": f"Error extracting info: {str(e)}",
            }

    async def python_interpreter(self, code: str) -> str:
        """
        Execute Python code.
        """
        return self.interpreter.execute(code)

    async def google_scholar(self, query: list[str]) -> dict[str, object]:
        """
        Search Google Scholar.
        Currently falls back to generic search_knowledge if specific API not available.
        """
        # For now, reuse search with academic domains
        if not self.tavily_client:
            return {
                "tool": "google_scholar",
                "error": "Tavily API key not configured.",
                "results": [],
            }

        knowledge_domains = [
            "scholar.google.com",
            "arxiv.org",
            "semanticscholar.org",
            "acm.org",
            "ieee.org",
        ]

        results: list[dict[str, str]] = []
        for q in query:
            try:
                response = self.tavily_client.search(
                    query=q,
                    search_depth="advanced",
                    max_results=5,
                    include_domains=knowledge_domains,
                )
                for res in response.get("results", []):
                    results.append(
                        {
                            "query": q,
                            "title": res["title"],
                            "url": res["url"],
                            "snippet": res["content"],
                        }
                    )
            except Exception as e:
                results.append(
                    {"query": q, "title": "", "url": "", "snippet": f"Error: {str(e)}"}
                )

        return {"tool": "google_scholar", "results": results}

    async def parse_file(self, files: list[str]) -> str:
        """
        Parse local files (PDF, DOCX, TXT, etc.).
        """
        results = []
        for filename in files:
            if not os.path.exists(filename):
                results.append(f"File {filename} not found.")
                continue

            try:
                content = ""
                ext = os.path.splitext(filename)[1].lower()

                if ext == ".pdf":
                    try:
                        from pypdf import PdfReader

                        reader = PdfReader(filename)
                        for page in reader.pages:
                            content += page.extract_text() + "\n"
                    except ImportError:
                        content = "Error: pypdf not installed."
                    except Exception as e:
                        content = f"Error reading PDF: {e}"

                elif ext == ".docx":
                    try:
                        import docx

                        doc = docx.Document(filename)
                        for para in doc.paragraphs:
                            content += para.text + "\n"
                    except ImportError:
                        content = "Error: python-docx not installed."
                    except Exception as e:
                        content = f"Error reading DOCX: {e}"

                else:
                    # Fallback to text
                    with open(filename, encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                if not content.startswith("Error"):
                    results.append(
                        f"File: {filename}\nContent:\n{content[:10000]}...(truncated)"
                    )
                else:
                    results.append(f"File: {filename}\n{content}")

            except Exception as e:
                results.append(f"Error processing file {filename}: {str(e)}")

        return "\n=======\n".join(results)
