"""
Deep Research Tools
Implementation of tools required for the Deep Research agent.
"""

import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
import requests

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

from src.core.llm.client import LLMManager, LLMMessage
from src.core.llm.converters import convert_langchain_to_llm_messages
from src.core.llm.config import load_llm_config
from src.core.prompt.manager import PromptManager
from src.capabilities.interpreter import PythonInterpreter

logger = logging.getLogger(__name__)


class DeepResearchTools:
    """
    Tools for Deep Research Agent.
    """

    def __init__(self, llm_manager: Optional[LLMManager] = None):
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

    async def search(self, query: List[str]) -> str:
        """
        Perform web searches for multiple queries.
        """
        if not self.tavily_client:
            return "Error: Tavily API key not configured."

        results = []
        for q in query:
            try:
                response = self.tavily_client.search(
                    query=q, search_depth="advanced", max_results=5, include_answer=True
                )

                query_result = f"Query: {q}\n"
                if response.get("answer"):
                    query_result += f"Quick Answer: {response['answer']}\n"

                for res in response.get("results", []):
                    query_result += f"Title: {res['title']}\nURL: {res['url']}\nSnippet: {res['content']}\n"

                results.append(query_result)
            except Exception as e:
                results.append(f"Error searching for '{q}': {str(e)}")

        return "\n=======\n".join(results)

    async def visit(self, url: List[str], goal: str) -> str:
        """
        Visit webpages and extract information based on the goal.
        """
        results = []
        for u in url:
            try:
                content = await self._fetch_page_content(u)
                if content.startswith("Error"):
                    results.append(f"URL: {u}\nResult: {content}")
                    continue

                # Summarize/Extract using LLM
                summary = await self._extract_info(content, goal)
                results.append(f"URL: {u}\n{summary}")
            except Exception as e:
                results.append(f"URL: {u}\nError: {str(e)}")

        return "\n=======\n".join(results)

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

    async def _extract_info(self, content: str, goal: str) -> str:
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
            return response.content
        except Exception as e:
            return f"Error extracting info: {str(e)}"

    async def python_interpreter(self, code: str) -> str:
        """
        Execute Python code.
        """
        return self.interpreter.execute(code)

    async def google_scholar(self, query: List[str]) -> str:
        """
        Search Google Scholar.
        Currently falls back to generic search_knowledge if specific API not available.
        """
        # For now, reuse search with academic domains
        if not self.tavily_client:
            return "Error: Tavily API key not configured."

        knowledge_domains = [
            "scholar.google.com",
            "arxiv.org",
            "semanticscholar.org",
            "acm.org",
            "ieee.org",
        ]

        results = []
        for q in query:
            try:
                response = self.tavily_client.search(
                    query=q,
                    search_depth="advanced",
                    max_results=5,
                    include_domains=knowledge_domains,
                )
                # ... format results same as search ...
                query_result = f"Query: {q}\n"
                for res in response.get("results", []):
                    query_result += f"Title: {res['title']}\nURL: {res['url']}\nSnippet: {res['content']}\n"
                results.append(query_result)
            except Exception as e:
                results.append(f"Error searching scholar for '{q}': {str(e)}")

        return "\n=======\n".join(results)

    async def parse_file(self, files: List[str]) -> str:
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
                    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
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
