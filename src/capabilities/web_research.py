"""
Web Research Capability

This module provides tools for real-time web research using Tavily.
"""

import os
import json
from typing import Dict, Any, Optional
from tavily import TavilyClient

class WebResearcher:
    """
    Provides web research capabilities including searching and content extraction.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the WebResearcher.
        
        Args:
            api_key: Tavily API key. If not provided, looks for TAVILY_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            # We don't raise error here to allow initialization, 
            # but methods will fail if called without key
            pass
            
        self.client = TavilyClient(api_key=self.api_key) if self.api_key else None

    def search(self, query: str, domains: Optional[list[str]] = None) -> str:
        """
        Search the web for a query and return relevant results.
        
        Args:
            query: The search query string.
            domains: Optional list of domains to restrict the search to.
            
        Returns:
            A JSON string containing search results with titles, urls, and snippets.
        """
        if not self.client:
            return "Error: Tavily API key not configured. Please set TAVILY_API_KEY environment variable."
            
        try:
            # Simple search optimized for context
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=5,
                include_domains=domains,
                include_answer=True
            )
            
            # Format results for LLM consumption
            results = []
            if response.get("answer"):
                results.append(f"Quick Answer: {response['answer']}")
                
            for result in response.get("results", []):
                results.append(f"Title: {result['title']}\nURL: {result['url']}\nSnippet: {result['content']}\n")
                
            return "\n---\n".join(results)
            
        except Exception as e:
            return f"Error performing search: {str(e)}"

    def search_knowledge(self, query: str) -> str:
        """
        Search specifically on knowledge platforms like Zhihu, Arxiv, and Google Scholar.
        """
        knowledge_domains = [
            "zhihu.com",
            "arxiv.org",
            "scholar.google.com",
            "semanticscholar.org"
        ]
        return self.search(query, domains=knowledge_domains)

    def browse(self, url: str) -> str:
        """
        Extract the main content from a specific webpage URL.
        
        Args:
            url: The URL to browse.
            
        Returns:
            The extracted text content of the page.
        """
        if not self.client:
            return "Error: Tavily API key not configured."
            
        try:
            # Use extract capability
            response = self.client.extract(urls=[url])
            
            if response.get("failed_results"):
                return f"Failed to extract content from {url}"
                
            results = response.get("results", [])
            if not results:
                return "No content found."
                
            content = results[0].get("raw_content") or results[0].get("content")
            
            # Truncate if too long (simple safety, though LLM context is large)
            if len(content) > 50000:
                content = content[:50000] + "...(truncated)"
                
            return f"Content from {url}:\n\n{content}"
            
        except Exception as e:
            return f"Error browsing URL: {str(e)}"

    def get_tools(self) -> Dict[str, Any]:
        """
        Return the tool definitions and implementations for registration.
        """
        return {
            "web_search": {
                "func": self.search,
                "schema": {
                    "name": "web_search",
                    "description": "Search the web for up-to-date information.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query"}
                        },
                        "required": ["query"]
                    }
                }
            },
            "web_search_knowledge": {
                "func": self.search_knowledge,
                "schema": {
                    "name": "web_search_knowledge",
                    "description": "Search specifically on knowledge platforms like Zhihu, Arxiv, and Google Scholar.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query"}
                        },
                        "required": ["query"]
                    }
                }
            },
            "web_browse": {
                "func": self.browse,
                "schema": {
                    "name": "web_browse",
                    "description": "Extract full content from a specific URL.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "The URL to visit"}
                        },
                        "required": ["url"]
                    }
                }
            }
        }
