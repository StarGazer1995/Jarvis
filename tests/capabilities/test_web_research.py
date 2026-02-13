import pytest
import os
from unittest.mock import Mock, patch
from src.capabilities.web_research import WebResearcher

class TestWebResearcher:
    
    @pytest.fixture
    def mock_tavily(self):
        with patch("src.capabilities.web_research.TavilyClient") as mock:
            yield mock

    def test_initialization_with_key(self, mock_tavily):
        """Test initialization with explicit API key."""
        researcher = WebResearcher(api_key="test-key")
        assert researcher.api_key == "test-key"
        mock_tavily.assert_called_once_with(api_key="test-key")

    def test_initialization_with_env_var(self, mock_tavily):
        """Test initialization with environment variable."""
        with patch.dict(os.environ, {"TAVILY_API_KEY": "env-key"}):
            researcher = WebResearcher()
            assert researcher.api_key == "env-key"
            mock_tavily.assert_called_once_with(api_key="env-key")

    def test_initialization_without_key(self, mock_tavily):
        """Test initialization without any key."""
        with patch.dict(os.environ, {}, clear=True):
            researcher = WebResearcher()
            assert researcher.api_key is None
            assert researcher.client is None

    def test_search_success(self, mock_tavily):
        """Test successful search."""
        mock_client = mock_tavily.return_value
        mock_client.search.return_value = {
            "answer": "A quick answer",
            "results": [
                {"title": "Result 1", "url": "http://1.com", "content": "Content 1"},
                {"title": "Result 2", "url": "http://2.com", "content": "Content 2"}
            ]
        }
        
        researcher = WebResearcher(api_key="test")
        result = researcher.search("query")
        
        assert "Quick Answer: A quick answer" in result
        assert "Title: Result 1" in result
        assert "URL: http://1.com" in result
        assert "Snippet: Content 1" in result
        mock_client.search.assert_called_with(
            query="query",
            search_depth="advanced",
            max_results=5,
            include_domains=None,
            include_answer=True
        )

    def test_search_knowledge_domains(self, mock_tavily):
        """Test search with knowledge domains."""
        mock_client = mock_tavily.return_value
        mock_client.search.return_value = {"results": []}
        
        researcher = WebResearcher(api_key="test")
        researcher.search_knowledge("query")
        
        call_args = mock_client.search.call_args[1]
        assert "arxiv.org" in call_args["include_domains"]
        assert "scholar.google.com" in call_args["include_domains"]

    def test_search_missing_key(self, mock_tavily):
        """Test search without API key."""
        # Force initialization with None key and None client
        with patch.dict(os.environ, {}, clear=True):
            researcher = WebResearcher(api_key=None)
            # Ensure client is None even if environment leaks
            researcher.client = None 
            
            result = researcher.search("query")
            assert "Error: Tavily API key not configured" in result

    def test_search_error(self, mock_tavily):
        """Test search exception handling."""
        mock_client = mock_tavily.return_value
        mock_client.search.side_effect = Exception("Search failed")
        
        researcher = WebResearcher(api_key="test")
        result = researcher.search("query")
        
        assert "Error performing search: Search failed" in result

    def test_browse_success(self, mock_tavily):
        """Test successful browsing/extraction."""
        mock_client = mock_tavily.return_value
        mock_client.extract.return_value = {
            "results": [{"raw_content": "Full page content"}]
        }
        
        researcher = WebResearcher(api_key="test")
        result = researcher.browse("http://example.com")
        
        assert "Content from http://example.com" in result
        assert "Full page content" in result
        mock_client.extract.assert_called_with(urls=["http://example.com"])

    def test_browse_fallback_content(self, mock_tavily):
        """Test browsing fallback to 'content' if 'raw_content' missing."""
        mock_client = mock_tavily.return_value
        mock_client.extract.return_value = {
            "results": [{"content": "Fallback content"}]
        }
        
        researcher = WebResearcher(api_key="test")
        result = researcher.browse("http://example.com")
        
        assert "Fallback content" in result

    def test_browse_failed_results(self, mock_tavily):
        """Test browsing with failed results."""
        mock_client = mock_tavily.return_value
        mock_client.extract.return_value = {
            "failed_results": ["http://example.com"]
        }
        
        researcher = WebResearcher(api_key="test")
        result = researcher.browse("http://example.com")
        
        assert "Failed to extract content" in result

    def test_browse_no_results(self, mock_tavily):
        """Test browsing with empty results."""
        mock_client = mock_tavily.return_value
        mock_client.extract.return_value = {"results": []}
        
        researcher = WebResearcher(api_key="test")
        result = researcher.browse("http://example.com")
        
        assert "No content found" in result

    def test_browse_truncation(self, mock_tavily):
        """Test content truncation."""
        mock_client = mock_tavily.return_value
        long_content = "a" * 60000
        mock_client.extract.return_value = {
            "results": [{"raw_content": long_content}]
        }
        
        researcher = WebResearcher(api_key="test")
        result = researcher.browse("http://example.com")
        
        assert len(result) < 60000
        assert "...(truncated)" in result

    def test_browse_missing_key(self, mock_tavily):
        """Test browse without API key."""
        with patch.dict(os.environ, {}, clear=True):
            researcher = WebResearcher(api_key=None)
            researcher.client = None
            
            result = researcher.browse("http://example.com")
            assert "Error: Tavily API key not configured" in result

    def test_browse_error(self, mock_tavily):
        """Test browse exception handling."""
        mock_client = mock_tavily.return_value
        mock_client.extract.side_effect = Exception("Extract failed")
        
        researcher = WebResearcher(api_key="test")
        result = researcher.browse("http://example.com")
        
        assert "Error browsing URL: Extract failed" in result

    def test_get_tools(self, mock_tavily):
        """Test get_tools returns correct schema."""
        researcher = WebResearcher(api_key="test")
        tools = researcher.get_tools()
        
        assert "web_search" in tools
        assert "web_search_knowledge" in tools
        assert "web_browse" in tools
        
        assert tools["web_search"]["func"] == researcher.search
        assert tools["web_browse"]["func"] == researcher.browse
        
        schema = tools["web_search"]["schema"]
        assert schema["name"] == "web_search"
        assert "query" in schema["input_schema"]["properties"]
