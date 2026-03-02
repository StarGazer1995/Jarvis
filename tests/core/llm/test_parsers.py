import pytest
import json
from src.core.llm.parsers import JSONOutputParser, AgentResponse
from pydantic import ValidationError


def test_json_output_parser_simple():
    parser = JSONOutputParser()
    text = '{"thought": "foo", "type": "answer", "content": "bar"}'
    result = parser.parse(text)
    assert result == {"thought": "foo", "type": "answer", "content": "bar"}


def test_json_output_parser_markdown():
    parser = JSONOutputParser()
    text = """
    Here is the json:
    ```json
    {
        "thought": "thinking...",
        "type": "tool_call",
        "content": {"name": "test", "arguments": {}}
    }
    ```
    """
    result = parser.parse(text)
    assert result["thought"] == "thinking..."
    assert result["type"] == "tool_call"


def test_json_output_parser_pydantic():
    parser = JSONOutputParser(pydantic_model=AgentResponse)
    text = '{"thought": "foo", "type": "answer", "content": "bar"}'
    result = parser.parse(text)
    assert isinstance(result, AgentResponse)
    assert result.thought == "foo"
    assert result.type == "answer"


def test_json_output_parser_invalid_json():
    parser = JSONOutputParser()
    text = '{"thought": "foo", "type": "answer", "content": "bar"'
    with pytest.raises(ValueError, match="Invalid JSON output"):
        parser.parse(text)


def test_json_output_parser_schema_fail():
    parser = JSONOutputParser(pydantic_model=AgentResponse)
    text = '{"thought": "foo"}'  # Missing required fields
    with pytest.raises(ValueError, match="Schema validation failed"):
        parser.parse(text)
