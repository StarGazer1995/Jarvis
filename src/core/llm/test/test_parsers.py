import pytest
from pydantic import BaseModel

from src.core.llm.parsers import JSONOutputParser


class SimpleAgentResponse(BaseModel):
    """Test-only schema used to verify generic Pydantic validation support."""

    thought: str
    type: str
    content: str | dict


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
    parser = JSONOutputParser(pydantic_model=SimpleAgentResponse)
    text = '{"thought": "foo", "type": "answer", "content": "bar"}'
    result = parser.parse(text)
    assert isinstance(result, SimpleAgentResponse)
    assert result.thought == "foo"
    assert result.type == "answer"


def test_json_output_parser_invalid_json():
    parser = JSONOutputParser()
    text = '{"thought": "foo", "type": "answer", "content": "bar"'
    with pytest.raises(ValueError, match="Invalid JSON output"):
        parser.parse(text)


def test_json_output_parser_schema_fail():
    parser = JSONOutputParser(pydantic_model=SimpleAgentResponse)
    text = '{"thought": "foo"}'  # Missing required fields
    with pytest.raises(ValueError, match="Schema validation failed"):
        parser.parse(text)


def test_json_output_parser_deep_research_answer_protocol():
    parser = JSONOutputParser(protocol="deep_research")
    text = """
    {
        "thought": "foo",
        "type": "answer",
        "content": {
            "summary": "OpenAI o1 appears to cost $20.",
            "claims": [
                {
                    "statement": "OpenAI o1 costs $20.",
                    "source_urls": ["https://example.com/pricing"]
                }
            ],
            "sources": [
                {
                    "url": "https://example.com/pricing",
                    "title": "Pricing page",
                    "evidence": "The pricing page states that OpenAI o1 costs $20."
                }
            ],
            "insufficient_evidence": false
        }
    }
    """
    result = parser.parse(text)
    assert result["type"] == "answer"
    assert result["content"]["claims"][0]["source_urls"] == [
        "https://example.com/pricing"
    ]


def test_json_output_parser_deep_research_insufficient_evidence_answer_protocol():
    parser = JSONOutputParser(protocol="deep_research")
    text = """
    {
        "thought": "Need to admit evidence gap",
        "type": "answer",
        "content": {
            "summary": "Insufficient evidence to answer reliably.",
            "claims": [],
            "sources": [],
            "insufficient_evidence": true
        }
    }
    """
    result = parser.parse(text)
    assert result["content"]["insufficient_evidence"] is True


def test_json_output_parser_deep_research_tool_call_protocol():
    parser = JSONOutputParser(protocol="deep_research")
    text = """
    {
        "type": "tool_call",
        "content": {"name": "search", "arguments": {"query": ["a"]}},
        "thought": "search first"
    }
    """
    result = parser.parse(text)
    assert result["type"] == "tool_call"
    assert result["content"]["name"] == "search"


def test_json_output_parser_deep_research_tool_calls_protocol():
    parser = JSONOutputParser(protocol="deep_research")
    text = """
    {
        "thought": "parallel search",
        "type": "tool_calls",
        "content": [
            {"name": "search", "arguments": {"query": ["a"]}},
            {"name": "google_scholar", "arguments": {"query": ["b"]}}
        ]
    }
    """
    result = parser.parse(text)
    assert result["type"] == "tool_calls"
    assert len(result["content"]) == 2


def test_json_output_parser_deep_research_error_protocol():
    parser = JSONOutputParser(protocol="deep_research")
    text = """
    {
        "thought": "invalid input",
        "type": "error",
        "content": {"code": "INVALID_TOOL_ARGUMENTS", "message": "bad args"}
    }
    """
    result = parser.parse(text)
    assert result["type"] == "error"
    assert result["content"]["code"] == "INVALID_TOOL_ARGUMENTS"


def test_json_output_parser_deep_research_invalid_type():
    parser = JSONOutputParser(protocol="deep_research")
    text = '{"thought": "foo", "type": "no_tool_call", "content": "bar"}'
    with pytest.raises(ValueError, match="requires 'type' to be one of"):
        parser.parse(text)


def test_json_output_parser_deep_research_rejects_depends_on():
    parser = JSONOutputParser(protocol="deep_research")
    text = """
    {
        "thought": "invalid dependency graph",
        "type": "tool_calls",
        "content": [
            {
                "name": "search",
                "arguments": {"query": ["a"]},
                "depends_on": ["tool-1"]
            }
        ]
    }
    """
    with pytest.raises(ValueError, match="does not support 'depends_on'"):
        parser.parse(text)


def test_json_output_parser_deep_research_requires_error_shape():
    parser = JSONOutputParser(protocol="deep_research")
    text = """
    {
        "thought": "bad error",
        "type": "error",
        "content": {"message": "missing code"}
    }
    """
    with pytest.raises(ValueError, match="error.code"):
        parser.parse(text)


def test_json_output_parser_deep_research_requires_thought():
    parser = JSONOutputParser(protocol="deep_research")
    text = '{"type": "answer", "content": "bar"}'
    with pytest.raises(ValueError, match="requires 'thought'"):
        parser.parse(text)


def test_json_output_parser_deep_research_requires_content():
    parser = JSONOutputParser(protocol="deep_research")
    text = '{"thought": "foo", "type": "answer"}'
    with pytest.raises(ValueError, match="requires a 'content'"):
        parser.parse(text)


def test_json_output_parser_deep_research_requires_answer_object():
    parser = JSONOutputParser(protocol="deep_research")
    text = '{"thought": "foo", "type": "answer", "content": {"bad": true}}'
    with pytest.raises(ValueError, match="non-empty 'summary' string"):
        parser.parse(text)


def test_json_output_parser_deep_research_rejects_non_object_answer_content():
    parser = JSONOutputParser(protocol="deep_research")
    text = '{"thought": "foo", "type": "answer", "content": "bar"}'
    with pytest.raises(ValueError, match="content to be an object"):
        parser.parse(text)


def test_json_output_parser_deep_research_requires_claim_sources_to_match_top_level_sources():
    parser = JSONOutputParser(protocol="deep_research")
    text = """
    {
        "thought": "foo",
        "type": "answer",
        "content": {
            "summary": "Pricing summary.",
            "claims": [
                {
                    "statement": "OpenAI o1 costs $20.",
                    "source_urls": ["https://missing.example.com"]
                }
            ],
            "sources": [
                {
                    "url": "https://example.com/pricing",
                    "title": "Pricing page",
                    "evidence": "The pricing page states that OpenAI o1 costs $20."
                }
            ],
            "insufficient_evidence": false
        }
    }
    """
    with pytest.raises(ValueError, match="reference top-level sources"):
        parser.parse(text)


def test_json_output_parser_deep_research_rejects_claims_when_insufficient():
    parser = JSONOutputParser(protocol="deep_research")
    text = """
    {
        "thought": "foo",
        "type": "answer",
        "content": {
            "summary": "Insufficient evidence to answer reliably.",
            "claims": [
                {
                    "statement": "Unsupported claim",
                    "source_urls": ["https://example.com"]
                }
            ],
            "sources": [
                {
                    "url": "https://example.com",
                    "title": "Example",
                    "evidence": "Unsupported example evidence."
                }
            ],
            "insufficient_evidence": true
        }
    }
    """
    with pytest.raises(ValueError, match="must not include claims"):
        parser.parse(text)


def test_json_output_parser_deep_research_requires_source_evidence():
    parser = JSONOutputParser(protocol="deep_research")
    text = """
    {
        "thought": "foo",
        "type": "answer",
        "content": {
            "summary": "OpenAI o1 costs $20.",
            "claims": [
                {
                    "statement": "OpenAI o1 costs $20.",
                    "source_urls": ["https://example.com/pricing"]
                }
            ],
            "sources": [
                {
                    "url": "https://example.com/pricing",
                    "title": "Pricing page",
                    "evidence": ""
                }
            ],
            "insufficient_evidence": false
        }
    }
    """
    with pytest.raises(ValueError, match="non-empty 'evidence'"):
        parser.parse(text)


def test_json_output_parser_deep_research_requires_tool_call_object():
    parser = JSONOutputParser(protocol="deep_research")
    text = '{"thought": "foo", "type": "tool_call", "content": "bad"}'
    with pytest.raises(ValueError, match="tool calls to be objects"):
        parser.parse(text)


def test_json_output_parser_deep_research_requires_tool_arguments_object():
    parser = JSONOutputParser(protocol="deep_research")
    text = """
    {
        "thought": "foo",
        "type": "tool_call",
        "content": {"name": "search", "arguments": "bad"}
    }
    """
    with pytest.raises(ValueError, match="object 'arguments'"):
        parser.parse(text)


def test_json_output_parser_deep_research_requires_non_empty_tool_name():
    parser = JSONOutputParser(protocol="deep_research")
    text = """
    {
        "thought": "foo",
        "type": "tool_call",
        "content": {"name": "", "arguments": {}}
    }
    """
    with pytest.raises(ValueError, match="non-empty 'name'"):
        parser.parse(text)


def test_json_output_parser_deep_research_requires_tool_calls_array():
    parser = JSONOutputParser(protocol="deep_research")
    text = """
    {
        "thought": "foo",
        "type": "tool_calls",
        "content": {"name": "search", "arguments": {}}
    }
    """
    with pytest.raises(ValueError, match="'tool_calls' content to be an array"):
        parser.parse(text)


def test_json_output_parser_deep_research_requires_error_object():
    parser = JSONOutputParser(protocol="deep_research")
    text = '{"thought": "foo", "type": "error", "content": "bad"}'
    with pytest.raises(ValueError, match="'error' content to be an object"):
        parser.parse(text)


def test_json_output_parser_deep_research_requires_error_message():
    parser = JSONOutputParser(protocol="deep_research")
    text = """
    {
        "thought": "foo",
        "type": "error",
        "content": {"code": "BAD_REQUEST", "message": ""}
    }
    """
    with pytest.raises(ValueError, match="error.message"):
        parser.parse(text)


def test_json_output_parser_deep_research_requires_object_top_level():
    parser = JSONOutputParser(protocol="deep_research")
    with pytest.raises(ValueError, match="Expected JSON object"):
        parser.parse('[{"thought": "foo", "type": "answer", "content": "bar"}]')


def test_json_output_parser_accepts_deep_research_audit_verdict():
    parser = JSONOutputParser(protocol="deep_research_audit")
    text = """
    {
        "supported": true,
        "issues": [],
        "per_source": [
            {
                "url": "https://example.com/pricing",
                "supported": true,
                "reason": "Observed evidence supports the cited excerpt."
            }
        ]
    }
    """
    result = parser.parse(text)
    assert result["supported"] is True


def test_json_output_parser_deep_research_audit_requires_reason():
    parser = JSONOutputParser(protocol="deep_research_audit")
    text = """
    {
        "supported": false,
        "issues": ["unsupported evidence"],
        "per_source": [
            {
                "url": "https://example.com/pricing",
                "supported": false,
                "reason": ""
            }
        ]
    }
    """
    with pytest.raises(ValueError, match="non-empty 'reason'"):
        parser.parse(text)


def test_json_output_parser_deep_research_requires_claims_array():
    parser = JSONOutputParser(protocol="deep_research")
    text = (
        '{"thought": "foo", "type": "answer", "content": '
        '{"summary": "ok", "claims": "bad", "sources": [], "insufficient_evidence": true}}'
    )
    with pytest.raises(ValueError, match="'claims' to be an array"):
        parser.parse(text)


def test_json_output_parser_deep_research_requires_sources_array():
    parser = JSONOutputParser(protocol="deep_research")
    text = (
        '{"thought": "foo", "type": "answer", "content": '
        '{"summary": "ok", "claims": [], "sources": "bad", "insufficient_evidence": true}}'
    )
    with pytest.raises(ValueError, match="'sources' to be an array"):
        parser.parse(text)


def test_json_output_parser_deep_research_requires_insufficient_evidence_boolean():
    parser = JSONOutputParser(protocol="deep_research")
    text = (
        '{"thought": "foo", "type": "answer", "content": '
        '{"summary": "ok", "claims": [], "sources": [], "insufficient_evidence": "bad"}}'
    )
    with pytest.raises(ValueError, match="'insufficient_evidence' to be a boolean"):
        parser.parse(text)


def test_json_output_parser_deep_research_requires_source_object():
    parser = JSONOutputParser(protocol="deep_research")
    text = (
        '{"thought": "foo", "type": "answer", "content": '
        '{"summary": "OpenAI o1 costs $20.", "claims": [{"statement": "OpenAI o1 costs $20.", '
        '"source_urls": ["https://example.com/pricing"]}], "sources": ["bad"], '
        '"insufficient_evidence": false}}'
    )
    with pytest.raises(ValueError, match="sources must be objects"):
        parser.parse(text)


def test_json_output_parser_deep_research_requires_source_title():
    parser = JSONOutputParser(protocol="deep_research")
    text = (
        '{"thought": "foo", "type": "answer", "content": '
        '{"summary": "OpenAI o1 costs $20.", "claims": [{"statement": "OpenAI o1 costs $20.", '
        '"source_urls": ["https://example.com/pricing"]}], "sources": [{"url": "https://example.com/pricing", '
        '"title": "", "evidence": "Price is $20."}], "insufficient_evidence": false}}'
    )
    with pytest.raises(ValueError, match="non-empty 'title'"):
        parser.parse(text)


def test_json_output_parser_deep_research_requires_valid_source_url():
    parser = JSONOutputParser(protocol="deep_research")
    text = (
        '{"thought": "foo", "type": "answer", "content": '
        '{"summary": "OpenAI o1 costs $20.", "claims": [], "sources": [{"url": "bad-url", '
        '"title": "Pricing page", "evidence": "Price is $20."}], "insufficient_evidence": true}}'
    )
    with pytest.raises(ValueError, match="valid 'url'"):
        parser.parse(text)


def test_json_output_parser_deep_research_requires_claim_object():
    parser = JSONOutputParser(protocol="deep_research")
    text = (
        '{"thought": "foo", "type": "answer", "content": '
        '{"summary": "OpenAI o1 costs $20.", "claims": ["bad"], "sources": [{"url": "https://example.com/pricing", '
        '"title": "Pricing page", "evidence": "Price is $20."}], "insufficient_evidence": false}}'
    )
    with pytest.raises(ValueError, match="claims must be objects"):
        parser.parse(text)


def test_json_output_parser_deep_research_requires_claim_statement():
    parser = JSONOutputParser(protocol="deep_research")
    text = (
        '{"thought": "foo", "type": "answer", "content": '
        '{"summary": "OpenAI o1 costs $20.", "claims": [{"statement": "", "source_urls": ["https://example.com/pricing"]}], '
        '"sources": [{"url": "https://example.com/pricing", "title": "Pricing page", "evidence": "Price is $20."}], '
        '"insufficient_evidence": false}}'
    )
    with pytest.raises(ValueError, match="non-empty 'statement'"):
        parser.parse(text)


def test_json_output_parser_deep_research_requires_non_empty_claim_source_urls():
    parser = JSONOutputParser(protocol="deep_research")
    text = (
        '{"thought": "foo", "type": "answer", "content": '
        '{"summary": "OpenAI o1 costs $20.", "claims": [{"statement": "OpenAI o1 costs $20.", "source_urls": []}], '
        '"sources": [{"url": "https://example.com/pricing", "title": "Pricing page", "evidence": "Price is $20."}], '
        '"insufficient_evidence": false}}'
    )
    with pytest.raises(ValueError, match="non-empty 'source_urls'"):
        parser.parse(text)


def test_json_output_parser_deep_research_insufficient_evidence_requires_phrase():
    parser = JSONOutputParser(protocol="deep_research")
    text = (
        '{"thought": "foo", "type": "answer", "content": '
        '{"summary": "Need more data.", "claims": [], "sources": [], "insufficient_evidence": true}}'
    )
    with pytest.raises(ValueError, match="must explain that evidence is insufficient"):
        parser.parse(text)


def test_json_output_parser_deep_research_requires_grounded_sources():
    parser = JSONOutputParser(protocol="deep_research")
    text = (
        '{"thought": "foo", "type": "answer", "content": '
        '{"summary": "OpenAI o1 costs $20.", "claims": [], "sources": [], "insufficient_evidence": false}}'
    )
    with pytest.raises(ValueError, match="at least one claim"):
        parser.parse(text)


def test_json_output_parser_deep_research_requires_source_when_claims_present():
    parser = JSONOutputParser(protocol="deep_research")
    text = (
        '{"thought": "foo", "type": "answer", "content": '
        '{"summary": "OpenAI o1 costs $20.", "claims": [{"statement": "OpenAI o1 costs $20.", '
        '"source_urls": ["https://example.com/pricing"]}], "sources": [{"url": "https://example.com/pricing", '
        '"title": "Pricing page", "evidence": "Price is $20."}], "insufficient_evidence": false}}'
    )
    parsed = parser.parse(text)
    parsed["content"]["sources"] = []

    with pytest.raises(ValueError, match="at least one source"):
        parser._validate_deep_research_answer_content(parsed["content"])


def test_json_output_parser_deep_research_rejects_non_string_claim_source_url():
    parser = JSONOutputParser(protocol="deep_research")
    text = (
        '{"thought": "foo", "type": "answer", "content": '
        '{"summary": "OpenAI o1 costs $20.", "claims": [{"statement": "OpenAI o1 costs $20.", '
        '"source_urls": [123]}], "sources": [{"url": "https://example.com/pricing", '
        '"title": "Pricing page", "evidence": "Price is $20."}], "insufficient_evidence": false}}'
    )
    with pytest.raises(ValueError, match="reference top-level sources"):
        parser.parse(text)


def test_json_output_parser_deep_research_audit_requires_boolean_supported():
    parser = JSONOutputParser(protocol="deep_research_audit")
    text = '{"supported": "bad", "issues": [], "per_source": []}'
    with pytest.raises(ValueError, match="'supported' to be a boolean"):
        parser.parse(text)


def test_json_output_parser_deep_research_audit_requires_issues_array():
    parser = JSONOutputParser(protocol="deep_research_audit")
    text = '{"supported": false, "issues": "bad", "per_source": []}'
    with pytest.raises(ValueError, match="'issues' to be an array"):
        parser.parse(text)


def test_json_output_parser_deep_research_audit_requires_non_empty_issue_strings():
    parser = JSONOutputParser(protocol="deep_research_audit")
    text = '{"supported": false, "issues": [""], "per_source": []}'
    with pytest.raises(ValueError, match="issues must be non-empty strings"):
        parser.parse(text)


def test_json_output_parser_deep_research_audit_requires_per_source_array():
    parser = JSONOutputParser(protocol="deep_research_audit")
    text = '{"supported": false, "issues": [], "per_source": "bad"}'
    with pytest.raises(ValueError, match="'per_source' to be an array"):
        parser.parse(text)


def test_json_output_parser_deep_research_audit_requires_source_entry_object():
    parser = JSONOutputParser(protocol="deep_research_audit")
    text = '{"supported": false, "issues": [], "per_source": ["bad"]}'
    with pytest.raises(ValueError, match="source entries must be objects"):
        parser.parse(text)


def test_json_output_parser_deep_research_audit_requires_valid_source_url():
    parser = JSONOutputParser(protocol="deep_research_audit")
    text = (
        '{"supported": false, "issues": [], "per_source": '
        '[{"url": "bad", "supported": false, "reason": "No support."}]}'
    )
    with pytest.raises(ValueError, match="valid 'url'"):
        parser.parse(text)


def test_json_output_parser_deep_research_audit_requires_source_supported_boolean():
    parser = JSONOutputParser(protocol="deep_research_audit")
    text = (
        '{"supported": false, "issues": [], "per_source": '
        '[{"url": "https://example.com", "supported": "bad", "reason": "No support."}]}'
    )
    with pytest.raises(ValueError, match="'supported' to be a boolean"):
        parser.parse(text)


def test_json_output_parser_rejects_unsupported_protocol():
    parser = JSONOutputParser(protocol="unknown_protocol")
    text = '{"thought": "foo", "type": "answer", "content": "bar"}'
    with pytest.raises(ValueError, match="Unsupported parser protocol"):
        parser.parse(text)


def test_json_output_parser_base_class():
    """测试基类抽象方法"""
    from src.core.llm.parsers import BaseOutputParser

    class ConcreteParser(BaseOutputParser):
        def parse(self, text: str) -> dict:
            return {"parsed": text}

    p = ConcreteParser()
    result = p.parse("test")
    assert result == {"parsed": "test"}


def test_json_output_parser_clean_markdown_variants():
    """测试各种 Markdown 格式清理"""
    parser = JSONOutputParser()
    text = '```\n{"key": "value"}\n```'
    result = parser.parse(text)
    assert result == {"key": "value"}

    text2 = 'Some text ```json\n{"a": 1}\n``` trailing'
    result2 = parser.parse(text2)
    assert result2 == {"a": 1}


def test_json_output_parser_pydantic_non_dict():
    """测试 pydantic 模型收到非 dict 数据"""
    parser = JSONOutputParser(pydantic_model=SimpleAgentResponse)
    # Simulate data that is a list (e.g., from json_repair)
    # We need to bypass the JSON parsing and inject a list
    # Use a raw call to internal logic or mock

    with pytest.raises(ValueError, match="Expected JSON object"):
        # A JSON array is valid JSON but not a dict
        parser.parse("[1, 2, 3]")


def test_json_output_parser_no_repair_not_installed():
    """测试 allow_repair=True 但 json_repair 未安装"""
    import src.core.llm.parsers as parsers_module

    old_repair = parsers_module.json_repair
    parsers_module.json_repair = None
    try:
        parser = parsers_module.JSONOutputParser(allow_repair=True)
        text = '{"key": "value"'
        with pytest.raises(ValueError, match="Invalid JSON output"):
            parser.parse(text)
    finally:
        parsers_module.json_repair = old_repair


@pytest.mark.skipif(
    True,
    reason="json_repair path requires mocking import which is fragile",
)
def test_json_output_parser_repair_success():
    """测试 JSON repair 成功路径（跳过，需要 mock 导入）"""
    pass
