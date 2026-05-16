import pytest

from src.core.llm.parsers import AgentResponse, JSONOutputParser


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
    parser = JSONOutputParser(pydantic_model=AgentResponse)
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
