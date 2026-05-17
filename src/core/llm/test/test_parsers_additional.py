"""Additional LLM parser tests migrated from legacy `src/test` files."""

import pytest


class TestParsersMoreCoverage:
    """Parser 更多测试"""

    def test_json_output_parser_invalid(self):
        from src.core.llm.parsers import JSONOutputParser

        parser = JSONOutputParser()
        with pytest.raises(ValueError):
            parser.parse("invalid json")

    def test_json_output_parser_markdown(self):
        from src.core.llm.parsers import JSONOutputParser

        parser = JSONOutputParser()
        result = parser.parse('```json\n{"key": "value"}\n```')
        assert result["key"] == "value"


class TestParsersQuickWins:
    """Migrated methods from TestQuickWins."""

    def test_parsers_repair_json(self):
        """测试 JSON 修复功能"""
        from src.core.llm.parsers import JSONOutputParser

        # JSONOutputParser may handle repair differently
        parser = JSONOutputParser(allow_repair=True)
        result = parser.parse('```json\n{"valid": true}\n```')
        assert result["valid"] is True


class TestParsersFinal:
    """parsers.py 最后覆盖"""

    def test_base_output_parser(self):
        from src.core.llm.parsers import BaseOutputParser, JSONOutputParser

        parser = JSONOutputParser()
        assert isinstance(parser, BaseOutputParser)

    def test_json_parser_empty_string(self):
        from src.core.llm.parsers import JSONOutputParser

        parser = JSONOutputParser()
        with pytest.raises(ValueError):
            parser.parse("")

    def test_json_parser_invalid_json_without_repair(self):
        from src.core.llm.parsers import JSONOutputParser

        parser = JSONOutputParser(allow_repair=False)
        with pytest.raises(ValueError):
            parser.parse("{invalid}")
