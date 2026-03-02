import pytest
from src.web.utils import clean_llm_response


def test_clean_llm_response_simple():
    assert clean_llm_response("hello") == "hello"


def test_clean_llm_response_json_block():
    response = '```json\n{"content": "hello"}\n```'
    assert clean_llm_response(response) == "hello"


def test_clean_llm_response_json_no_block():
    response = '{"content": "hello"}'
    # It parses JSON correctly even without code blocks
    assert clean_llm_response(response) == "hello"


def test_clean_llm_response_invalid_json():
    response = "```json\n{invalid}\n```"
    # Parsing fails, returns original including backticks
    assert clean_llm_response(response) == response


def test_clean_llm_response_empty():
    assert clean_llm_response("") == ""
    assert clean_llm_response(None) == ""


def test_clean_llm_response_no_content_field():
    response = '```json\n{"other": "value"}\n```'
    # Parsed but no "content", so regex logic ends, try block finishes
    # Code:
    # if isinstance(parsed, dict) and "content" in parsed: return ...
    # Fall through to return response (original arg? No, return cleaned?)
    # Wait, let's check code again.
    # match = re.search(...) -> cleaned = match.group(1)
    # try: parsed = ...; if content in parsed: return content
    # except: pass
    # return response (The ORIGINAL argument)

    # So if it's valid JSON but no content field, it returns the ORIGINAL full string with backticks.
    assert clean_llm_response(response) == response
