import json
import re


def clean_llm_response(response: str) -> str:
    """
    Clean LLM response by attempting to parse JSON and extract 'content'.
    Handles cases where response is wrapped in markdown code blocks.
    """
    if not response:
        return ""

    cleaned = response.strip()

    # Remove markdown code blocks if present
    # e.g. ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "content" in parsed:
            return parsed["content"]
    except json.JSONDecodeError:
        pass

    return response
