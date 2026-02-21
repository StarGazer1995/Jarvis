"""
LLM Output Parsers

This module provides parsers for LLM responses, specifically handling JSON output
and ensuring strict schema validation.
"""

import json
import logging
import re
from typing import Any, Dict, Optional, Union, List, Type
from abc import ABC, abstractmethod
from pydantic import BaseModel, ValidationError

try:
    import json_repair
except ImportError:
    json_repair = None

class BaseOutputParser(ABC):
    """Base class for output parsers."""
    
    @abstractmethod
    def parse(self, text: str) -> Any:
        """Parse the output text."""
        pass

class JSONOutputParser(BaseOutputParser):
    """
    Parses JSON output from LLM responses.
    
    Features:
    - Handles Markdown code blocks (```json ... ```)
    - strict schema validation (optional)
    - robust error handling using json_repair (if installed)
    """
    
    def __init__(self, pydantic_model: Optional[Type[BaseModel]] = None, allow_repair: bool = False):
        """
        Initialize the parser.
        
        Args:
            pydantic_model: Optional Pydantic model to validate against.
            allow_repair: Whether to attempt JSON repair when parsing fails.
        """
        self.pydantic_model = pydantic_model
        self.allow_repair = allow_repair
        self.logger = logging.getLogger("llm.parsers.json")
        
    def parse(self, text: str) -> Union[Dict[str, Any], BaseModel]:
        """
        Parse the text into a dictionary or Pydantic model.
        
        Args:
            text: The text to parse.
            
        Returns:
            Parsed dictionary or Pydantic model instance.
            
        Raises:
            ValueError: If parsing fails.
        """
        cleaned_text = self._clean_json_text(text)
        
        try:
            # First attempt: standard json.loads
            data = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            # Second attempt: use json_repair if available
            if json_repair and self.allow_repair:
                try:
                    self.logger.warning(f"Standard JSON parse failed, attempting repair: {e}")
                    decoded_object = json_repair.repair_json(cleaned_text, return_objects=True)
                    if isinstance(decoded_object, (dict, list)):
                        data = decoded_object
                    else:
                        raise ValueError(f"Repair returned non-JSON object: {type(decoded_object)}")
                except Exception as repair_error:
                    self.logger.error(f"JSON repair failed: {repair_error}\nText: {text}")
                    raise ValueError(f"Invalid JSON output (repair failed): {repair_error}")
            else:
                self.logger.error(f"Failed to parse JSON and json_repair not installed: {e}\nText: {text}")
                raise ValueError(f"Invalid JSON output: {e}")
            
        if self.pydantic_model:
            try:
                # If data is a list (e.g. from json_repair), validation might fail if model expects dict
                if not isinstance(data, dict):
                     raise ValueError(f"Expected JSON object, got {type(data)}")
                return self.pydantic_model.model_validate(data)
            except ValidationError as e:
                self.logger.error(f"Schema validation failed: {e}\nData: {data}")
                raise ValueError(f"Schema validation failed: {e}")
                
        return data
        
    def _clean_json_text(self, text: str) -> str:
        """
        Clean the text to extract JSON.
        
        - Removes Markdown code blocks.
        - Strips whitespace.
        """
        text = text.strip()
        
        # Check for markdown code blocks
        json_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if json_block_match:
            return json_block_match.group(1).strip()
            
        return text

class AgentResponse(BaseModel):
    """Standard Agent Response Schema"""
    thought: str
    type: str  # "answer", "tool_call", "error"
    content: Union[str, Dict[str, Any]]
