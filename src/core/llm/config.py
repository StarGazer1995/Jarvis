"""
LLM Configuration Module

This module provides configuration management for LLM integration in the ARK engine.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


from .types import LLMProvider, LLMConfig


class LLMConfigManager:
    """Manager for LLM configuration."""
    
    def __init__(self):
        """Initialize LLM configuration manager."""
        self._config: Optional[LLMConfig] = None
    
    def load_config(self, config_source: Optional[Dict[str, Any]] = None) -> LLMConfig:
        """
        Load LLM configuration from various sources.
        
        Args:
            config_source: Optional configuration dictionary
            
        Returns:
            LLMConfig instance
        """
        if config_source:
            self._config = LLMConfig.from_dict(config_source)
        else:
            # Try to load from environment variables
            self._config = LLMConfig.from_env()
        
        return self._config
    
    def get_config(self) -> Optional[LLMConfig]:
        """Get current LLM configuration."""
        return self._config
    
    def update_config(self, **kwargs) -> None:
        """Update LLM configuration."""
        if self._config is None:
            self._config = LLMConfig.from_env()
        
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
    
    def is_configured(self) -> bool:
        """Check if LLM is properly configured."""
        if self._config is None:
            return False
        
        # For OpenAI provider, API key is required
        if self._config.provider == LLMProvider.OPENAI:
            return self._config.api_key is not None
        
        # Mock provider doesn't require API key
        return True
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider."""
        if self._config is None:
            return {"provider": "none", "configured": False}
        
        return {
            "provider": self._config.provider.value,
            "model": self._config.model,
            "configured": self.is_configured(),
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature
        }


# Global configuration manager instance
llm_config_manager = LLMConfigManager()


def get_llm_config() -> Optional[LLMConfig]:
    """Get global LLM configuration."""
    return llm_config_manager.get_config()


def load_llm_config(config_source: Optional[Dict[str, Any]] = None) -> LLMConfig:
    """Load global LLM configuration."""
    return llm_config_manager.load_config(config_source)


def is_llm_configured() -> bool:
    """Check if LLM is globally configured."""
    return llm_config_manager.is_configured()