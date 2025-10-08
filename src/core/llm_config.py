"""
LLM Configuration Module

This module provides configuration management for LLM integration in the ARK engine.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    MOCK = "mock"


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    provider: LLMProvider
    api_key: Optional[str] = None
    model: str = "gpt-3.5-turbo"
    base_url: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: int = 30
    retry_attempts: int = 3
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LLMConfig':
        """Create LLMConfig from dictionary."""
        provider_str = config_dict.get('provider', 'mock')
        provider = LLMProvider(provider_str)
        
        return cls(
            provider=provider,
            api_key=config_dict.get('api_key'),
            model=config_dict.get('model', 'gpt-3.5-turbo'),
            base_url=config_dict.get('base_url'),
            max_tokens=config_dict.get('max_tokens', 1000),
            temperature=config_dict.get('temperature', 0.7),
            timeout=config_dict.get('timeout', 30),
            retry_attempts=config_dict.get('retry_attempts', 3)
        )
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Create LLMConfig from environment variables."""
        provider_str = os.getenv('LLM_PROVIDER', 'mock')
        provider = LLMProvider(provider_str)
        
        return cls(
            provider=provider,
            api_key=os.getenv('OPENAI_API_KEY'),
            model=os.getenv('LLM_MODEL', 'gpt-3.5-turbo'),
            base_url=os.getenv('LLM_BASE_URL'),
            max_tokens=int(os.getenv('LLM_MAX_TOKENS', '1000')),
            temperature=float(os.getenv('LLM_TEMPERATURE', '0.7')),
            timeout=int(os.getenv('LLM_TIMEOUT', '30')),
            retry_attempts=int(os.getenv('LLM_RETRY_ATTEMPTS', '3'))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert LLMConfig to dictionary."""
        return {
            'provider': self.provider.value,
            'api_key': self.api_key,
            'model': self.model,
            'base_url': self.base_url,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'timeout': self.timeout,
            'retry_attempts': self.retry_attempts
        }


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