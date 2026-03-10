"""
LLM Configuration Module

This module provides configuration management for LLM integration in the ARK engine.
"""

import logging
from typing import Dict, Any, Optional


from .types import LLMProvider, LLMConfig
# Import for type checking or lazy loading to avoid circular imports if any
# from ..config.loader import LLMConfig as YamlLLMConfig, ModelConfig

logger = logging.getLogger(__name__)


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

        # Mock provider removed
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
            "temperature": self._config.temperature,
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


def convert_to_client_config(yaml_config: Any) -> LLMConfig:
    """
    Convert the loaded YAML configuration into the Client LLMConfig format.

    Args:
        yaml_config: YamlLLMConfig object from src.core.config.loader

    Returns:
        LLMConfig: Client configuration object
    """
    # 1. Determine active provider
    # Default to global default, can be overridden by env
    provider_name = yaml_config.global_config.default_provider

    if provider_name not in yaml_config.providers:
        raise ValueError(f"Provider '{provider_name}' not found in configuration")

    provider_cfg = yaml_config.providers[provider_name]

    if not provider_cfg.enabled:
        raise ValueError(f"Provider '{provider_name}' is disabled in configuration")

    # 2. Determine model
    model_name = provider_cfg.default_model

    model_cfg = provider_cfg.models.get(model_name)
    if not model_cfg:
        # Fallback if specific model config not found, use defaults
        # Create a default ModelConfig-like object or use defaults
        from ..config.loader import ModelConfig

        model_cfg = ModelConfig()

    # 3. Construct Client LLMConfig
    # Map 'type' from provider config (e.g. 'openai') to LLMProvider enum
    try:
        provider_type = LLMProvider(provider_cfg.type)
    except ValueError:
        # Fallback or custom string
        provider_type = provider_cfg.type

    client_config = LLMConfig(
        provider=provider_type,
        model=model_name,
        api_key=provider_cfg.api_key,
        base_url=provider_cfg.base_url,
        temperature=model_cfg.temperature,
        max_tokens=model_cfg.max_tokens,
        timeout=provider_cfg.timeout.total if provider_cfg.timeout else 30.0,
        provider_name=provider_name,
    )

    return client_config
