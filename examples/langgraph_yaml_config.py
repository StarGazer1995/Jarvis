"""
LangGraph ARK Engine - YAML Config Example

This script demonstrates how to load LLM configuration from `config/llm_config.yaml`
and use it to initialize the ARK Engine.

Usage:
    # Ensure you have config/llm_config.yaml with valid API keys
    # (or set env vars referenced in the yaml like NVIDIA_API_KEY)
    
    export NVIDIA_API_KEY=nvapi-... # If using nvidia_nim default
    python examples/langgraph_yaml_config.py
"""

import asyncio
import logging
import os
import sys
from typing import Any, Dict

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.core.ark.engine import ARKEngine
from src.core.config.loader import load_llm_config as load_yaml_config
from src.core.config.loader import LLMConfig as YamlLLMConfig, ProviderConfig, ModelConfig
from src.core.llm.types import LLMConfig as ClientLLMConfig, LLMProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("example.yaml_config")

def convert_to_client_config(yaml_config: YamlLLMConfig) -> ClientLLMConfig:
    """
    Convert the loaded YAML configuration into the Client LLMConfig format.
    """
    # 1. Determine active provider
    # Default to global default, can be overridden by env
    provider_name = yaml_config.global_config.default_provider
    logger.info(f"Selected Provider: {provider_name}")
    
    if provider_name not in yaml_config.providers:
        raise ValueError(f"Provider '{provider_name}' not found in configuration")
        
    provider_cfg: ProviderConfig = yaml_config.providers[provider_name]
    
    if not provider_cfg.enabled:
        raise ValueError(f"Provider '{provider_name}' is disabled in configuration")

    # 2. Determine model
    model_name = provider_cfg.default_model
    logger.info(f"Selected Model: {model_name}")
    
    model_cfg: ModelConfig = provider_cfg.models.get(model_name)
    if not model_cfg:
        # Fallback if specific model config not found, use defaults
        logger.warning(f"Model config for '{model_name}' not found, using defaults")
        model_cfg = ModelConfig()

    # 3. Construct Client LLMConfig
    # Map 'type' from provider config (e.g. 'openai') to LLMProvider enum
    try:
        provider_type = LLMProvider(provider_cfg.type)
    except ValueError:
        # Fallback or custom string
        provider_type = provider_cfg.type
        
    client_config = ClientLLMConfig(
        provider=provider_type,
        model=model_name,
        api_key=provider_cfg.api_key,
        base_url=provider_cfg.base_url,
        temperature=model_cfg.temperature,
        max_tokens=model_cfg.max_tokens,
        timeout=provider_cfg.timeout.total if provider_cfg.timeout else 30.0,
        retry_attempts=provider_cfg.retry.max_attempts if provider_cfg.retry else 3,
        stream=yaml_config.features.streaming_enabled,
        extra_params={
            "top_p": model_cfg.top_p,
            "frequency_penalty": model_cfg.frequency_penalty,
            "presence_penalty": model_cfg.presence_penalty
        },
        provider_name=provider_name  # Pass original provider name
    )
    
    return client_config

async def main():
    logger.info("Loading configuration from config/llm_config.yaml...")
    
    try:
        # Load the YAML configuration
        # Determine environment from JARVIS_ENV or default to 'production' for this real-world demo
        env = os.getenv("JARVIS_ENV", "production")
        logger.info(f"Target Environment: {env}")
        
        yaml_config = load_yaml_config(environment=env)
        
        # Convert to the format LLMManager expects
        client_config = convert_to_client_config(yaml_config)
        
        logger.info("-" * 30)
        logger.info(f"Initialized Config:")
        logger.info(f"  Provider: {client_config.provider}")
        logger.info(f"  Model:    {client_config.model}")
        logger.info(f"  API Base: {client_config.base_url}")
        logger.info("-" * 30)
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    # Initialize Engine
    engine = ARKEngine()
    
    # Configure the LLM Manager
    await engine.llm_manager.add_client("default", client_config)
    
    # Initialize Engine (Tools, Graph)
    if not await engine.initialize():
        logger.error("Engine initialization failed")
        return

    # Run Interaction
    user_input = "Who are you and what model are you using?"
    print(f"\nUser: {user_input}")
    print("-" * 50)
    
    response = await engine.process_input(user_input)
    
    print("-" * 50)
    print(f"Agent: {response}")

    await engine.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
