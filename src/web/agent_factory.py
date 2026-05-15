import asyncio
import logging
import os
from copy import deepcopy
from typing import Any

from src.core.agent.deep_research import DeepResearchAgent
from src.core.config.loader import load_llm_config
from src.core.config.server import SimpleMCPServerConfig
from src.core.llm.client import LLMManager
from src.core.llm.config import convert_to_client_config
from src.jarvis_agent import JarvisAgent, JarvisConfig

logger = logging.getLogger(__name__)

# Cache for LLM configuration
_llm_config_cache = {}


def get_llm_config(env: str = "production"):
    """
    Load and cache LLM configuration from YAML.
    """
    global _llm_config_cache

    if env in _llm_config_cache:
        return _llm_config_cache[env]

    try:
        yaml_config = load_llm_config(environment=env)
        client_config = convert_to_client_config(yaml_config)
        _llm_config_cache[env] = client_config
        logger.info(
            f"Loaded LLM Config: {client_config.provider_name} ({client_config.model})"
        )
        return client_config
    except Exception as e:
        logger.warning(f"Failed to load YAML configuration for env {env}: {e}")
        return None


def _apply_user_settings(
    client_config: Any, user_settings: dict[str, Any] | None
) -> Any:
    if client_config is None:
        return None

    effective_config = deepcopy(client_config)
    if not user_settings:
        return effective_config

    openai_api_key = user_settings.get("openai_api_key")
    if openai_api_key:
        effective_config.api_key = openai_api_key

    tavily_api_key = user_settings.get("tavily_api_key")
    if tavily_api_key:
        os.environ["TAVILY_API_KEY"] = tavily_api_key

    confluence_page_token = user_settings.get("confluence_page_token")
    if confluence_page_token:
        os.environ["CONFLUENCE_PAGE_TOKEN"] = confluence_page_token

    beacon_model_token = user_settings.get("beacon_model_token")
    if beacon_model_token:
        os.environ["BEACON_MODEL_TOKEN"] = beacon_model_token

    return effective_config


async def create_agent(
    agent_type: str = "jarvis",
    env: str = "production",
    user_settings: dict[str, Any] | None = None,
) -> Any:
    """
    Factory function to create and initialize an agent.

    Args:
        agent_type: "jarvis" or "deep_research"
        env: Environment name for config loading
        user_settings: User token settings

    Returns:
        Initialized agent instance
    """
    client_config = get_llm_config(env)
    effective_config = _apply_user_settings(client_config, user_settings)

    if agent_type == "deep_research":
        # Initialize Deep Research Agent
        agent_config = {}
        agent = DeepResearchAgent(agent_config)

        if effective_config:
            agent.llm_manager = LLMManager(effective_config)
            # Update tools with the new LLM manager
            if hasattr(agent, "tools") and hasattr(agent.tools, "llm_manager"):
                agent.tools.llm_manager = agent.llm_manager

        # Initialize agent
        if hasattr(agent, "initialize"):
            if asyncio.iscoroutinefunction(agent.initialize):
                await agent.initialize()
            else:
                agent.initialize()

        return agent

    else:
        # Initialize Jarvis Agent (Default)
        config = JarvisConfig()

        # Add default MCP servers
        config.mcp_servers = [
            SimpleMCPServerConfig(
                name="demo_server",
                command=["python", "-m", "demo_mcp_server"],
                description="Demonstration MCP server with basic tools",
            )
        ]

        agent = JarvisAgent(config)

        # Inject configuration into LLM Manager
        if effective_config:
            agent.ark_engine.llm_manager._default_config = effective_config

        success = await agent.initialize()

        if not success:
            logger.error("Failed to initialize Jarvis Agent.")
            return None

        await agent.start()
        return agent
