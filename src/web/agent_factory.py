import os
import logging
import asyncio
from typing import Optional, Dict, Any

from src.jarvis_agent import JarvisAgent, JarvisConfig
from src.core.agent.deep_research import DeepResearchAgent
from src.core.config.server import SimpleMCPServerConfig
from src.core.config.loader import load_llm_config
from src.core.llm.config import convert_to_client_config
from src.core.llm.client import LLMManager

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


async def create_agent(agent_type: str = "jarvis", env: str = "production") -> Any:
    """
    Factory function to create and initialize an agent.

    Args:
        agent_type: "jarvis" or "deep_research"
        env: Environment name for config loading

    Returns:
        Initialized agent instance
    """
    client_config = get_llm_config(env)

    if agent_type == "deep_research":
        # Initialize Deep Research Agent
        agent_config = {}
        agent = DeepResearchAgent(agent_config)

        if client_config:
            agent.llm_manager = LLMManager(client_config)
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
        if client_config:
            agent.ark_engine.llm_manager._default_config = client_config

        success = await agent.initialize()

        if not success:
            logger.error("Failed to initialize Jarvis Agent.")
            return None

        await agent.start()
        return agent
