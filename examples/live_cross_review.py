"""
Live Cross-Review Demo (Generic Refinement Loop)

This script demonstrates a real-time multi-document cross-review agent using the
generic RefinementLoop capability.

Usage:
    export TAVILY_API_KEY="your-key"
    python examples/live_cross_review.py
"""

import asyncio
import logging
import os
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.jarvis_agent import JarvisAgent, JarvisConfig
from src.core.config.loader import load_llm_config as load_yaml_config
from src.core.config.loader import (
    LLMConfig as YamlLLMConfig,
    ProviderConfig,
    ModelConfig,
)
from src.core.llm.types import LLMConfig as ClientLLMConfig, LLMProvider
from src.capabilities.web_research import WebResearcher
from src.capabilities.refinement import RefinementLoop

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("example.cross_review")


def convert_to_client_config(yaml_config: YamlLLMConfig) -> ClientLLMConfig:
    """
    Convert the loaded YAML configuration into the Client LLMConfig format.
    """
    provider_name = yaml_config.global_config.default_provider
    if provider_name not in yaml_config.providers:
        raise ValueError(f"Provider '{provider_name}' not found")

    provider_cfg = yaml_config.providers[provider_name]
    if not provider_cfg.enabled:
        raise ValueError(f"Provider '{provider_name}' is disabled")

    model_name = provider_cfg.default_model
    model_cfg = provider_cfg.models.get(model_name) or ModelConfig()

    try:
        provider_type = LLMProvider(provider_cfg.type)
    except ValueError:
        provider_type = provider_cfg.type

    return ClientLLMConfig(
        provider=provider_type,
        model=model_name,
        api_key=provider_cfg.api_key,
        base_url=provider_cfg.base_url,
        temperature=model_cfg.temperature,
        max_tokens=model_cfg.max_tokens,
        timeout=provider_cfg.timeout.total
        if getattr(provider_cfg, "timeout", None)
        else 600.0,
        provider_name=provider_name,
    )


async def main():
    logger.info("Starting Live Cross-Review Demo (Generic Loop)...")

    if not os.getenv("TAVILY_API_KEY"):
        logger.warning("TAVILY_API_KEY not found. Web search will fail.")

    # 1. Setup Agent
    try:
        env = os.getenv("JARVIS_ENV", "production")
        yaml_config = load_yaml_config(environment=env)
        client_config = convert_to_client_config(yaml_config)
        logger.info(
            f"Loaded config for: {client_config.provider_name}/{client_config.model}"
        )
    except Exception as e:
        logger.error(f"Config load failed: {e}")
        return

    agent = JarvisAgent(JarvisConfig(name="ResearchTeam", log_level="INFO"))
    agent.ark_engine.llm_manager._default_config = client_config
    await agent.ark_engine.llm_manager.add_client("default", client_config)

    await agent.start()

    researcher = WebResearcher()
    agent.register_capabilities([researcher])

    # 2. Define Generator and Reviewer Functions

    async def generator_func(prompt: str) -> str:
        """The generator performs the research/writing task."""
        return await agent.process_message(prompt)

    async def reviewer_func(content: str) -> str:
        """The reviewer validates the content."""
        reviewer_prompt = f"""
        You are a strict QA Reviewer. Validate the research report below.
        
        Report:
        {content}
        
        Check for:
        1. Coverage of BOTH 'OpenAI o1' AND 'Claude 3.5 Sonnet'.
        2. Specific pricing numbers for BOTH.
        3. Relevant source URLs listed.
        
        Output "PASS" if good.
        Output "RETRY: <instructions>" if missing info.
        """
        # We assume the same agent can role-play the reviewer
        # In a real system, you might use a different agent or model
        return await agent.process_message(reviewer_prompt)

    # 3. Execute via RefinementLoop Capability
    loop = RefinementLoop(
        generator=generator_func, reviewer=reviewer_func, max_retries=2
    )

    task_prompt = (
        "I need a cross-review of the latest AI models. "
        "Search for pricing and capabilities of 'OpenAI o1' and 'Claude 3.5 Sonnet'. "
        "PRIORITY: Use the 'web_search_knowledge' tool first to find high-quality information from Zhihu, Arxiv, and Scholar. "
        "Only use standard web_search if knowledge sources are insufficient. "
        "Provide a comparison table with differences. "
        "IMPORTANT: List all source URLs at the end."
    )

    print(f"\nUser: {task_prompt}\n")
    print("-" * 50)

    # Run the loop
    final_result = await loop.run(task_prompt)

    # Save the result
    saved_path = loop.save_result(
        final_result, directory="reports", prefix="cross_review", task=task_prompt
    )

    print("-" * 50)
    print(f"\n✨ Final Report Saved Successfully!")
    print(f"Path: {saved_path}")
    print("-" * 50)

    await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
