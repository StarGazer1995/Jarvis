"""
Live Cross-Review Demo (Multi-Agent Loop)

This script demonstrates a real-time multi-document cross-review agent with a Reviewer loop.
It uses Tavily for web research and implements a "Researcher -> Reviewer -> Researcher" feedback cycle.

Usage:
    export TAVILY_API_KEY="your-key"
    python examples/live_cross_review.py
"""

import asyncio
import logging
import os
import sys
from typing import List, Dict, Any

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.jarvis_agent import JarvisAgent, JarvisConfig
from src.core.config.loader import load_llm_config as load_yaml_config
from src.core.config.loader import LLMConfig as YamlLLMConfig, ProviderConfig, ModelConfig
from src.core.llm.types import LLMConfig as ClientLLMConfig, LLMProvider
from src.capabilities.web_research import WebResearcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
        timeout=provider_cfg.timeout.total if getattr(provider_cfg, 'timeout', None) else 300.0,
        provider_name=provider_name
    )

async def run_reviewer_check(agent: JarvisAgent, research_output: str) -> str:
    """
    Run the Reviewer agent to validate the research output.
    """
    reviewer_prompt = f"""
    You are a strict QA Reviewer. Your job is to validate the research report below.
    
    Research Report:
    {research_output}
    
    Check for the following:
    1. Are BOTH 'OpenAI o1' AND 'Claude 3.5 Sonnet' covered in detail?
    2. Are there specific pricing numbers for BOTH models?
    3. Are the sources listed and do they look relevant?
    
    If the report is complete and accurate, output exactly: "PASS"
    If information is missing or incorrect, output "RETRY: " followed by specific instructions on what is missing.
    
    Example Failure: "RETRY: Missing pricing for Claude 3.5 Sonnet. Please search specifically for 'Claude 3.5 Sonnet pricing'."
    """
    
    # We use the same agent but with a pure prompt (no tools needed for review ideally, but we keep it simple)
    # Ideally we would disable tools for the reviewer, but here we just prompt it.
    response = await agent.process_message(reviewer_prompt)
    return response

async def main():
    logger.info("Starting Live Cross-Review Demo (Multi-Agent)...")
    
    if not os.getenv("TAVILY_API_KEY"):
        logger.warning("TAVILY_API_KEY not found. Web search will fail.")
        logger.warning("Please set it via: export TAVILY_API_KEY='tvly-...'")

    # 1. Setup Agent
    try:
        env = os.getenv("JARVIS_ENV", "production")
        yaml_config = load_yaml_config(environment=env)
        client_config = convert_to_client_config(yaml_config)
        logger.info(f"Loaded config for: {client_config.provider_name}/{client_config.model}")
    except Exception as e:
        logger.error(f"Config load failed: {e}")
        return

    agent = JarvisAgent(JarvisConfig(name="ResearchTeam", log_level="INFO"))
    agent.ark_engine.llm_manager._default_config = client_config
    await agent.ark_engine.llm_manager.add_client("default", client_config)
    
    await agent.start()
    
    researcher = WebResearcher()
    agent.register_capabilities([researcher])
    
    # 2. Initial Research Task
    task_prompt = (
        "I need a cross-review of the latest AI models. "
        "Please search for the latest pricing and capabilities of 'OpenAI o1' and 'Claude 3.5 Sonnet'. "
        "Use the web_search tool to find information. "
        "Then, provide a comparison table highlighting their differences in pricing and key features. "
        "IMPORTANT: List all source URLs at the end."
    )
    
    print(f"\n[Phase 1] Initial Research...\nUser: {task_prompt}\n")
    print("-" * 50)
    
    research_output = await agent.process_message(task_prompt)
    print(f"\n[Phase 1] Output:\n{research_output}\n")
    print("-" * 50)
    
    # 3. Review Loop
    max_retries = 2
    for attempt in range(max_retries):
        print(f"\n[Phase 2] Reviewing (Attempt {attempt+1}/{max_retries})...")
        review_result = await run_reviewer_check(agent, research_output)
        print(f"Reviewer Verdict: {review_result}")
        
        if "PASS" in review_result:
            print("\n✅ Research Approved!")
            break
        elif "RETRY:" in review_result:
            print("\n❌ Research Rejected. Triggering Follow-up...")
            
            # Extract instructions
            instructions = review_result.split("RETRY:", 1)[1].strip()
            
            # 4. Follow-up Research
            follow_up_prompt = (
                f"The previous research was incomplete. Reviewer feedback: {instructions}\n"
                "Please perform additional searches to address these gaps specifically. "
                "Then, combine the new findings with the previous information to generate an UPDATED comparison table. "
                "Ensure ALL models (OpenAI o1 AND Claude 3.5 Sonnet) are covered."
            )
            
            print(f"\n[Phase 3] Follow-up Research...\nInstruction: {follow_up_prompt}\n")
            research_output = await agent.process_message(follow_up_prompt)
            print(f"\n[Phase 3] Updated Output:\n{research_output}\n")
            print("-" * 50)
        else:
            print("\n⚠️ Reviewer gave ambiguous response. Stopping loop.")
            break
            
    await agent.stop()

if __name__ == "__main__":
    asyncio.run(main())
