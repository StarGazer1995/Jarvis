"""
Dynamic Multi-Agent Supervisor Example

This script demonstrates adding agents dynamically to a running Supervisor system.
It uses the new `create_dynamic_supervisor_graph` and `DynamicAgentRegistry`.
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.core.config.loader import load_llm_config as load_yaml_config
from src.core.config.loader import LLMConfig as YamlLLMConfig, ModelConfig
from src.core.llm.types import LLMConfig as ClientLLMConfig, LLMProvider, LLMMessage
from src.core.llm.client import LLMManager
from langchain_core.messages import HumanMessage
from src.core.ark.supervisor import (
    create_dynamic_supervisor_graph,
    AgentSpec,
    create_agent_node,
    MultiAgentState,
    DynamicAgentRegistry,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("example.dynamic_supervisor")


# --- Helper Config Loading ---
def convert_to_client_config(yaml_config: YamlLLMConfig) -> ClientLLMConfig:
    provider_name = yaml_config.global_config.default_provider
    provider_cfg = yaml_config.providers[provider_name]
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
        else 60.0,
        provider_name=provider_name,
    )


async def main():
    logger.info("Starting Dynamic Supervisor Demo...")

    # 1. Load Config & Init LLM Manager
    try:
        env = os.getenv("JARVIS_ENV", "production")
        yaml_config = load_yaml_config(environment=env)
        client_config = convert_to_client_config(yaml_config)
        llm_manager = LLMManager(client_config)
        await llm_manager.initialize_default_client()
        logger.info(
            f"LLM Manager initialized with {client_config.provider_name}/{client_config.model}"
        )
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    # 2. Define Agents Logic

    def get_user_task(messages):
        for m in reversed(messages):
            if isinstance(m, HumanMessage) and not m.name:
                return m.content
        return messages[0].content if messages else ""

    async def researcher_logic(state: MultiAgentState) -> Dict[str, Any]:
        task = get_user_task(state["messages"])
        logger.info(f"[Researcher] Researching: {task}")

        # Use LLM to simulate research results
        prompt = f"Please research and provide key information about: {task}. Focus on new features and technical details. Keep it concise."
        response = await llm_manager.generate_response(
            [LLMMessage(role="user", content=prompt)]
        )
        content = response.content

        return {"content": content, "data": {"research_summary": content}}

    async def writer_logic(state: MultiAgentState) -> Dict[str, Any]:
        task = get_user_task(state["messages"])
        logger.info("[Writer] Writing draft based on research...")

        # Get context from history (all previous messages)
        context = "\n".join([f"{m.type}: {m.content}" for m in state["messages"]])

        prompt = (
            f"Based on the following conversation history and research:\n{context}\n\n"
            f"Write a concise, polished summary/draft for the task: {task}\n"
            "Focus on clarity and structure."
        )
        response = await llm_manager.generate_response(
            [LLMMessage(role="user", content=prompt)]
        )

        return {"content": response.content, "data": {"draft": response.content}}

    async def reviewer_logic(state: MultiAgentState) -> Dict[str, Any]:
        logger.info("[Reviewer] Reviewing draft...")

        # Get the last message which should be the draft from Writer
        last_msg = state["messages"][-1]
        draft = last_msg.content

        prompt = (
            f"Review the following draft for quality, accuracy, and tone:\n\n{draft}\n\n"
            "If it's good, start with 'APPROVED'. If improvements are needed, start with 'REVISION NEEDED' and list suggestions."
        )
        response = await llm_manager.generate_response(
            [LLMMessage(role="user", content=prompt)]
        )

        return {"content": response.content, "data": {"review": response.content}}

    # Create Nodes
    research_node = create_agent_node("Researcher", researcher_logic)
    writer_node = create_agent_node("Writer", writer_logic)
    reviewer_node = create_agent_node("Reviewer", reviewer_logic)

    # 3. Setup Dynamic Registry
    registry = DynamicAgentRegistry()

    # Register ONLY Researcher and Writer initially
    registry.register_agent(
        AgentSpec(
            name="Researcher", description="Finds information.", node=research_node
        )
    )
    registry.register_agent(
        AgentSpec(
            name="Writer",
            description="Writes content based on research.",
            node=writer_node,
        )
    )

    logger.info("Initial Agents: Researcher, Writer")

    # 4. Create Dynamic Graph
    graph = create_dynamic_supervisor_graph(llm_manager, registry)

    # 5. Run Workflow 1 (Should use Researcher -> Writer)
    logger.info("\n--- Run 1: Basic Workflow ---")
    user_input = "Research Python 3.13 and write a summary."
    inputs = {"messages": [HumanMessage(content=user_input)], "structured_data": {}}

    # We use a limit to prevent infinite loops if LLM keeps going
    step_count = 0
    async for output in graph.astream(inputs):
        for key, value in output.items():
            if key == "supervisor":
                logger.info(f"👮 [Supervisor] Next -> {value['next']}")

                # SIMULATE DYNAMIC ADDITION:
                # If the supervisor decides 'Writer', let's add the 'Reviewer' agent dynamically!
                if value["next"] == "Writer" and step_count == 0:
                    logger.info(">>> DYNAMICALLY ADDING 'Reviewer' AGENT! <<<")
                    registry.register_agent(
                        AgentSpec(
                            name="Reviewer",
                            description="Reviews the written content for quality. Use AFTER Writer.",
                            node=reviewer_node,
                        )
                    )
                    # Note: Since the Supervisor has already decided 'Writer', the NEXT time it runs
                    # (after Writer finishes), it will see 'Reviewer' in the list.
                    step_count += 1
            elif key == "universal_worker":
                msgs = value.get("messages", [])
                if msgs:
                    logger.info(f"👷 [Worker] {msgs[-1].content}")

    logger.info("Run 1 Finished.")


if __name__ == "__main__":
    asyncio.run(main())
