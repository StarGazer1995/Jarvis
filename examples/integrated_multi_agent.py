"""
Integrated Multi-Agent Supervisor Example

This script demonstrates a Supervisor-Worker architecture where:
1. A Supervisor Agent orchestrates the workflow.
2. Multiple specialized agents (Researcher, Writer, QualityAssurance) perform tasks.
3. Agents communicate via a shared state protocol.
4. One agent (QualityAssurance) encapsulates a complex Refinement Loop.
"""

import asyncio
import logging
import os
import sys
import json
from typing import Dict, Any

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.core.config.loader import load_llm_config as load_yaml_config
from src.core.config.loader import LLMConfig as YamlLLMConfig, ProviderConfig, ModelConfig
from src.core.llm.types import LLMConfig as ClientLLMConfig, LLMProvider, LLMMessage
from src.core.llm.client import LLMManager
from langchain_core.messages import HumanMessage
from src.core.ark.graph import create_supervisor_graph
from src.core.ark.utils import AgentSpec, create_agent_node
from src.core.ark.state import MultiAgentState
from src.capabilities.refinement.loop import RefinementLoop
from src.capabilities.web_research import WebResearcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("example.supervisor")

# --- Helper Config Loading (Copied from other examples) ---
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
        timeout=provider_cfg.timeout.total if getattr(provider_cfg, 'timeout', None) else 60.0,
        provider_name=provider_name
    )

async def main():
    logger.info("Starting Multi-Agent Supervisor Demo...")
    
    # 1. Load Config & Init LLM Manager
    try:
        env = os.getenv("JARVIS_ENV", "production")
        yaml_config = load_yaml_config(environment=env)
        client_config = convert_to_client_config(yaml_config)
        
        llm_manager = LLMManager(client_config)
        await llm_manager.initialize_default_client()
        # We also need a secondary LLMManager for sub-agents if we want to parallelize or use different configs
        # For simplicity, we share the same manager.
        
        logger.info(f"LLM Manager initialized with {client_config.provider_name}/{client_config.model}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    # 2. Define Agents
    
    # --- Helper: Find latest user instruction ---
    def get_user_task(messages):
        # Find the last message that is from the user (no name)
        for m in reversed(messages):
            if isinstance(m, HumanMessage) and not m.name:
                return m.content
        return messages[0].content if messages else ""

    # --- Agent A: Researcher ---
    # This agent performs web research.
    async def researcher_logic(state: MultiAgentState) -> Dict[str, Any]:
        task = get_user_task(state["messages"])
        
        logger.info(f"[Researcher] Received task: {task}")
        
        # Simulate Research (or use WebResearcher if API key present)
        results = []
        if os.getenv("TAVILY_API_KEY"):
            logger.info("[Researcher] Using Tavily for real search...")
            researcher = WebResearcher()
            # We call the search method directly
            # Note: WebResearcher methods are usually synchronous or async?
            # Looking at source, they are tool functions. Let's assume we can use them.
            # But WebResearcher.search is a bound method returning a string usually.
            # For this demo, let's just mock if complex.
            # Let's try to simple mock for reliability of the demo unless we are sure.
            search_result = f"Real search result for '{task}' would go here."
            sources = ["https://python.org", "https://docs.python.org/3.13/"]
            content = researcher.search(task, domains=sources)
        else:
            logger.info("[Researcher] No API key, using mock data.")
            content = f"MOCK RESEARCH for '{task}':\n- Feature 1: Free-threaded Python (No GIL)\n- Feature 2: JIT Compiler\n- Feature 3: Defined semantics for locals()"
            sources = ["mock://python-313-news"]
            
        return {
            "content": content,
            "data": {"sources": sources, "research_summary": content}
        }

    research_node = create_agent_node("Researcher", researcher_logic)

    # --- Agent B: Writer ---
    # This agent writes content based on research.
    async def writer_logic(state: MultiAgentState) -> Dict[str, Any]:
        task = get_user_task(state["messages"])
        
        # Check for shared data
        structured_data = state.get("structured_data", {})
        research_summary = structured_data.get("research_summary", "No research data found.")
        
        logger.info(f"[Writer] Received task: {task}")
        logger.info(f"[Writer] Context from shared data: {research_summary[:50]}...")
        
        # Simple LLM generation
        prompt = f"Using the following research:\n{research_summary}\n\nTask: {task}\n\nWrite a concise paragraph."
        response = await llm_manager.generate_response([LLMMessage(role="user", content=prompt)])
        
        return {
            "content": response.content,
            "data": {"draft": response.content}
        }
        
    writer_node = create_agent_node("Writer", writer_logic)

    # --- Agent C: QualityAssurance (Refinement Loop) ---
    # This agent refines the content using the RefinementLoop capability.
    async def qa_logic(state: MultiAgentState) -> Dict[str, Any]:
        task = get_user_task(state["messages"])
        
        # Ideally, we refine the *Draft* from the Writer.
        # But the Supervisor might just say "Review the last output".
        # Let's assume the task string contains instructions or we pick up the draft.
        structured_data = state.get("structured_data", {})
        draft = structured_data.get("draft", "")
        
        if not draft:
            # Fallback: Treat the task itself as the content to generate/refine
            initial_prompt = task
        else:
            initial_prompt = f"Refine this text for clarity and tone:\n\n{draft}"
            
        logger.info(f"[QA] Starting Refinement Loop for: {initial_prompt[:50]}...")
        
        # Define Generator/Reviewer for the loop
        async def generator(p: str) -> str:
            res = await llm_manager.generate_response([LLMMessage(role="user", content=p)])
            return res.content
            
        async def reviewer(c: str) -> str:
            # Simple self-check
            prompt = f"Review this text:\n{c}\n\nIs it professional and clear? Output PASS or RETRY: <reason>"
            res = await llm_manager.generate_response([LLMMessage(role="user", content=prompt)])
            return res.content

        loop = RefinementLoop(generator, reviewer, max_retries=1)
        final_result = await loop.run(initial_prompt)
        
        # Save the result
        saved_path = loop.save_result(final_result, directory="reports", prefix="multi_agent_summary", task=task)
        logger.info(f"[QA] Final report saved to: {saved_path}")
        
        return {
            "content": f"QA Approved Version (Saved to {saved_path}):\n{final_result}",
            "data": {"final_output": final_result, "report_path": saved_path}
        }
        
    qa_node = create_agent_node("QualityAssurance", qa_logic)

    # 3. Create Supervisor Graph
    agents = [
        AgentSpec(name="Researcher", description="Use this for finding information, data, or facts.", node=research_node),
        AgentSpec(name="Writer", description="Use this to write drafts, summaries, or content based on research.", node=writer_node),
        AgentSpec(name="QualityAssurance", description="Use this to review, refine, and polish text. ALWAYS use this before finishing.", node=qa_node)
    ]
    
    graph = create_supervisor_graph(llm_manager, agents)
    # graph.debug = True
    
    # 4. Run Workflow
    user_input = "Find out what's new in Python 3.13 and write a polished summary."
    print(f"\nUser: {user_input}")
    print("-" * 50)

    
    inputs = {
        "messages": [HumanMessage(content=user_input)],
        "structured_data": {}
    }
    
    async for output in graph.astream(inputs):
        for key, value in output.items():
            if key == "supervisor":
                print(f"👮 [Supervisor] Next -> {value['next']}")
            else:
                # Worker output
                # The state update contains messages
                msgs = value.get("messages", [])
                if msgs:
                    last_msg = msgs[-1]
                    print(f"👷 [{key}] {last_msg.content[:100]}...")
                    if "structured_data" in value and value["structured_data"]:
                         print(f"   (Shared Data Updated: {list(value['structured_data'].keys())})")
            print("...")

    print("-" * 50)
    print("Workflow Finished.")

if __name__ == "__main__":
    asyncio.run(main())
