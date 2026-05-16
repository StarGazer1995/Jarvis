"""
第一梯队覆盖补全 - llm/config, prompt/manager, nodes/tools, ark/utils
"""

import os
from unittest.mock import MagicMock

# ═══════════════════════════════════════════════════════════════
# src/core/llm/config.py
# ═══════════════════════════════════════════════════════════════


class TestLLMConfigTier1:
    """llm/config.py 全覆盖"""

    def test_is_configured_for_non_openai(self):
        from src.core.llm.config import LLMConfigManager

        mgr = LLMConfigManager()
        mgr.load_config({"provider": "ollama", "model": "llama3"})
        assert mgr.is_configured() is True

    def test_get_provider_info_with_config(self):
        from src.core.llm.config import LLMConfigManager

        mgr = LLMConfigManager()
        mgr.load_config(
            {"provider": "anthropic", "api_key": "sk-ant", "model": "claude-3"}
        )
        info = mgr.get_provider_info()
        assert info["configured"] is True
        assert info["model"] == "claude-3"

    def test_convert_to_client_config(self):
        import tempfile

        import yaml

        from src.core.config.loader import load_llm_config as load_yaml
        from src.core.llm.config import convert_to_client_config

        data = {
            "global": {"default_provider": "openai"},
            "providers": {
                "openai": {
                    "type": "openai",
                    "enabled": True,
                    "default_model": "gpt-4",
                    "models": {"gpt-4": {"max_tokens": 2048}},
                    "api_key": "sk-test",
                }
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            p = f.name
        try:
            yaml_cfg = load_yaml(config_path=p)
            client_cfg = convert_to_client_config(yaml_cfg)
            assert client_cfg is not None
            assert client_cfg.model == "gpt-4"
            assert client_cfg.max_tokens == 2048
        finally:
            os.unlink(p)


# ═══════════════════════════════════════════════════════════════
# src/core/prompt/manager.py
# ═══════════════════════════════════════════════════════════════


class TestPromptManagerTier1:
    """prompt/manager.py 全覆盖"""

    def test_add_and_get_template(self):
        from langchain_core.prompts import ChatPromptTemplate

        from src.core.prompt.manager import PromptManager

        pm = PromptManager()
        template = ChatPromptTemplate.from_template("Hello {{name}}!")
        pm.add_template("greet", template)
        retrieved = pm.get_template("greet")
        assert retrieved is template
        assert pm.get_template("nonexistent") is None

    def test_build_conversation_messages(self):
        from src.core.prompt.manager import PromptManager

        pm = PromptManager()
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        messages = pm.build_conversation_messages("test input", history, {})
        assert len(messages) >= 2

    def test_build_tool_usage_prompt(self):
        from src.core.prompt.manager import PromptManager

        pm = PromptManager()
        tools = [{"name": "search", "description": "Search web"}]
        result = pm.build_tool_usage_prompt("test query", "search", tools)
        assert "test query" in result


# ═══════════════════════════════════════════════════════════════
# src/core/ark/nodes/tools.py
# ═══════════════════════════════════════════════════════════════


class TestToolsNodeTier1:
    """ToolsNode 全覆盖"""

    def test_tools_node_creation_defaults(self):
        from src.core.ark.nodes.tools import ToolsNode

        mcp = MagicMock()
        node = ToolsNode(mcp_client=mcp)
        assert node.max_concurrency == 10
        assert node.mcp_client is mcp


# ═══════════════════════════════════════════════════════════════
# src/core/ark/utils.py
# ═══════════════════════════════════════════════════════════════


class TestArkUtilsTier1:
    """ark/utils.py 全覆盖"""

    def test_create_agent_node(self):
        from src.core.ark.state import MultiAgentState
        from src.core.ark.utils import create_agent_node

        async def simple_agent(state: MultiAgentState):
            return "Hello from agent"

        node_func = create_agent_node("test_agent", simple_agent)
        assert node_func is not None
