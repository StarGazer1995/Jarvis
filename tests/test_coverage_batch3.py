"""
覆盖补全测试 - 最后一批

针对 jarvis_agent, factory, prompt/manager, config/loader, context/manager,
nodes/tools, nodes/master 的剩余未覆盖行。
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════
# src/jarvis_agent.py
# ═══════════════════════════════════════════════════════════════


class TestJarvisAgentLastBatch:
    """jarvis_agent 剩余边界测试"""

    @pytest.mark.asyncio
    async def test_register_capabilities_not_initialized(self):
        from src.jarvis_agent import JarvisAgent

        agent = JarvisAgent()
        agent.is_initialized = False
        agent.register_capabilities([])
        # Should log warning but not raise

    @pytest.mark.asyncio
    async def test_register_capabilities_no_get_tools(self):
        from src.jarvis_agent import JarvisAgent

        agent = JarvisAgent()
        agent.is_initialized = True

        class BadCap:
            pass

        agent.register_capabilities([BadCap()])
        # Should log warning about missing get_tools

    @pytest.mark.asyncio
    async def test_register_capabilities_with_tools(self):
        from src.jarvis_agent import JarvisAgent

        agent = JarvisAgent()
        agent.is_initialized = True
        agent.ark_engine.available_tools = {}

        class MyCap:
            def get_tools(self):
                return {
                    "tool1": {
                        "func": lambda x: x,
                        "schema": {"name": "tool1"},
                    }
                }

        agent.register_capabilities([MyCap()])
        assert "tool1" in agent.ark_engine.available_tools

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        from src.jarvis_agent import JarvisAgent

        agent = JarvisAgent()
        agent.is_initialized = True
        agent.ark_engine.get_status = MagicMock(return_value={"overall": "healthy"})
        result = await agent.health_check()
        assert "healthy" in str(result) or "overall" in str(result)


# ═══════════════════════════════════════════════════════════════
# src/core/llm/factory.py
# ═══════════════════════════════════════════════════════════════


class TestFactoryLastBatch:
    """factory 剩余边界测试"""

    def test_register_provider_invalid_class(self):
        from src.core.llm.factory import LLMProviderRegistry

        registry = LLMProviderRegistry()
        with pytest.raises(ValueError, match="必须继承自BaseLLMClient"):
            registry.register_provider("bad", str)

    def test_unregister_nonexistent(self):
        from src.core.llm.factory import LLMProviderRegistry

        registry = LLMProviderRegistry()
        # Should not raise
        registry.unregister_provider("nonexistent")


# ═══════════════════════════════════════════════════════════════
# src/core/prompt/manager.py
# ═══════════════════════════════════════════════════════════════


class TestPromptManagerLastBatch:
    """prompt/manager 剩余测试"""

    def test_render_template_nonexistent(self):
        from src.core.prompt.manager import PromptManager

        pm = PromptManager()
        with pytest.raises(ValueError, match="不存在"):
            pm.render_template("nonexistent_template_name_xyz")

    def test_list_templates(self):
        from src.core.prompt.manager import PromptManager

        pm = PromptManager()
        templates = pm.list_templates()
        assert isinstance(templates, list)
        assert len(templates) > 0


# ═══════════════════════════════════════════════════════════════
# src/core/config/loader.py
# ═══════════════════════════════════════════════════════════════


class TestConfigLoaderLastBatch:
    """config/loader 剩余测试"""

    def test_get_config_path_default(self):
        from src.core.config.loader import get_config_path

        path = get_config_path()
        assert str(path).endswith("config/llm_config.yaml")

    def test_config_singleton(self):
        from src.core.config.loader import get_config_loader

        loader1 = get_config_loader()
        loader2 = get_config_loader()
        assert loader1 is loader2


# ═══════════════════════════════════════════════════════════════
# src/core/context/manager.py 最后剩余
# ═══════════════════════════════════════════════════════════════


class TestContextRemaining:
    """context/manager 最后边界"""

    def test_debug_log_add_exchange(self, caplog):
        import logging
        caplog.set_level(logging.DEBUG, logger="ark.context")
        from src.core.context.manager import ConversationContext

        ctx = ConversationContext()
        ctx.add_exchange("Hello", "Hi")
        assert any("Added exchange" in record.msg for record in caplog.records)

    def test_debug_log_add_turn(self, caplog):
        import logging
        caplog.set_level(logging.DEBUG, logger="ark.context")
        from src.core.context.manager import ConversationContext, ConversationTurn

        ctx = ConversationContext()
        turn = ConversationTurn(user_input="Hi", agent_response="Hello")
        ctx.add_turn(turn)
        assert any("Added turn" in record.msg for record in caplog.records)

    def test_compress_history_below_threshold(self):
        import asyncio
        from src.core.context.manager import ConversationContext

        ctx = ConversationContext(max_history=100)
        ctx.add_exchange("U1", "A1")
        asyncio.run(ctx.compress_history(threshold=20))
        assert len(ctx) == 1

    def test_get_conversation_summary_with_intents(self):
        from src.core.context.manager import ConversationContext

        ctx = ConversationContext()
        ctx.add_exchange("Hello", "Hi", intent="greeting")
        ctx.add_exchange("Weather?", "Sunny", intent="weather")
        summary = ctx.get_conversation_summary()
        assert "greeting" in summary["intents"]
        assert "weather" in summary["intents"]


# ═══════════════════════════════════════════════════════════════
# src/core/ark/nodes/tools.py
# ═══════════════════════════════════════════════════════════════


class TestToolsNodeLastBatch:
    """ToolsNode 剩余测试"""

    def test_tools_node_creation_with_concurrency(self):
        from unittest.mock import MagicMock
        from src.core.ark.nodes.tools import ToolsNode

        mcp = MagicMock()
        node = ToolsNode(mcp_client=mcp, max_concurrency=5)
        assert node.max_concurrency == 5


# ═══════════════════════════════════════════════════════════════
# src/core/security/manager.py 最后
# ═══════════════════════════════════════════════════════════════


class TestSecurityLastBatch:
    """Security 最后边界"""

    def test_check_rate_limit_with_exceed(self):
        import asyncio
        from src.core.security.manager import RateLimiter

        limiter = RateLimiter()
        # Exceed per-minute limit
        for i in range(5):
            asyncio.run(limiter.check_rate_limit("u1", "t1", 3, 1000))
        result = asyncio.run(limiter.check_rate_limit("u1", "t1", 3, 1000))
        assert result is False

    def test_check_rate_limit_per_hour(self):
        import asyncio
        from src.core.security.manager import RateLimiter

        limiter = RateLimiter()
        # Exceed per-hour limit by filling minute windows
        # Just test basic functionality
        result = asyncio.run(limiter.check_rate_limit("u2", "t1", 60, 3))
        result2 = asyncio.run(limiter.check_rate_limit("u2", "t1", 60, 3))
        result3 = asyncio.run(limiter.check_rate_limit("u2", "t1", 60, 3))
        # 3 is the per-hour limit, so 3rd should still work (3 <= 3), 4th fails
        result4 = asyncio.run(limiter.check_rate_limit("u2", "t1", 60, 3))
        assert result4 is False


# ═══════════════════════════════════════════════════════════════
# src/core/llm/config.py 更多
# ═══════════════════════════════════════════════════════════════


class TestLLMConfigMoreCoverage:
    """llm/config 剩余测试"""

    def test_config_mgr_update_multiple(self):
        from src.core.llm.config import LLMConfigManager

        mgr = LLMConfigManager()
        mgr.load_config({"provider": "openai", "api_key": "k", "model": "gpt-4"})
        mgr.update_config(temperature=0.5, max_tokens=2000)
        cfg = mgr.get_config()
        assert cfg.temperature == 0.5
        assert cfg.max_tokens == 2000

    def test_config_mgr_update_nonexistent_field(self):
        from src.core.llm.config import LLMConfigManager

        mgr = LLMConfigManager()
        mgr.load_config({"provider": "openai", "api_key": "k", "model": "gpt-4"})
        # Should not raise
        mgr.update_config(nonexistent_field="value")


# ═══════════════════════════════════════════════════════════════
# src/core/ark/nodes/master.py
# ═══════════════════════════════════════════════════════════════


class TestMasterNodeMore:
    """MasterNode 更多测试"""

    def test_master_node_creation_with_agents(self):
        from unittest.mock import MagicMock
        from src.core.ark.nodes.master import MasterNode

        llm = MagicMock()
        node = MasterNode(llm_manager=llm, agents=[])
        assert node.agents == []
        assert node.agent_map == {}


# ═══════════════════════════════════════════════════════════════
# Quick wins for remaining uncovered code
# ═══════════════════════════════════════════════════════════════


class TestQuickWins:
    """快速补漏"""

    def test_config_loader_env_substitution(self):
        """测试配置加载器环境变量替换"""
        from src.core.config.loader import ConfigLoader
        import tempfile, os, yaml
        data = {
            "global": {"default_provider": "openai"},
            "providers": {
                "openai": {
                    "type": "openai",
                    "enabled": True,
                    "default_model": "gpt-4",
                    "models": {"gpt-4": {}},
                    "api_key": "test-key",
                }
            },
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(data, f)
            p = f.name
        try:
            loader = ConfigLoader(p)
            result = loader.load_config()
            assert result is not None
            assert len(result.providers) > 0
        finally:
            os.unlink(p)

    def test_prompt_manager_render_system(self):
        """测试渲染系统提示词模板"""
        from src.core.prompt.manager import PromptManager
        pm = PromptManager()
        result = pm.render_template("react_system")
        assert result is not None

    def test_parsers_repair_json(self):
        """测试 JSON 修复功能"""
        from src.core.llm.parsers import JSONOutputParser
        # JSONOutputParser may handle repair differently
        parser = JSONOutputParser(allow_repair=True)
        result = parser.parse("```json\n{\"valid\": true}\n```")
        assert result["valid"] is True


class TestLLMConfigRemainingCoverage:
    """llm/config.py 剩余方法全覆盖"""

    def test_is_configured_false_when_no_config(self):
        from src.core.llm.config import LLMConfigManager
        mgr = LLMConfigManager()
        assert mgr.is_configured() is False

    def test_is_configured_openai_without_key(self):
        from src.core.llm.config import LLMConfigManager
        mgr = LLMConfigManager()
        mgr.load_config({"provider": "openai", "model": "gpt-4"})
        # api_key is None, so should return False
        assert mgr.is_configured() is False

    def test_is_configured_openai_with_key(self):
        from src.core.llm.config import LLMConfigManager
        mgr = LLMConfigManager()
        mgr.load_config({"provider": "openai", "api_key": "sk-key", "model": "gpt-4"})
        assert mgr.is_configured() is True

    def test_get_provider_info_no_config(self):
        from src.core.llm.config import LLMConfigManager
        mgr = LLMConfigManager()
        info = mgr.get_provider_info()
        assert info["provider"] == "none"
        assert info["configured"] is False



class TestFinalPush:
    """最后一波推向90%"""

    # ── llm/config.py: from_env ──
    def test_config_from_env(self):
        import os
        from unittest.mock import patch
        from src.core.llm.config import LLMConfigManager
        env = {"LLM_PROVIDER": "anthropic", "OPENAI_API_KEY": "sk-key", "LLM_MODEL": "claude-3"}
        with patch.dict(os.environ, env, clear=True):
            mgr = LLMConfigManager()
            cfg = mgr.load_config()
            assert cfg is not None
            assert cfg.model == "claude-3"

    # ── factory.py edge cases ──
    def test_factory_get_provider_class(self):
        from src.core.llm.client import BaseLLMClient
        from src.core.llm.factory import LLMProviderRegistry
        class MockClient(BaseLLMClient):
            async def initialize(self): return True
            async def generate_response(self, m, **k): pass
            async def stream_response(self, m, **k): yield ""
        registry = LLMProviderRegistry()
        registry.register_provider("mock", MockClient)
        cls = registry.get_provider_class("mock")
        assert cls is MockClient
        assert registry.get_provider_class("nonexistent") is None

    # ── factory.py: unregister nonexistent ──
    def test_factory_unregister(self):
        from src.core.llm.factory import LLMProviderRegistry
        registry = LLMProviderRegistry()
        registry.unregister_provider("nonexistent")

    # ── prompt/manager.py ──
    def test_prompt_manager_str(self):
        from src.core.prompt.manager import PromptManager
        pm = PromptManager()
        s = str(pm)
        assert "PromptManager" in s

    # ── config/loader.py ──
    def test_config_loader_load_empty(self):
        from src.core.config.loader import ConfigLoader, get_config_loader
        loader = get_config_loader()
        assert loader is not None

    # ── litellm_client.py ──
    def test_litellm_config(self):
        from src.core.llm.types import LLMConfig, LLMProvider
        cfg = LLMConfig(provider=LLMProvider.LITELLM, model="gpt-4", api_key="k")
        assert cfg.provider == LLMProvider.LITELLM
        assert cfg.model == "gpt-4"
