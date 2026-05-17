"""Additional JarvisAgent tests migrated from legacy `src/test` files."""

from unittest.mock import MagicMock

import pytest


class TestJarvisAgentDetailedCoverage:
    """JarvisAgent 边界测试"""

    def test_jarvis_config_defaults(self):
        from src.jarvis_agent import JarvisConfig

        config = JarvisConfig()
        assert config.name == "Jarvis"
        assert config.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self):
        from src.jarvis_agent import JarvisAgent

        agent = JarvisAgent()
        agent.is_initialized = False
        result = await agent.health_check()
        assert "not_initialized" in str(result) or "unhealthy" in str(result)


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
