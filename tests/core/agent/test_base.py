"""
Base Agent 单元测试

测试 src/core/agent/base.py 的 BaseAgent 类。
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.core.agent.base import BaseAgent
from src.core.agent.types import AgentState


class ConcreteAgent(BaseAgent):
    """Concrete subclass for testing BaseAgent."""

    async def process_input(self, user_input, callbacks=None, **kwargs):
        return f"Processed: {user_input}"


class TestBaseAgent:
    """测试 BaseAgent 基础功能"""

    @pytest.mark.asyncio
    async def test_init_default(self):
        """测试默认初始化"""
        agent = ConcreteAgent()
        assert agent.state == AgentState.INITIALIZING
        assert agent.llm_enabled is True
        assert agent.available_tools == {}

    @pytest.mark.asyncio
    async def test_init_with_config(self):
        """测试带配置初始化"""
        agent = ConcreteAgent(config={"enable_llm": False, "llm": {"provider": "mock"}})
        assert agent.llm_enabled is False

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """测试初始化成功"""
        agent = ConcreteAgent(config={"enable_llm": False})
        result = await agent.initialize()
        assert result is True
        assert agent.state == AgentState.READY

    @pytest.mark.asyncio
    async def test_initialize_with_llm_enabled(self):
        """测试启用 LLM 时初始化"""
        agent = ConcreteAgent(config={"enable_llm": True})
        with patch.object(agent, "_initialize_llm", new=AsyncMock(return_value=None)):
            result = await agent.initialize()
            assert result is True
            assert agent.state == AgentState.READY

    @pytest.mark.asyncio
    async def test_initialize_failure(self):
        """测试初始化失败"""
        agent = ConcreteAgent(config={"enable_llm": True})
        with patch.object(
            agent,
            "_initialize_llm",
            new=AsyncMock(side_effect=Exception("Init failed")),
        ):
            result = await agent.initialize()
            assert result is False
            assert agent.state == AgentState.ERROR

    @pytest.mark.asyncio
    async def test_process_input_implemented(self):
        """测试 process_input 实现"""
        agent = ConcreteAgent()
        result = await agent.process_input("Hello")
        assert result == "Processed: Hello"

    @pytest.mark.asyncio
    async def test_initialize_llm_failure(self):
        """测试 LLM 初始化失败"""
        agent = ConcreteAgent(config={"enable_llm": True})
        with patch.object(
            agent.llm_manager,
            "initialize_default_client",
            new=AsyncMock(return_value=False),
        ):
            await agent._initialize_llm()
            # Should not raise, just log

    @pytest.mark.asyncio
    async def test_initialize_llm_success(self):
        """测试 LLM 初始化成功"""
        agent = ConcreteAgent(config={"enable_llm": True})
        with patch.object(
            agent.llm_manager,
            "initialize_default_client",
            new=AsyncMock(return_value=True),
        ):
            await agent._initialize_llm()
            # Should not raise
