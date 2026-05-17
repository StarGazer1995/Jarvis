"""Additional ReAct agent tests migrated from `src/test`."""

from unittest.mock import patch

import pytest


class TestReactAgentCoverage:
    """覆盖 react.py 中未覆盖的边界"""

    @pytest.mark.asyncio
    async def test_process_input_not_ready(self):
        """测试 Agent 未就绪时的处理"""
        from src.core.agent.react import ReActAgent
        from src.core.agent.types import AgentState

        agent = ReActAgent()
        agent.state = AgentState.INITIALIZING
        response = await agent.process_input("test")
        assert "not ready" in response.lower()

    @pytest.mark.asyncio
    async def test_process_input_error(self):
        """测试 process_input 异常处理"""
        from src.core.agent.react import ReActAgent
        from src.core.agent.types import AgentState

        agent = ReActAgent()
        agent.state = AgentState.READY
        with patch.object(agent, "_run_loop", side_effect=ValueError("test error")):
            response = await agent.process_input("test")
            assert "error" in response.lower()
