"""
ARK Utils 单元测试

测试 src/core/ark/utils.py 中的工具函数。
"""

from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.core.ark.state import MultiAgentState
from src.core.ark.utils import AgentSpec, create_agent_node


class TestAgentSpec:
    """测试 AgentSpec 数据类"""

    def test_agent_spec_creation(self):
        spec = AgentSpec(name="test", description="A test agent", node=Mock())
        assert spec.name == "test"
        assert spec.description == "A test agent"
        assert spec.node is not None


class TestCreateAgentNode:
    """测试 create_agent_node 函数"""

    @pytest.mark.asyncio
    async def test_sync_str_return(self):
        """测试同步函数返回字符串"""

        def sync_str(state: MultiAgentState) -> str:
            return "hello world"

        state = MultiAgentState(messages=[])
        node = create_agent_node("agent_a", sync_str)
        result = await node(state)
        assert "messages" in result
        assert result["sender"] == "agent_a"
        assert result["messages"][0].content == "hello world"

    @pytest.mark.asyncio
    async def test_sync_dict_with_messages_and_content(self):
        """测试同步函数返回包含 messages 和 content 的 dict"""

        def sync_dict(state: MultiAgentState) -> dict:
            return {"messages": ["msg1"], "content": "hello", "data": {"key": "val"}}

        state = MultiAgentState(messages=[])
        node = create_agent_node("agent_b", sync_dict)
        result = await node(state)
        assert result["sender"] == "agent_b"
        assert result["messages"][0].content == "hello"
        assert result["messages"][0].name == "agent_b"

    @pytest.mark.asyncio
    async def test_sync_dict_with_messages_no_content(self):
        """测试同步函数返回包含 messages 但不含 content 的 dict（完整状态更新）"""

        def sync_full_state(state: MultiAgentState) -> dict:
            return {"messages": [HumanMessage(content="done")], "sender": "worker"}

        state = MultiAgentState(messages=[])
        node = create_agent_node("agent_c", sync_full_state)
        result = await node(state)
        assert result["sender"] == "agent_c"  # sender 会被覆盖为 agent_name
        assert result["messages"][0].content == "done"

    @pytest.mark.asyncio
    async def test_sync_dict_without_messages_with_content(self):
        """测试同步函数返回不含 messages 但有 content 的 dict"""

        def sync_no_messages(state: MultiAgentState) -> dict:
            return {"content": "some content", "data": {"info": "test"}}

        state = MultiAgentState(messages=[])
        node = create_agent_node("agent_d", sync_no_messages)
        result = await node(state)
        assert result["sender"] == "agent_d"
        assert result["messages"][0].content == "some content"
        assert result["messages"][0].name == "agent_d"

    @pytest.mark.asyncio
    async def test_sync_dict_plain(self):
        """测试同步函数返回普通 dict（无 messages, content, data）"""

        def sync_plain(state: MultiAgentState) -> dict:
            return {"foo": "bar"}

        state = MultiAgentState(messages=[])
        node = create_agent_node("agent_e", sync_plain)
        result = await node(state)
        assert result["sender"] == "agent_e"
        assert result["messages"][0].content == "{'foo': 'bar'}"

    @pytest.mark.asyncio
    async def test_async_func(self):
        """测试异步函数"""

        async def async_func(state: MultiAgentState) -> str:
            return "async result"

        state = MultiAgentState(messages=[])
        node = create_agent_node("agent_f", async_func)
        result = await node(state)
        assert result["messages"][0].content == "async result"

    @pytest.mark.asyncio
    async def test_base_message_return(self):
        """测试返回 BaseMessage"""

        def msg_func(state: MultiAgentState) -> BaseMessage:
            return AIMessage(content="msg response")

        state = MultiAgentState(messages=[])
        node = create_agent_node("agent_g", msg_func)
        result = await node(state)
        assert result["sender"] == "agent_g"
        assert result["messages"][0].content == "msg response"
        assert result["messages"][0].name == "agent_g"

    @pytest.mark.asyncio
    async def test_other_type_return(self):
        """测试返回其他类型（int）"""

        def int_func(state: MultiAgentState) -> int:
            return 42

        state = MultiAgentState(messages=[])
        node = create_agent_node("agent_h", int_func)
        result = await node(state)
        assert result["messages"][0].content == "42"
