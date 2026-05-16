from unittest.mock import Mock

from src.core.ark.nodes.master import MasterNode
from src.core.llm.client import LLMManager


class TestMasterNode:
    def test_init(self):
        llm_manager = Mock(spec=LLMManager)
        node = MasterNode(llm_manager=llm_manager)
        assert node.prompt_manager is not None

    def test_get_system_prompt_empty_state(self):
        llm_manager = Mock(spec=LLMManager)
        node = MasterNode(llm_manager=llm_manager)

        state = {"todo_list": [], "available_tools": {}, "messages": []}

        prompt_messages = node._get_system_prompt(state)
        prompt = prompt_messages[0].content

        assert "No tasks in todo list." in prompt
        assert "No tools available." in prompt
        assert (
            "You are Jarvis, an intelligent agent acting as an Orchestrator." in prompt
        )

    def test_get_system_prompt_with_data(self):
        llm_manager = Mock(spec=LLMManager)
        node = MasterNode(llm_manager=llm_manager)

        state = {
            "todo_list": [{"id": "1", "status": "pending", "description": "Test Task"}],
            "available_tools": {
                "test_tool": {
                    "description": "A test tool",
                    "input_schema": {"type": "object"},
                }
            },
            "messages": [],
        }

        prompt_messages = node._get_system_prompt(state)
        prompt = prompt_messages[0].content

        assert "Current Todo List:" in prompt
        assert "- [1] pending: Test Task" in prompt
        assert "Available Tools:" in prompt
        assert "- test_tool: A test tool" in prompt

    def test_init_with_agents(self):
        """测试带 agents 初始化"""
        from src.core.ark.utils import AgentSpec

        llm_manager = Mock(spec=LLMManager)
        agent1 = Mock(spec=AgentSpec)
        agent1.name = "agent1"
        agent2 = Mock(spec=AgentSpec)
        agent2.name = "agent2"
        agents = [agent1, agent2]
        node = MasterNode(llm_manager=llm_manager, agents=agents)
        assert len(node.agents) == 2
        assert "agent1" in node.agent_map
        assert "agent2" in node.agent_map

    def test_init_empty_agents(self):
        """测试空 agents 列表"""
        llm_manager = Mock(spec=LLMManager)
        node = MasterNode(llm_manager=llm_manager, agents=[])
        assert node.agents == []
        assert node.agent_map == {}

    def test_convert_messages(self):
        """测试消息转换"""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        llm_manager = Mock(spec=LLMManager)
        node = MasterNode(llm_manager=llm_manager)

        msgs = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]

        result = node._convert_messages(msgs)
        assert len(result) == 3
        assert result[0].role == "system"
        assert result[1].role == "user"
        assert result[2].role == "assistant"

    def test_convert_messages_with_tool(self):
        """测试包含 ToolMessage 的消息转换"""
        from langchain_core.messages import (
            AIMessage,
            ToolMessage,
        )

        llm_manager = Mock(spec=LLMManager)
        node = MasterNode(llm_manager=llm_manager)

        msgs = [
            AIMessage(content="", tool_calls=[{"name": "test", "args": {}, "id": "1"}]),
            ToolMessage(content="Result", tool_call_id="1", name="test"),
        ]

        result = node._convert_messages(msgs)
        assert len(result) == 2
        # ToolMessage should be converted to user role with Observation prefix
        assert result[1].role == "user"
        assert "Observation:" in result[1].content

    def test_convert_messages_with_agent_map(self):
        """测试 agent_map 中的消息"""
        from langchain_core.messages import HumanMessage

        llm_manager = Mock(spec=LLMManager)
        node = MasterNode(llm_manager=llm_manager)
        node.agent_map = {"worker1": Mock()}

        msgs = [
            HumanMessage(content="output", name="worker1"),
        ]

        result = node._convert_messages(msgs)
        assert len(result) == 1
        assert "WORKER OUTPUT" in result[0].content

    def test_get_system_prompt_with_result(self):
        """测试待办事项有结果时"""
        llm_manager = Mock(spec=LLMManager)
        node = MasterNode(llm_manager=llm_manager)

        state = {
            "todo_list": [
                {
                    "id": "1",
                    "status": "completed",
                    "description": "Task 1",
                    "result": "Done",
                }
            ],
            "available_tools": {},
            "messages": [],
        }

        prompt = node._get_system_prompt(state)
        content = prompt[0].content
        assert "Result: Done" in content
