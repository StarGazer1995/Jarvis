"""
Prompt Manager 单元测试

测试 src/core/prompt/manager.py 的 PromptManager 类。
"""

import pytest
from langchain_core.messages import HumanMessage

from src.core.prompt.manager import PromptManager


@pytest.fixture
def manager():
    """Create a PromptManager."""
    return PromptManager()


class TestPromptManager:
    """测试 PromptManager 基础功能"""

    def test_init(self, manager):
        assert manager is not None
        assert hasattr(manager, "templates")

    def test_add_and_get_template(self, manager):
        """测试添加和获取模板"""
        from langchain_core.prompts import ChatPromptTemplate

        template = ChatPromptTemplate.from_messages([("system", "Test template")])
        manager.add_template("test_template", template)
        assert manager.get_template("test_template") == template
        assert manager.get_template("nonexistent") is None

    def test_list_templates(self, manager):
        """测试列出模板"""
        from langchain_core.prompts import ChatPromptTemplate

        manager.add_template("t1", ChatPromptTemplate.from_messages([("system", "T1")]))
        manager.add_template("t2", ChatPromptTemplate.from_messages([("system", "T2")]))
        templates = manager.list_templates()
        assert "t1" in templates
        assert "t2" in templates

    def test_render_template_not_found(self, manager):
        """测试渲染不存在的模板"""
        with pytest.raises(ValueError, match="提示词模板 'nonexistent' 不存在"):
            manager.render_template("nonexistent")

    def test_render_template_success(self, manager):
        """测试成功渲染模板"""
        from langchain_core.prompts import ChatPromptTemplate

        template = ChatPromptTemplate.from_messages([("system", "Hello {user_name}!")])
        manager.add_template("greeting", template)
        messages = manager.render_template("greeting", user_name="World")
        assert len(messages) == 1
        assert messages[0].content == "Hello World!"

    def test_render_template_failure(self, manager):
        """测试模板渲染失败"""
        from langchain_core.prompts import ChatPromptTemplate

        template = ChatPromptTemplate.from_messages([("system", "Hello {user_name}!")])
        manager.add_template("greeting", template)
        with pytest.raises(Exception):
            manager.render_template("greeting")

    def test_build_conversation_messages(self, manager):
        """测试构建对话消息"""
        from langchain_core.prompts import ChatPromptTemplate

        # Add system template
        template = ChatPromptTemplate.from_messages(
            [("system", "You are {agent_name}.")]
        )
        manager.add_template("jarvis_system", template)

        messages = manager.build_conversation_messages(
            user_input="Hello",
            conversation_history=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
            system_context={
                "agent_name": "Jarvis",
                "user_preferences": {"language": "en"},
                "available_tools": [{"name": "search", "description": "Search tool"}],
            },
        )

        assert len(messages) >= 1
        # First message should be system message
        assert "Jarvis" in messages[0].content

        # Should have HumanMessage for "Hello"
        last_msg = messages[-1]
        assert isinstance(last_msg, HumanMessage)
        assert last_msg.content == "Hello"

    def test_build_tool_usage_prompt(self, manager):
        """测试构建工具使用提示词"""
        from langchain_core.prompts import ChatPromptTemplate

        template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Tools: {available_tools}. Input: {user_input}. Intent: {intent}",
                )
            ]
        )
        manager.add_template("tool_usage_decision", template)

        result = manager.build_tool_usage_prompt(
            user_input="Search for Python",
            intent="search",
            available_tools=[
                {"name": "search", "description": "Web search"},
            ],
        )
        assert "Search for Python" in result
        assert "search" in result
        assert "Web search" in result

    def test_build_tool_usage_prompt_no_tools(self, manager):
        """测试无可用工具时构建提示词"""
        from langchain_core.prompts import ChatPromptTemplate

        template = ChatPromptTemplate.from_messages(
            [("system", "Tools: {available_tools}")]
        )
        manager.add_template("tool_usage_decision", template)

        result = manager.build_tool_usage_prompt(
            user_input="Hello",
            intent="chat",
            available_tools=[],
        )
        assert "No tools available" in result

    def test_build_response_generation_prompt(self, manager):
        """测试构建响应生成提示词"""
        from langchain_core.prompts import ChatPromptTemplate

        template = ChatPromptTemplate.from_messages(
            [("system", "Generate response. Input: {user_input}, Intent: {intent}")]
        )
        manager.add_template("response_generation", template)

        result = manager.build_response_generation_prompt(
            user_input="Hello",
            intent="chat",
            confidence=0.95,
            entities=[{"type": "greeting", "value": "hello"}],
            tool_results=[{"tool": "search", "result": "found"}],
            context={"user": "test"},
        )
        assert "Hello" in result
        assert "chat" in result

    def test_build_response_generation_prompt_empty(self, manager):
        """测试空数据时构建响应生成提示词"""
        from langchain_core.prompts import ChatPromptTemplate

        template = ChatPromptTemplate.from_messages(
            [("system", "Input: {user_input}, Intent: {intent}")]
        )
        manager.add_template("response_generation", template)

        result = manager.build_response_generation_prompt(
            user_input="Hi",
            intent="chat",
            confidence=0.5,
            entities=[],
            tool_results=[],
            context={},
        )
        assert "Hi" in result

    def test_build_conversation_messages_no_history(self, manager):
        """测试无历史消息时构建对话"""
        from langchain_core.prompts import ChatPromptTemplate

        template = ChatPromptTemplate.from_messages(
            [("system", "You are {agent_name}.")]
        )
        manager.add_template("jarvis_system", template)

        messages = manager.build_conversation_messages(
            user_input="Hi",
            conversation_history=[],
            system_context={
                "agent_name": "Jarvis",
                "user_preferences": {},
                "available_tools": [],
            },
        )
        assert len(messages) == 2  # system + user
        assert messages[-1].content == "Hi"

    def test_deep_research_prompt_includes_evidence_grounding(self, manager):
        """测试 deep research prompt 包含证据约束"""
        messages = manager.render_template(
            "deep_research_system", current_date="2026-05-20"
        )

        assert len(messages) == 1
        content = messages[0].content
        assert (
            "Use only information supported by collected tool observations." in content
        )
        assert (
            "If you cannot support a conclusion, explicitly say that the evidence is insufficient."
            in content
        )
        assert "claim-to-source mappings" in content
        assert "'summary'" in content
        assert "'claims'" in content
        assert "'sources'" in content
        assert "'insufficient_evidence'" in content
        assert "'evidence'" in content

    def test_deep_research_auditor_prompt_includes_observation_payload(self, manager):
        """测试 deep research auditor prompt 包含原始 observation 审计输入"""
        messages = manager.render_template(
            "deep_research_auditor",
            report_content='"Claims:\\n1. Example [Sources: https://example.com]"',
            observed_evidence_json='{"https://example.com": ["example evidence"]}',
        )

        assert len(messages) == 1
        content = messages[0].content
        assert "observed_evidence_by_url" in content
        assert '"supported"' in content
        assert '"issues"' in content
        assert '"per_source"' in content
