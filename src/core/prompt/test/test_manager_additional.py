"""Additional prompt manager tests migrated from legacy `src/test` files."""

import pytest


class TestPromptManagerDetailedCoverage:
    """PromptManager 边界测试"""

    def test_prompt_manager_init(self):
        from src.core.prompt.manager import PromptManager

        pm = PromptManager()
        assert pm is not None
        assert hasattr(pm, "templates")

    def test_prompt_manager_list_templates(self):
        from src.core.prompt.manager import PromptManager

        pm = PromptManager()
        templates = pm.list_templates()
        assert isinstance(templates, list)


class TestPromptManagerRemaining:
    """PromptManager 更多测试"""

    def test_prompt_manager_list_templates(self):
        from src.core.prompt.manager import PromptManager

        pm = PromptManager()
        templates = pm.list_templates()
        assert isinstance(templates, list)
        assert len(templates) > 0

    def test_prompt_manager_str(self):
        from src.core.prompt.manager import PromptManager

        pm = PromptManager()
        s = str(pm)
        assert "PromptManager" in s


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


class TestPromptManagerQuickWins:
    """Migrated methods from TestQuickWins."""

    def test_prompt_manager_render_system(self):
        """测试渲染系统提示词模板"""
        from src.core.prompt.manager import PromptManager

        pm = PromptManager()
        result = pm.render_template("react_system")
        assert result is not None


class TestPromptManagerFinalPush:
    """Migrated methods from TestFinalPush."""

    def test_prompt_manager_str(self):
        from src.core.prompt.manager import PromptManager

        pm = PromptManager()
        s = str(pm)
        assert "PromptManager" in s


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
