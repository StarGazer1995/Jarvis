"""
Tests for LLM integration in ARK engine.
"""

import pytest
from langchain_core.prompts import ChatPromptTemplate
from src.core.llm.config import LLMConfig, LLMProvider
from src.core.prompt.manager import PromptManager


class TestLLMConfig:
    """Test LLM configuration functionality."""

    def test_llm_config_creation(self):
        """Test LLMConfig creation."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI, model="gpt-3.5-turbo", api_key="test-key"
        )
        assert config.provider == LLMProvider.OPENAI
        assert config.model == "gpt-3.5-turbo"
        assert config.api_key == "test-key"

    def test_llm_config_from_dict(self):
        """Test LLM configuration creation from dictionary."""
        config_dict = {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "test-key",
            "max_tokens": 2000,
            "temperature": 0.5,
        }

        config = LLMConfig.from_dict(config_dict)

        assert config.provider == LLMProvider.OPENAI
        assert config.model == "gpt-4"
        assert config.api_key == "test-key"
        assert config.max_tokens == 2000
        assert config.temperature == 0.5

    def test_llm_config_from_dict_provider_case_insensitive(self):
        config = LLMConfig.from_dict({"provider": "OPENAI", "model": "gpt-4"})
        assert config.provider == LLMProvider.OPENAI

    def test_llm_config_from_dict_provider_fallback(self):
        config = LLMConfig.from_dict({"provider": "unknown-provider", "model": "gpt-4"})
        assert config.provider == LLMProvider.OPENAI

    def test_llm_config_to_dict(self):
        """Test LLM configuration conversion to dictionary."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-3.5-turbo",
            api_key="test-key",
            max_tokens=1500,
        )

        config_dict = config.to_dict()

        assert config_dict["provider"] == "openai"
        assert config_dict["model"] == "gpt-3.5-turbo"
        assert config_dict["api_key"] == "test-key"
        assert config_dict["max_tokens"] == 1500


class TestPromptManager:
    """Test prompt manager functionality."""

    @pytest.fixture
    def prompt_manager(self):
        """Create prompt manager instance."""
        return PromptManager()

    def test_prompt_manager_initialization(self, prompt_manager):
        """Test prompt manager initialization."""
        # PromptManager now automatically loads default templates
        assert len(prompt_manager.templates) > 0
        assert "response_generation" in prompt_manager.templates

    def test_load_default_templates(self, prompt_manager):
        """Test loading default templates."""
        # Templates are already loaded in __init__, but we can call it again
        prompt_manager.load_default_templates()

        # Check that default templates are loaded by name
        assert "jarvis_system" in prompt_manager.templates
        assert "conversation_response" in prompt_manager.templates
        assert "tool_usage_decision" in prompt_manager.templates
        assert "response_generation" in prompt_manager.templates

    def test_build_conversation_messages(self, prompt_manager):
        """Test building conversation messages."""
        prompt_manager.load_default_templates()

        messages = prompt_manager.build_conversation_messages(
            user_input="Hello",
            conversation_history=[],
            system_context={"agent_name": "Jarvis"},
            intent_info={"intent": "greeting", "confidence": 0.9},
            tool_results=[],
        )

        assert isinstance(messages, list)
        assert len(messages) > 0
        from langchain_core.messages import HumanMessage, SystemMessage

        assert any(isinstance(msg, SystemMessage) for msg in messages)
        assert any(isinstance(msg, HumanMessage) for msg in messages)

    def test_react_system_prompt_mentions_parallel_tool_calls(self, prompt_manager):
        """ReAct prompt should explicitly allow batched tool calls."""
        prompt = prompt_manager.render_template("react_system")[0].content
        assert "tool_calls" in prompt
        assert "parallel" in prompt.lower()

    def test_master_system_prompt_mentions_parallel_tool_calls(self, prompt_manager):
        """Master prompt should direct the orchestrator to batch independent tools."""
        prompt = prompt_manager.render_template(
            "master_system",
            todo_status="No tasks in todo list.",
            tools_desc="Available Tools:\n- search: Search the web",
            agents_desc="",
        )[0].content
        assert "tool_calls" in prompt
        assert "depends_on" in prompt
        assert "parallel" in prompt.lower()

    def test_add_get_and_list_template(self, prompt_manager):
        """Custom templates should be added and discoverable."""
        custom = ChatPromptTemplate.from_messages([("system", "hello {name}")])
        prompt_manager.add_template("custom_template", custom)

        assert prompt_manager.get_template("custom_template") is custom
        assert "custom_template" in prompt_manager.list_templates()

    def test_render_template_missing_raises(self, prompt_manager):
        """Missing template names should raise a clear error."""
        with pytest.raises(ValueError, match="不存在"):
            prompt_manager.render_template("missing_template")

    def test_build_conversation_messages_with_history(self, prompt_manager):
        """History should be converted into LangChain message objects."""
        messages = prompt_manager.build_conversation_messages(
            user_input="Current question",
            conversation_history=[
                {"role": "user", "content": "Earlier user"},
                {"role": "assistant", "content": "Earlier assistant"},
                {"role": "ignored", "content": "Should not appear"},
            ],
            system_context={
                "agent_name": "Jarvis",
                "user_preferences": {"lang": "zh"},
                "available_tools": [{"name": "search"}],
            },
        )

        contents = [message.content for message in messages]
        assert "Earlier user" in contents
        assert "Earlier assistant" in contents
        assert "Should not appear" not in contents
        assert contents[-1] == "Current question"

    def test_build_tool_usage_prompt_formats_tool_list(self, prompt_manager):
        """Tool usage prompt should embed tool names and descriptions."""
        prompt = prompt_manager.build_tool_usage_prompt(
            user_input="查天气",
            intent="weather",
            available_tools=[
                {"name": "get_weather", "description": "Get weather"},
                {"name": "get_time", "description": "Get time"},
            ],
        )

        assert "get_weather" in prompt
        assert "Get time" in prompt

    def test_build_response_generation_prompt_formats_payloads(self, prompt_manager):
        """Response generation prompt should serialize entities, tool results, and context."""
        prompt = prompt_manager.build_response_generation_prompt(
            user_input="总结一下",
            intent="summary",
            confidence=0.9,
            entities=[{"name": "AI"}],
            tool_results=[{"tool": "search", "result": "done"}],
            context={"lang": "zh"},
        )

        assert "summary" in prompt
        assert '"name": "AI"' in prompt
        assert '"tool": "search"' in prompt
        assert '"lang": "zh"' in prompt
