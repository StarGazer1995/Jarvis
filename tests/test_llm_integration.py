"""
Tests for LLM integration in ARK engine.
"""

import pytest
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
        from langchain_core.messages import SystemMessage, HumanMessage

        assert any(isinstance(msg, SystemMessage) for msg in messages)
        assert any(isinstance(msg, HumanMessage) for msg in messages)
