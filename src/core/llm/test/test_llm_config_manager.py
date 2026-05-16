"""
LLM Config Manager 单元测试

测试 src/core/llm/config.py 中的 LLMConfigManager 类。
"""

import os
from unittest.mock import patch

from src.core.llm.config import LLMConfigManager
from src.core.llm.types import LLMProvider


class TestLLMConfigManager:
    def test_init(self):
        mgr = LLMConfigManager()
        assert mgr.get_config() is None

    def test_load_config_from_dict(self):
        mgr = LLMConfigManager()
        config = mgr.load_config({"provider": "anthropic", "api_key": "ant-key"})
        assert config.provider == LLMProvider.ANTHROPIC
        assert config.api_key == "ant-key"

    def test_load_config_default(self):
        mgr = LLMConfigManager()
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}, clear=True):
            config = mgr.load_config()
        assert config.provider == LLMProvider.OLLAMA

    def test_get_config_after_load(self):
        mgr = LLMConfigManager()
        assert mgr.get_config() is None
        mgr.load_config({"provider": "openai", "api_key": "k"})
        assert mgr.get_config() is not None

    def test_update_config_no_existing(self):
        mgr = LLMConfigManager()
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=True):
            mgr.update_config(temperature=0.5)
        config = mgr.get_config()
        assert config is not None
        assert config.temperature == 0.5

    def test_update_config_with_existing(self):
        mgr = LLMConfigManager()
        mgr.load_config({"provider": "openai", "api_key": "k"})
        mgr.update_config(temperature=0.3, max_tokens=500)
        config = mgr.get_config()
        assert config.temperature == 0.3
        assert config.max_tokens == 500

    def test_update_config_invalid_key(self):
        mgr = LLMConfigManager()
        mgr.load_config({"provider": "openai", "api_key": "k"})
        mgr.update_config(nonexistent_field="value")  # Should not raise
        config = mgr.get_config()
        assert not hasattr(config, "nonexistent_field")


class TestLLMConfigEdgeCases:
    """测试 LLMConfig 边缘情况"""

    def test_is_configured_openai_without_key(self):
        """测试 OpenAI provider 没有 API key"""
        from src.core.llm.config import LLMConfigManager

        mgr = LLMConfigManager()
        mgr.load_config({"provider": "openai"})
        assert mgr.is_configured() is False

    def test_is_configured_openai_with_key(self):
        """测试 OpenAI provider 有 API key"""
        from src.core.llm.config import LLMConfigManager

        mgr = LLMConfigManager()
        mgr.load_config({"provider": "openai", "api_key": "sk-test"})
        assert mgr.is_configured() is True

    def test_is_configured_non_openai(self):
        """测试非 OpenAI provider"""
        from src.core.llm.config import LLMConfigManager

        mgr = LLMConfigManager()
        mgr.load_config({"provider": "anthropic"})
        # Non-OpenAI providers don't require API key check
        assert mgr.is_configured() is True

    def test_is_configured_no_config(self):
        """测试未加载配置时"""
        from src.core.llm.config import LLMConfigManager

        mgr = LLMConfigManager()
        assert mgr.is_configured() is False

    def test_get_provider_info_no_config(self):
        """测试未加载配置时获取 provider 信息"""
        from src.core.llm.config import LLMConfigManager

        mgr = LLMConfigManager()
        info = mgr.get_provider_info()
        assert info == {"provider": "none", "configured": False}

    def test_get_provider_info_configured(self):
        """测试已加载配置时获取 provider 信息"""
        from src.core.llm.config import LLMConfigManager

        mgr = LLMConfigManager()
        mgr.load_config({"provider": "openai", "api_key": "sk-test", "model": "gpt-4"})
        info = mgr.get_provider_info()
        assert info["provider"] == "openai"
        assert info["configured"] is True
        assert info["model"] == "gpt-4"

    def test_get_llm_config_global_fn(self):
        """测试全局 get_llm_config 函数"""
        from src.core.llm.config import get_llm_config, llm_config_manager

        # Reset
        llm_config_manager._config = None
        assert get_llm_config() is None

    def test_is_llm_configured_global_fn(self):
        """测试全局 is_llm_configured 函数"""
        from src.core.llm.config import is_llm_configured, llm_config_manager

        llm_config_manager._config = None
        assert is_llm_configured() is False

    def test_load_llm_config_global_fn(self):
        """测试全局 load_llm_config 函数"""
        from src.core.llm.config import load_llm_config

        config = load_llm_config({"provider": "anthropic", "api_key": "ant-key"})
        assert config is not None
        assert config.api_key == "ant-key"
