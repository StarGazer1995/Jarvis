"""
LLM Config Manager 单元测试

测试 src/core/llm/config.py 中的 LLMConfigManager 类。
"""

import os
from unittest.mock import patch

import pytest

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
