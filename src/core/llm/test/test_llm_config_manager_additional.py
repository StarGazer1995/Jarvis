"""Additional LLM config manager tests migrated from legacy `src/test` files."""

import os
from unittest.mock import patch


class TestLLMConfigMoreCoverage:
    """llm/config 剩余测试"""

    def test_config_mgr_update_multiple(self):
        from src.core.llm.config import LLMConfigManager

        mgr = LLMConfigManager()
        mgr.load_config({"provider": "openai", "api_key": "k", "model": "gpt-4"})
        mgr.update_config(temperature=0.5, max_tokens=2000)
        cfg = mgr.get_config()
        assert cfg.temperature == 0.5
        assert cfg.max_tokens == 2000

    def test_config_mgr_update_nonexistent_field(self):
        from src.core.llm.config import LLMConfigManager

        mgr = LLMConfigManager()
        mgr.load_config({"provider": "openai", "api_key": "k", "model": "gpt-4"})
        # Should not raise
        mgr.update_config(nonexistent_field="value")


class TestLLMConfigRemainingCoverage:
    """llm/config.py 剩余方法全覆盖"""

    def test_is_configured_false_when_no_config(self):
        from src.core.llm.config import LLMConfigManager

        mgr = LLMConfigManager()
        assert mgr.is_configured() is False

    def test_is_configured_openai_without_key(self):
        from src.core.llm.config import LLMConfigManager

        mgr = LLMConfigManager()
        mgr.load_config({"provider": "openai", "model": "gpt-4"})
        # api_key is None, so should return False
        assert mgr.is_configured() is False

    def test_is_configured_openai_with_key(self):
        from src.core.llm.config import LLMConfigManager

        mgr = LLMConfigManager()
        mgr.load_config({"provider": "openai", "api_key": "sk-key", "model": "gpt-4"})
        assert mgr.is_configured() is True

    def test_get_provider_info_no_config(self):
        from src.core.llm.config import LLMConfigManager

        mgr = LLMConfigManager()
        info = mgr.get_provider_info()
        assert info["provider"] == "none"
        assert info["configured"] is False


class TestLLMConfigFinalPush:
    """Migrated methods from TestFinalPush."""

    def test_config_from_env(self):
        import os

        from src.core.llm.config import LLMConfigManager

        env = {
            "LLM_PROVIDER": "anthropic",
            "OPENAI_API_KEY": "sk-key",
            "LLM_MODEL": "claude-3",
        }
        with patch.dict(os.environ, env, clear=True):
            mgr = LLMConfigManager()
            cfg = mgr.load_config()
            assert cfg is not None
            assert cfg.model == "claude-3"


class TestLLMConfigTier1:
    """llm/config.py 全覆盖"""

    def test_is_configured_for_non_openai(self):
        from src.core.llm.config import LLMConfigManager

        mgr = LLMConfigManager()
        mgr.load_config({"provider": "ollama", "model": "llama3"})
        assert mgr.is_configured() is True

    def test_get_provider_info_with_config(self):
        from src.core.llm.config import LLMConfigManager

        mgr = LLMConfigManager()
        mgr.load_config(
            {"provider": "anthropic", "api_key": "sk-ant", "model": "claude-3"}
        )
        info = mgr.get_provider_info()
        assert info["configured"] is True
        assert info["model"] == "claude-3"

    def test_convert_to_client_config(self):
        import tempfile

        import yaml

        from src.core.config.loader import load_llm_config as load_yaml
        from src.core.llm.config import convert_to_client_config

        data = {
            "global": {"default_provider": "openai"},
            "providers": {
                "openai": {
                    "type": "openai",
                    "enabled": True,
                    "default_model": "gpt-4",
                    "models": {"gpt-4": {"max_tokens": 2048}},
                    "api_key": "sk-test",
                }
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            p = f.name
        try:
            yaml_cfg = load_yaml(config_path=p)
            client_cfg = convert_to_client_config(yaml_cfg)
            assert client_cfg is not None
            assert client_cfg.model == "gpt-4"
            assert client_cfg.max_tokens == 2048
        finally:
            os.unlink(p)
