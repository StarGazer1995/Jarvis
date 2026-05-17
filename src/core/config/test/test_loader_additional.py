"""Additional config loader tests migrated from legacy `src/test` files."""


class TestConfigLoaderLastBatch:
    """config/loader 剩余测试"""

    def test_get_config_path_default(self):
        from src.core.config.loader import get_config_path

        path = get_config_path()
        assert str(path).endswith("config/llm_config.yaml")

    def test_config_singleton(self):
        from src.core.config.loader import get_config_loader

        loader1 = get_config_loader()
        loader2 = get_config_loader()
        assert loader1 is loader2


class TestConfigLoaderQuickWins:
    """Migrated methods from TestQuickWins."""

    def test_config_loader_env_substitution(self):
        """测试配置加载器环境变量替换"""
        import os
        import tempfile

        import yaml

        from src.core.config.loader import ConfigLoader

        data = {
            "global": {"default_provider": "openai"},
            "providers": {
                "openai": {
                    "type": "openai",
                    "enabled": True,
                    "default_model": "gpt-4",
                    "models": {"gpt-4": {}},
                    "api_key": "test-key",
                }
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            p = f.name
        try:
            loader = ConfigLoader(p)
            result = loader.load_config()
            assert result is not None
            assert len(result.providers) > 0
        finally:
            os.unlink(p)


class TestConfigLoaderFinalPush:
    """Migrated methods from TestFinalPush."""

    def test_config_loader_load_empty(self):
        from src.core.config.loader import get_config_loader

        loader = get_config_loader()
        assert loader is not None
