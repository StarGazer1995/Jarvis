"""
配置加载器测试模块

测试YAML配置文件的加载、解析、验证和环境变量替换功能
"""

import os
import tempfile
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from src.core.common.exceptions import ConfigurationError
from src.core.config.loader import (
    ConfigLoader,
    FeatureConfig,
    GlobalConfig,
    LLMConfig,
    ModelConfig,
    ProviderConfig,
    RetryConfig,
    TimeoutConfig,
    get_config_path,
    load_llm_config,
)


class TestRetryConfig:
    """测试重试配置"""

    def test_retry_config_creation(self):
        """测试重试配置创建"""
        config = RetryConfig(max_attempts=3, initial_delay=1.0, exponential_base=2.0)
        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.exponential_base == 2.0

    def test_retry_config_defaults(self):
        """测试重试配置默认值"""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.exponential_base == 2.0


class TestTimeoutConfig:
    """测试超时配置"""

    def test_timeout_config_creation(self):
        """测试超时配置创建"""
        config = TimeoutConfig(connect=30.0, read=60.0, total=90.0)
        assert config.connect == 30.0
        assert config.read == 60.0
        assert config.total == 90.0

    def test_timeout_config_defaults(self):
        """测试超时配置默认值"""
        config = TimeoutConfig()
        assert config.connect == 30.0
        assert config.read == 120.0
        assert config.total == 180.0


class TestModelConfig:
    """测试模型配置"""

    def test_model_config_creation(self):
        """测试模型配置创建"""
        config = ModelConfig(max_tokens=4000, temperature=0.7, top_p=0.9)
        assert config.max_tokens == 4000
        assert config.temperature == 0.7
        assert config.top_p == 0.9

    def test_model_config_defaults(self):
        """测试模型配置默认值"""
        config = ModelConfig()
        assert config.max_tokens == 4096
        assert config.temperature == 0.7
        assert config.top_p == 1.0


class TestProviderConfig:
    """测试提供商配置"""

    def test_provider_config_creation(self):
        """测试提供商配置创建"""
        model = ModelConfig()
        retry = RetryConfig(max_attempts=5)
        timeout = TimeoutConfig(total=120.0)

        config = ProviderConfig(
            enabled=True,
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            default_model="gpt-4",
            models=[model],
            retry=retry,
            timeout=timeout,
        )

        assert config.enabled is True
        assert config.api_key == "test-key"
        assert config.base_url == "https://api.openai.com/v1"
        assert config.default_model == "gpt-4"
        assert len(config.models) == 1
        assert config.retry.max_attempts == 5
        assert config.timeout.total == 120.0

    def test_provider_config_defaults(self):
        """测试提供商配置默认值"""
        config = ProviderConfig()
        assert config.enabled is True
        assert config.api_key is None
        assert config.base_url is None
        assert config.default_model == "gpt-4"
        assert config.models == {}


class TestConfigLoader:
    """测试配置加载器"""

    @pytest.fixture
    def sample_config_data(self) -> dict[str, Any]:
        """示例配置数据"""
        return {
            "global": {
                "default_provider": "openai",
                "retry": {
                    "max_attempts": 3,
                    "initial_delay": 1.0,
                    "exponential_base": 2.0,
                },
                "timeout": {"connect": 30.0, "read": 120.0, "total": 180.0},
                "log_level": "INFO",
            },
            "providers": {
                "openai": {
                    "enabled": True,
                    "api_key": "${OPENAI_API_KEY}",
                    "base_url": "https://api.openai.com/v1",
                    "default_model": "gpt-4",
                    "models": {"gpt-4": {"max_tokens": 4000, "temperature": 0.7}},
                },
                "mock": {
                    "enabled": True,
                    "default_model": "mock-model",
                    "models": {
                        "mock-model": {
                            "max_tokens": 2048,
                            "temperature": 0.5,
                            "response_delay": 0.1,
                        }
                    },
                },
            },
            "environment": {"development": {"debug": True, "log_level": "DEBUG"}},
            "features": {
                "streaming_enabled": True,
                "streaming_chunk_size": 1024,
                "context_max_history": 10,
                "context_max_tokens": 8192,
                "cache_enabled": True,
                "cache_ttl": 3600,
            },
        }

    @pytest.fixture
    def temp_config_file(self, sample_config_data):
        """创建临时配置文件"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_config_data, f)
            temp_path = f.name

        yield temp_path

        # 清理
        os.unlink(temp_path)

    def test_load_config_from_file(self, temp_config_file):
        """测试从文件加载配置"""
        loader = ConfigLoader(temp_config_file)
        config = loader.load_config()

        assert isinstance(config, LLMConfig)
        assert config.global_config.default_provider == "openai"
        assert "openai" in config.providers
        assert "mock" in config.providers
        assert config.providers["openai"].enabled is True

    def test_load_config_with_env_substitution(self, temp_config_file):
        """测试环境变量替换"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
            loader = ConfigLoader(temp_config_file)
            config = loader.load_config()

            assert config.providers["openai"].api_key == "test-api-key"

    def test_load_config_missing_env_var(self, temp_config_file):
        """测试缺失环境变量"""
        with patch.dict(os.environ, {}, clear=True):
            loader = ConfigLoader(temp_config_file)
            config = loader.load_config()

            # 应该保持原始值 (实际上如果未定义且无默认值，会被替换为空字符串或None，取决于解析逻辑)
            # 当前实现中，ConfigLoader._substitute_env_vars将缺失的环境变量替换为空字符串
            # 但YAML解析空值可能导致None
            assert config.providers["openai"].api_key is None

    def test_load_config_file_not_found(self):
        """测试配置文件不存在"""
        loader = ConfigLoader("nonexistent.yaml")

        with pytest.raises(ConfigurationError, match="配置加载失败"):
            loader.load_config()

    def test_load_config_invalid_yaml(self):
        """测试无效的YAML文件"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name

        try:
            loader = ConfigLoader(temp_path)
            with pytest.raises(ConfigurationError, match="YAML解析失败"):
                loader.load_config()
        finally:
            os.unlink(temp_path)

    def test_validate_config_valid(self, sample_config_data):
        """测试有效配置验证"""
        from src.core.config.loader import Environment

        loader = ConfigLoader("dummy.yaml")
        config = loader._parse_config(sample_config_data, Environment.DEVELOPMENT)

        # 不应该抛出异常
        loader._validate_config(config)

    def test_validate_config_missing_providers(self):
        """测试缺失提供商配置"""
        loader = ConfigLoader("dummy.yaml")
        config = LLMConfig(
            global_config=GlobalConfig(default_provider="nonexistent"),
            providers={},
            features=FeatureConfig(),
        )

        with pytest.raises(ConfigurationError, match="默认提供商.*未配置"):
            loader._validate_config(config)

    def test_validate_config_invalid_default_provider(self, sample_config_data):
        """测试无效的默认提供商"""
        from src.core.config.loader import Environment

        loader = ConfigLoader("dummy.yaml")
        sample_config_data["global"]["default_provider"] = "nonexistent"

        with pytest.raises(ConfigurationError, match="默认提供商.*未配置"):
            config = loader._parse_config(sample_config_data, Environment.DEVELOPMENT)
            loader._validate_config(config)

    def test_substitute_env_vars(self):
        """测试环境变量替换"""
        loader = ConfigLoader("dummy.yaml")

        with patch.dict(
            os.environ, {"TEST_VAR": "test_value", "ANOTHER_VAR": "another_value"}
        ):
            config_content = """
key1: ${TEST_VAR}
key2: prefix_${ANOTHER_VAR}_suffix
key3: ${MISSING_VAR}
key4: no_substitution
nested:
  key5: ${TEST_VAR}
"""

            result = loader._substitute_env_vars(config_content)

            assert "test_value" in result
            assert "prefix_another_value_suffix" in result
            assert "key3:" in result  # MISSING_VAR becomes empty string
            assert "no_substitution" in result

    def test_parse_config_complete(self, sample_config_data):
        """测试完整配置解析"""
        from src.core.config.loader import Environment

        loader = ConfigLoader("dummy.yaml")
        config = loader._parse_config(sample_config_data, Environment.DEVELOPMENT)

        # 验证全局配置
        assert config.global_config.default_provider == "openai"
        assert config.global_config.retry.max_attempts == 3
        assert config.global_config.timeout.total == 180.0  # 修正默认值
        assert config.global_config.log_level == "INFO"

        # 验证提供商配置
        assert len(config.providers) == 2
        assert "openai" in config.providers
        assert "mock" in config.providers

        openai_config = config.providers["openai"]
        assert openai_config.enabled is True
        assert openai_config.default_model == "gpt-4"
        assert "gpt-4" in openai_config.models

        # 验证功能配置
        assert config.features.streaming_enabled is True
        assert config.features.context_max_history == 10


class TestConfigHelperFunctions:
    """测试配置辅助函数"""

    def test_get_config_path_with_path(self):
        """测试指定路径的配置路径获取"""
        custom_path = "/custom/path/config.yaml"
        result = get_config_path(custom_path)
        assert str(result) == custom_path

    def test_get_config_path_default(self):
        """测试默认配置路径获取"""
        result = get_config_path()
        assert str(result).endswith("config/llm_config.yaml")
        assert "00_Jarvis" in str(result)

    @patch("src.core.config.loader.ConfigLoader.load_config")
    def test_load_llm_config(self, mock_load):
        """测试LLM配置加载函数"""
        mock_config = LLMConfig(
            global_config=GlobalConfig(default_provider="test"),
            providers={},
            features=FeatureConfig(),
        )
        mock_load.return_value = mock_config

        result = load_llm_config(environment="development", config_path="test.yaml")

        mock_load.assert_called_once_with("development", False)
        assert result == mock_config

    @patch("src.core.config.loader.ConfigLoader.load_config")
    def test_load_llm_config_default_path(self, mock_load):
        """测试默认路径的LLM配置加载"""
        mock_config = LLMConfig(
            global_config=GlobalConfig(default_provider="test"),
            providers={},
            features=FeatureConfig(),
        )
        mock_load.return_value = mock_config

        result = load_llm_config()

        # 验证调用了load_config
        mock_load.assert_called_once_with(None, False)
        assert result == mock_config


class TestConfigLoaderEdgeCases:
    """测试配置加载器边缘情况"""

    def test_empty_config_file(self):
        """测试空配置文件"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            loader = ConfigLoader(temp_path)
            with pytest.raises(ConfigurationError):
                loader.load_config()
        finally:
            os.unlink(temp_path)

    def test_config_with_only_comments(self):
        """测试只有注释的配置文件"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("# This is a comment\n# Another comment\n")
            temp_path = f.name

        try:
            loader = ConfigLoader(temp_path)
            with pytest.raises(ConfigurationError):
                loader.load_config()
        finally:
            os.unlink(temp_path)

    def test_config_with_circular_env_vars(self):
        """测试循环环境变量引用"""
        with patch.dict(os.environ, {"VAR1": "${VAR2}", "VAR2": "${VAR1}"}):
            loader = ConfigLoader("dummy.yaml")
            content = "key: ${VAR1}"

            # VAR1 resolves to VAR2's value, which is ${VAR1}
            result = loader._substitute_env_vars(content)
            assert "${VAR2}" in result

    def test_config_with_nested_env_vars(self):
        """测试嵌套环境变量"""
        with patch.dict(
            os.environ, {"BASE_URL": "https://api.example.com", "VERSION": "v1"}
        ):
            loader = ConfigLoader("dummy.yaml")
            content = "url: ${BASE_URL}/${VERSION}/chat"

            result = loader._substitute_env_vars(content)
            assert "https://api.example.com/v1/chat" in result

    def test_config_with_special_characters(self):
        """测试包含特殊字符的配置"""
        from src.core.config.loader import Environment

        config_data = {
            "global": {"default_provider": "test"},
            "providers": {
                "test": {"api_key": "key_with_!@#$%^&*()_+-={}[]|\\:;\"'<>?,./"}
            },
        }

        loader = ConfigLoader("dummy.yaml")
        config = loader._parse_config(config_data, Environment.DEVELOPMENT)

        assert (
            config.providers["test"].api_key
            == "key_with_!@#$%^&*()_+-={}[]|\\:;\"'<>?,./"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
