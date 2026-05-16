"""
配置管理器单元测试

测试配置管理器的各种功能和配置源优先级。
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.llm.client import LLMProvider
from src.core.llm.utils.config_manager import (
    ConfigManager,
    create_llm_config,
    get_config_manager,
)
from src.core.llm.utils.error_handler import LLMConfigurationError


class TestConfigManager:
    """配置管理器测试类"""

    @pytest.fixture
    def temp_config_file(self):
        """临时配置文件"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            config_content = """
global:
  timeout: 60.0
  extra_params:
    top_p: 0.9

openai:
  model: gpt-4
  temperature: 0.5
  max_tokens: 2000
  base_url: https://api.openai.com/v1
"""
            f.write(config_content)
            f.flush()
            yield f.name

        # 清理
        os.unlink(f.name)

    @pytest.fixture
    def env_vars(self):
        """环境变量设置"""
        env_vars = {
            "OPENAI_API_KEY": "test-api-key-from-env",
            "OPENAI_MODEL": "gpt-3.5-turbo-from-env",
            "OPENAI_TEMPERATURE": "0.8",
            "OPENAI_MAX_TOKENS": "1500",
        }

        with patch.dict(os.environ, env_vars):
            yield env_vars

    def test_config_manager_initialization(self):
        """测试配置管理器初始化"""
        manager = ConfigManager()
        assert manager.config_file is None
        assert manager._file_config == {}

    def test_config_manager_with_file(self, temp_config_file):
        """测试带配置文件的配置管理器"""
        manager = ConfigManager(temp_config_file)
        assert manager.config_file == Path(temp_config_file)
        assert "global" in manager._file_config
        assert "openai" in manager._file_config

    def test_default_config_creation(self):
        """测试默认配置创建"""
        manager = ConfigManager()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            config = manager.create_config(LLMProvider.OPENAI)

        assert config.provider == LLMProvider.OPENAI
        assert config.api_key == "test-key"
        assert config.model == "gpt-3.5-turbo"  # 默认值
        assert config.temperature == 0.7  # 默认值
        assert config.max_tokens == 1000  # 默认值

    def test_env_var_override(self, env_vars):
        """测试环境变量覆盖"""
        manager = ConfigManager()
        config = manager.create_config(LLMProvider.OPENAI)

        assert config.api_key == "test-api-key-from-env"
        assert config.model == "gpt-3.5-turbo-from-env"
        assert config.temperature == 0.8
        assert config.max_tokens == 1500

    def test_file_config_override(self, temp_config_file):
        """测试配置文件覆盖"""
        manager = ConfigManager(temp_config_file)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            config = manager.create_config(LLMProvider.OPENAI)

        assert config.api_key == "test-key"  # 来自环境变量
        assert config.model == "gpt-4"  # 来自配置文件
        assert config.temperature == 0.5  # 来自配置文件
        assert config.max_tokens == 2000  # 来自配置文件
        assert config.timeout == 60.0  # 来自全局配置

    def test_kwargs_override(self, temp_config_file, env_vars):
        """测试直接参数覆盖"""
        manager = ConfigManager(temp_config_file)

        config = manager.create_config(
            LLMProvider.OPENAI, model="gpt-4-turbo", temperature=0.9, max_tokens=3000
        )

        assert config.api_key == "test-api-key-from-env"  # 来自环境变量
        assert config.model == "gpt-4-turbo"  # 来自kwargs
        assert config.temperature == 0.9  # 来自kwargs
        assert config.max_tokens == 3000  # 来自kwargs

    def test_missing_api_key_error(self):
        """测试缺少API密钥错误"""
        manager = ConfigManager()

        with pytest.raises(LLMConfigurationError) as exc_info:
            manager.create_config(LLMProvider.OPENAI)

        assert "API密钥未设置" in str(exc_info.value)
        assert exc_info.value.config_field == "api_key"

    def test_invalid_temperature_error(self):
        """测试无效温度错误"""
        manager = ConfigManager()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with pytest.raises(LLMConfigurationError) as exc_info:
                manager.create_config(LLMProvider.OPENAI, temperature=3.0)

        assert "temperature必须在0-2之间" in str(exc_info.value)

    def test_invalid_max_tokens_error(self):
        """测试无效最大tokens错误"""
        manager = ConfigManager()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with pytest.raises(LLMConfigurationError) as exc_info:
                manager.create_config(LLMProvider.OPENAI, max_tokens=-1)

        assert "max_tokens必须大于0" in str(exc_info.value)

    def test_invalid_timeout_error(self):
        """测试无效超时错误"""
        manager = ConfigManager()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with pytest.raises(LLMConfigurationError) as exc_info:
                manager.create_config(LLMProvider.OPENAI, timeout=-1)

        assert "timeout必须大于0" in str(exc_info.value)

    def test_env_value_conversion(self):
        """测试环境变量值转换"""
        manager = ConfigManager()

        # 测试浮点数转换
        assert manager._convert_env_value("temperature", "0.5") == 0.5
        assert manager._convert_env_value("timeout", "30.0") == 30.0

        # 测试整数转换
        assert manager._convert_env_value("max_tokens", "1000") == 1000

        # 测试字符串保持不变
        assert manager._convert_env_value("model", "gpt-4") == "gpt-4"

    def test_env_value_conversion_error(self):
        """测试环境变量值转换错误"""
        manager = ConfigManager()

        with pytest.raises(LLMConfigurationError):
            manager._convert_env_value("temperature", "invalid")

        with pytest.raises(LLMConfigurationError):
            manager._convert_env_value("max_tokens", "invalid")

    def test_get_available_models(self):
        """测试获取可用模型"""
        manager = ConfigManager()

        openai_models = manager.get_available_models(LLMProvider.OPENAI)
        assert "gpt-4" in openai_models
        assert "gpt-3.5-turbo" in openai_models

        # 测试不支持的提供商
        other_models = manager.get_available_models(LLMProvider.ANTHROPIC)
        assert other_models == []

    def test_create_config_template(self):
        """测试创建配置模板"""
        manager = ConfigManager()

        with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as f:
            template_path = f.name

        try:
            manager.create_config_template(template_path)

            # 验证文件存在
            assert Path(template_path).exists()

            # 验证内容
            with open(template_path) as f:
                content = f.read()
                assert "global:" in content
                assert "openai:" in content
                assert "gpt-3.5-turbo" in content

        finally:
            os.unlink(template_path)

    def test_dotenv_loading(self):
        """测试.env文件加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            with open(env_path, "w") as f:
                f.write("OPENAI_API_KEY=test-key-from-dotenv\n")
                f.write("OPENAI_MODEL=gpt-4-from-dotenv\n")

            try:
                # Mock dotenv加载
                with patch(
                    "src.core.llm.utils.config_manager.load_dotenv"
                ) as mock_load:
                    with patch(
                        "src.core.llm.utils.config_manager.Path.cwd"
                    ) as mock_cwd:
                        mock_cwd.return_value = Path(tmpdir)

                        ConfigManager()

                        # 验证load_dotenv被调用
                        mock_load.assert_called()

            except Exception as e:
                raise e


class TestGlobalFunctions:
    """全局函数测试"""

    def test_get_config_manager_singleton(self):
        """测试配置管理器单例"""
        manager1 = get_config_manager()
        manager2 = get_config_manager()

        assert manager1 is manager2

    def test_get_config_manager_with_file(self, temp_config_file):
        """测试带文件的配置管理器"""
        manager = get_config_manager(temp_config_file)
        assert manager.config_file == Path(temp_config_file)

    def test_create_llm_config_convenience(self):
        """测试便捷配置创建函数"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            config = create_llm_config(
                LLMProvider.OPENAI, model="gpt-4", temperature=0.5
            )

        assert config.provider == LLMProvider.OPENAI
        assert config.api_key == "test-key"
        assert config.model == "gpt-4"
        assert config.temperature == 0.5


class TestConfigManagerEdgeCases:
    """配置管理器边缘情况测试"""

    def test_nonexistent_config_file(self):
        """测试不存在的配置文件"""
        manager = ConfigManager("/nonexistent/path/config.yml")

        # 应该不会抛出异常，只是记录警告
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            config = manager.create_config(LLMProvider.OPENAI)
            assert config.api_key == "test-key"

    def test_invalid_yaml_file(self):
        """测试无效的YAML文件"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()
            invalid_yaml_path = f.name

        try:
            # 应该不会抛出异常，只是记录错误
            manager = ConfigManager(invalid_yaml_path)

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                config = manager.create_config(LLMProvider.OPENAI)
                assert config.api_key == "test-key"

        finally:
            os.unlink(invalid_yaml_path)

    def test_unsupported_config_file_format(self):
        """测试不支持的配置文件格式"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"test": "value"}')
            f.flush()
            json_path = f.name

        try:
            # 应该记录警告但不抛出异常
            manager = ConfigManager(json_path)

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                config = manager.create_config(LLMProvider.OPENAI)
                assert config.api_key == "test-key"

        finally:
            os.unlink(json_path)

    def test_missing_yaml_dependency(self):
        """测试缺少PyYAML依赖"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("test: value")
            f.flush()
            yaml_path = f.name

        try:
            with patch("src.core.llm.utils.config_manager.YAML_AVAILABLE", False):
                with patch(
                    "src.core.llm.utils.config_manager.logging.getLogger"
                ) as mock_get_logger:
                    mock_logger = MagicMock()
                    mock_get_logger.return_value = mock_logger

                    ConfigManager(yaml_path)

                    # 验证错误被记录
                    mock_logger.error.assert_called()
                    assert "加载配置文件失败" in str(mock_logger.error.call_args)

        finally:
            os.unlink(yaml_path)

    def test_missing_dotenv_dependency(self):
        """测试缺少python-dotenv依赖"""
        with patch("src.core.llm.utils.config_manager.DOTENV_AVAILABLE", False):
            # 应该不会抛出异常，只是跳过.env文件加载
            manager = ConfigManager()

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                config = manager.create_config(LLMProvider.OPENAI)
                assert config.api_key == "test-key"
