"""
LLM配置管理模块

提供灵活的配置管理，支持环境变量、配置文件和默认值。
"""

import os
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from ..types import LLMConfig, LLMProvider
from .error_handler import LLMConfigurationError


class ConfigManager:
    """
    配置管理器

    支持多种配置源的优先级管理：
    1. 直接传入的参数（最高优先级）
    2. 环境变量
    3. 配置文件
    4. 默认值（最低优先级）
    """

    # 默认配置
    DEFAULT_CONFIG = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 1000,
        "timeout": 30.0,
        "base_url": None,
        "extra_params": {},
    }

    # 环境变量映射
    ENV_MAPPING = {
        "api_key": ["OPENAI_API_KEY", "LLM_API_KEY"],
        "model": ["OPENAI_MODEL", "LLM_MODEL"],
        "temperature": ["OPENAI_TEMPERATURE", "LLM_TEMPERATURE"],
        "max_tokens": ["OPENAI_MAX_TOKENS", "LLM_MAX_TOKENS"],
        "timeout": ["OPENAI_TIMEOUT", "LLM_TIMEOUT"],
        "base_url": ["OPENAI_BASE_URL", "LLM_BASE_URL"],
    }

    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        初始化配置管理器

        Args:
            config_file: 配置文件路径
        """
        self.logger = logging.getLogger(__name__)
        self.config_file = Path(config_file) if config_file else None
        self._file_config = {}

        # 加载.env文件
        self._load_dotenv()

        # 加载配置文件
        if self.config_file:
            self._load_config_file()

    def _load_dotenv(self) -> None:
        """
        加载.env文件
        """
        if not DOTENV_AVAILABLE:
            self.logger.debug("python-dotenv未安装，跳过.env文件加载")
            return

        # 查找.env文件
        env_paths = [
            Path.cwd() / ".env",
            Path.cwd() / "config" / ".env",
            Path.home() / ".jarvis" / ".env",
        ]

        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path)
                self.logger.debug(f"已加载.env文件: {env_path}")
                break

    def _load_config_file(self) -> None:
        """
        加载配置文件
        """
        if not self.config_file.exists():
            self.logger.warning(f"配置文件不存在: {self.config_file}")
            return

        try:
            if self.config_file.suffix.lower() in [".yml", ".yaml"]:
                self._load_yaml_config()
            else:
                self.logger.warning(f"不支持的配置文件格式: {self.config_file.suffix}")
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")

    def _load_yaml_config(self) -> None:
        """
        加载YAML配置文件
        """
        if not YAML_AVAILABLE:
            raise LLMConfigurationError(
                "PyYAML未安装，无法加载YAML配置文件，请运行: uv add PyYAML"
            )

        with open(self.config_file, "r", encoding="utf-8") as f:
            self._file_config = yaml.safe_load(f) or {}

        self.logger.debug(f"已加载YAML配置文件: {self.config_file}")

    def create_config(
        self, provider: LLMProvider = LLMProvider.OPENAI, **kwargs
    ) -> LLMConfig:
        """
        创建LLM配置

        Args:
            provider: LLM提供商
            **kwargs: 配置参数

        Returns:
            LLM配置对象

        Raises:
            LLMConfigurationError: 配置错误
        """
        # 合并配置（优先级：kwargs > 环境变量 > 配置文件 > 默认值）
        config_data = self._merge_config(provider, **kwargs)

        # 验证配置
        self._validate_config(config_data)

        # 创建配置对象
        return LLMConfig(
            provider=provider,
            api_key=config_data["api_key"],
            model=config_data["model"],
            temperature=config_data["temperature"],
            max_tokens=config_data["max_tokens"],
            timeout=config_data["timeout"],
            base_url=config_data["base_url"],
            extra_params=config_data["extra_params"],
        )

    def _merge_config(self, provider: LLMProvider, **kwargs) -> Dict[str, Any]:
        """
        合并配置数据

        Args:
            provider: LLM提供商
            **kwargs: 配置参数

        Returns:
            合并后的配置数据
        """
        # 从默认配置开始
        config = self.DEFAULT_CONFIG.copy()

        # 应用配置文件中的设置
        provider_name = provider.value.lower()
        if provider_name in self._file_config:
            config.update(self._file_config[provider_name])

        # 应用全局配置文件设置
        if "global" in self._file_config:
            for key, value in self._file_config["global"].items():
                if key not in config or config[key] == self.DEFAULT_CONFIG.get(key):
                    config[key] = value

        # 应用环境变量
        for config_key, env_keys in self.ENV_MAPPING.items():
            for env_key in env_keys:
                env_value = os.getenv(env_key)
                if env_value is not None:
                    config[config_key] = self._convert_env_value(config_key, env_value)
                    break

        # 应用直接传入的参数
        config.update(kwargs)

        return config

    def _convert_env_value(self, key: str, value: str) -> Any:
        """
        转换环境变量值为正确的类型

        Args:
            key: 配置键
            value: 环境变量值

        Returns:
            转换后的值
        """
        if key in ["temperature", "timeout"]:
            try:
                return float(value)
            except ValueError:
                raise LLMConfigurationError(f"环境变量{key}必须是数字: {value}")

        elif key in ["max_tokens"]:
            try:
                return int(value)
            except ValueError:
                raise LLMConfigurationError(f"环境变量{key}必须是整数: {value}")

        else:
            return value

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        验证配置数据

        Args:
            config: 配置数据

        Raises:
            LLMConfigurationError: 配置错误
        """
        # 检查必需的字段
        if not config.get("api_key"):
            raise LLMConfigurationError(
                "API密钥未设置，请设置环境变量或在配置中提供", config_field="api_key"
            )

        if not config.get("model"):
            raise LLMConfigurationError("模型名称未设置", config_field="model")

        # 检查数值范围
        if not (0 <= config.get("temperature", 0.7) <= 2):
            raise LLMConfigurationError(
                "temperature必须在0-2之间", config_field="temperature"
            )

        if config.get("max_tokens", 1000) <= 0:
            raise LLMConfigurationError(
                "max_tokens必须大于0", config_field="max_tokens"
            )

        if config.get("timeout", 30.0) <= 0:
            raise LLMConfigurationError("timeout必须大于0", config_field="timeout")

    def get_available_models(self, provider: LLMProvider) -> List[str]:
        """
        获取可用的模型列表

        Args:
            provider: LLM提供商

        Returns:
            模型列表
        """
        if provider == LLMProvider.OPENAI:
            return [
                "gpt-4",
                "gpt-4-turbo",
                "gpt-4-turbo-preview",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k",
            ]
        else:
            return []

    def create_config_template(self, output_path: Union[str, Path]) -> None:
        """
        创建配置文件模板

        Args:
            output_path: 输出路径
        """
        template = {
            "global": {"timeout": 30.0, "extra_params": {}},
            "openai": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 1000,
                "base_url": None,
            },
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not YAML_AVAILABLE:
            raise LLMConfigurationError(
                "PyYAML未安装，无法创建YAML配置文件，请运行: uv add PyYAML"
            )

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(template, f, default_flow_style=False, allow_unicode=True)

        self.logger.info(f"配置文件模板已创建: {output_path}")


# 全局配置管理器实例
_config_manager = None


def get_config_manager(config_file: Optional[Union[str, Path]] = None) -> ConfigManager:
    """
    获取全局配置管理器实例

    Args:
        config_file: 配置文件路径

    Returns:
        配置管理器实例
    """
    global _config_manager

    if _config_manager is None or config_file is not None:
        _config_manager = ConfigManager(config_file)

    return _config_manager


def create_llm_config(
    provider: LLMProvider = LLMProvider.OPENAI,
    config_file: Optional[Union[str, Path]] = None,
    **kwargs,
) -> LLMConfig:
    """
    便捷函数：创建LLM配置

    Args:
        provider: LLM提供商
        config_file: 配置文件路径
        **kwargs: 配置参数

    Returns:
        LLM配置对象
    """
    manager = get_config_manager(config_file)
    return manager.create_config(provider, **kwargs)
