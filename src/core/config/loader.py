"""
YAML配置文件加载器和验证器

该模块提供了加载、解析和验证LLM配置文件的功能，支持：
- YAML配置文件解析
- 环境变量替换
- 配置验证和默认值处理
- 环境特定配置覆盖
- 配置热重载
"""

import os
import re
import yaml
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from ..common.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class Environment(Enum):
    """支持的环境类型"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0


@dataclass
class TimeoutConfig:
    """超时配置"""
    connect: float = 30.0
    read: float = 120.0
    total: float = 180.0


@dataclass
class RateLimitConfig:
    """速率限制配置"""
    requests_per_minute: int = 60
    tokens_per_minute: int = 90000


@dataclass
class ModelConfig:
    """模型配置"""
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    deployment_name: Optional[str] = None  # Azure OpenAI专用
    response_delay: float = 0.0  # Mock专用


@dataclass
class ProviderConfig:
    """LLM提供商配置"""
    type: str = "openai"  # 默认为openai，用于区分提供商类型
    enabled: bool = True
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    default_model: str = "gpt-4"
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    retry: Optional[RetryConfig] = None
    timeout: Optional[TimeoutConfig] = None
    rate_limit: Optional[RateLimitConfig] = None
    
    # Azure OpenAI特定配置
    azure_endpoint: Optional[str] = None
    api_version: Optional[str] = None
    organization: Optional[str] = None


@dataclass
class GlobalConfig:
    """全局配置"""
    default_provider: str = "openai"
    fallback_providers: List[str] = field(default_factory=list)
    retry: RetryConfig = field(default_factory=RetryConfig)
    timeout: TimeoutConfig = field(default_factory=TimeoutConfig)
    log_level: str = "INFO"


@dataclass
class FeatureConfig:
    """功能特性配置"""
    streaming_enabled: bool = True
    streaming_chunk_size: int = 1024
    context_max_history: int = 10
    context_max_tokens: int = 8192
    cache_enabled: bool = True
    cache_ttl: int = 3600
    cache_max_size: int = 1000
    monitoring_enabled: bool = True
    monitoring_metrics_interval: int = 60
    security_input_validation: bool = True
    security_output_filtering: bool = True
    security_max_input_length: int = 10000


@dataclass
class LLMConfig:
    """完整的LLM配置"""
    global_config: GlobalConfig
    providers: Dict[str, ProviderConfig]
    features: FeatureConfig
    environment: Environment = Environment.DEVELOPMENT


class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径，默认为 config/llm_config.yaml
        """
        if config_path is None:
            # 默认配置文件路径
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "config" / "llm_config.yaml"
        
        self.config_path = Path(config_path)
        self._config_cache: Optional[Dict[str, Any]] = None
        self._last_modified: Optional[float] = None
        
        logger.info(f"配置加载器初始化，配置文件路径: {self.config_path}")
    
    def load_config(self, environment: Optional[str] = None, 
                   reload: bool = False) -> LLMConfig:
        """
        加载配置文件
        
        Args:
            environment: 环境名称 (development/testing/production)
            reload: 是否强制重新加载配置
            
        Returns:
            LLMConfig: 解析后的配置对象
            
        Raises:
            ConfigError: 配置加载或验证失败
        """
        try:
            # 检查是否需要重新加载
            if reload or self._should_reload():
                self._load_raw_config()
            
            if self._config_cache is None:
                raise ConfigurationError("配置文件加载失败")
            
            # 确定环境
            env = self._determine_environment(environment)
            logger.info(f"使用环境配置: {env.value}")
            
            # 解析配置
            config = self._parse_config(self._config_cache, env)
            
            # 验证配置
            self._validate_config(config)
            
            logger.info("配置加载完成")
            return config
            
        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            raise ConfigurationError(f"配置加载失败: {e}") from e
    
    def _should_reload(self) -> bool:
        """检查是否需要重新加载配置文件"""
        if not self.config_path.exists():
            return False
        
        current_modified = self.config_path.stat().st_mtime
        if self._last_modified is None or current_modified > self._last_modified:
            return True
        
        return False
    
    def _load_raw_config(self) -> None:
        """加载原始配置文件"""
        if not self.config_path.exists():
            raise ConfigurationError(f"配置文件不存在: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 替换环境变量
            content = self._substitute_env_vars(content)
            
            # 解析YAML
            self._config_cache = yaml.safe_load(content)
            self._last_modified = self.config_path.stat().st_mtime
            
            logger.debug("原始配置文件加载完成")
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"YAML解析失败: {e}") from e
        except Exception as e:
            raise ConfigurationError(f"配置文件读取失败: {e}") from e
    
    def _substitute_env_vars(self, content: str) -> str:
        """
        替换配置文件中的环境变量
        
        支持格式：
        - ${VAR_NAME}
        - ${VAR_NAME:default_value}
        """
        def replace_var(match):
            var_expr = match.group(1)
            if ':' in var_expr:
                var_name, default_value = var_expr.split(':', 1)
            else:
                var_name, default_value = var_expr, ''
            
            value = os.getenv(var_name, default_value)
            logger.debug(f"环境变量替换: {var_name} -> {'***' if 'key' in var_name.lower() else value}")
            return value
        
        # 匹配 ${VAR_NAME} 或 ${VAR_NAME:default}
        pattern = r'\$\{([^}]+)\}'
        return re.sub(pattern, replace_var, content)
    
    def _determine_environment(self, environment: Optional[str]) -> Environment:
        """确定当前环境"""
        if environment:
            try:
                return Environment(environment.lower())
            except ValueError:
                logger.warning(f"无效的环境名称: {environment}，使用默认环境")
        
        # 从环境变量获取
        env_name = os.getenv('JARVIS_ENV', 'development').lower()
        try:
            return Environment(env_name)
        except ValueError:
            logger.warning(f"无效的环境变量值: {env_name}，使用development")
            return Environment.DEVELOPMENT
    
    def _parse_config(self, raw_config: Dict[str, Any], 
                     environment: Environment) -> LLMConfig:
        """解析配置对象"""
        # 解析全局配置
        global_raw = raw_config.get('global', {})
        global_config = self._parse_global_config(global_raw)
        
        # 解析提供商配置
        providers_raw = raw_config.get('providers', {})
        providers = self._parse_providers_config(providers_raw, global_config)
        
        # 解析功能配置
        features_raw = raw_config.get('features', {})
        features = self._parse_features_config(features_raw)
        
        # 应用环境特定配置
        env_config = raw_config.get('environments', {}).get(environment.value, {})
        if env_config:
            global_config, providers = self._apply_environment_config(
                global_config, providers, env_config
            )
        
        return LLMConfig(
            global_config=global_config,
            providers=providers,
            features=features,
            environment=environment
        )
    
    def _parse_global_config(self, global_raw: Dict[str, Any]) -> GlobalConfig:
        """解析全局配置"""
        retry_raw = global_raw.get('retry', {})
        retry = RetryConfig(
            max_attempts=retry_raw.get('max_attempts', 3),
            initial_delay=retry_raw.get('initial_delay', 1.0),
            max_delay=retry_raw.get('max_delay', 60.0),
            exponential_base=retry_raw.get('exponential_base', 2.0)
        )
        
        timeout_raw = global_raw.get('timeout', {})
        timeout = TimeoutConfig(
            connect=timeout_raw.get('connect', 30.0),
            read=timeout_raw.get('read', 120.0),
            total=timeout_raw.get('total', 180.0)
        )
        
        return GlobalConfig(
            default_provider=global_raw.get('default_provider', 'openai'),
            fallback_providers=global_raw.get('fallback_providers', []),
            retry=retry,
            timeout=timeout,
            log_level=global_raw.get('log_level', 'INFO')
        )
    
    def _parse_providers_config(self, providers_raw: Dict[str, Any], 
                               global_config: GlobalConfig) -> Dict[str, ProviderConfig]:
        """解析提供商配置"""
        providers = {}
        
        for provider_name, provider_raw in providers_raw.items():
            # 解析重试配置
            retry_raw = provider_raw.get('retry', {})
            retry = RetryConfig(
                max_attempts=retry_raw.get('max_attempts', global_config.retry.max_attempts),
                initial_delay=retry_raw.get('initial_delay', global_config.retry.initial_delay),
                max_delay=retry_raw.get('max_delay', global_config.retry.max_delay),
                exponential_base=retry_raw.get('exponential_base', global_config.retry.exponential_base)
            )
            
            # 解析超时配置
            timeout_raw = provider_raw.get('timeout', {})
            timeout = TimeoutConfig(
                connect=timeout_raw.get('connect', global_config.timeout.connect),
                read=timeout_raw.get('read', global_config.timeout.read),
                total=timeout_raw.get('total', global_config.timeout.total)
            )
            
            # 解析速率限制配置
            rate_limit = None
            if 'rate_limit' in provider_raw:
                rate_limit_raw = provider_raw['rate_limit']
                rate_limit = RateLimitConfig(
                    requests_per_minute=rate_limit_raw.get('requests_per_minute', 60),
                    tokens_per_minute=rate_limit_raw.get('tokens_per_minute', 90000)
                )
            
            # 解析模型配置
            models = {}
            models_raw = provider_raw.get('models', {})
            for model_name, model_raw in models_raw.items():
                models[model_name] = ModelConfig(
                    max_tokens=model_raw.get('max_tokens', 4096),
                    temperature=model_raw.get('temperature', 0.7),
                    top_p=model_raw.get('top_p', 1.0),
                    frequency_penalty=model_raw.get('frequency_penalty', 0.0),
                    presence_penalty=model_raw.get('presence_penalty', 0.0),
                    deployment_name=model_raw.get('deployment_name'),
                    response_delay=model_raw.get('response_delay', 0.0)
                )
            
            providers[provider_name] = ProviderConfig(
                type=provider_raw.get('type', provider_name),
                enabled=provider_raw.get('enabled', True),
                api_key=provider_raw.get('api_key'),
                base_url=provider_raw.get('base_url'),
                default_model=provider_raw.get('default_model', 'gpt-4'),
                models=models,
                retry=retry,
                timeout=timeout,
                rate_limit=rate_limit,
                azure_endpoint=provider_raw.get('azure_endpoint'),
                api_version=provider_raw.get('api_version'),
                organization=provider_raw.get('organization')
            )
        
        return providers
    
    def _parse_features_config(self, features_raw: Dict[str, Any]) -> FeatureConfig:
        """解析功能配置"""
        streaming = features_raw.get('streaming', {})
        context = features_raw.get('context', {})
        cache = features_raw.get('cache', {})
        monitoring = features_raw.get('monitoring', {})
        security = features_raw.get('security', {})
        
        return FeatureConfig(
            streaming_enabled=streaming.get('enabled', True),
            streaming_chunk_size=streaming.get('chunk_size', 1024),
            context_max_history=context.get('max_history', 10),
            context_max_tokens=context.get('max_tokens', 8192),
            cache_enabled=cache.get('enabled', True),
            cache_ttl=cache.get('ttl', 3600),
            cache_max_size=cache.get('max_size', 1000),
            monitoring_enabled=monitoring.get('enabled', True),
            monitoring_metrics_interval=monitoring.get('metrics_interval', 60),
            security_input_validation=security.get('input_validation', True),
            security_output_filtering=security.get('output_filtering', True),
            security_max_input_length=security.get('max_input_length', 10000)
        )
    
    def _apply_environment_config(self, global_config: GlobalConfig, 
                                 providers: Dict[str, ProviderConfig],
                                 env_config: Dict[str, Any]) -> tuple:
        """应用环境特定配置"""
        # 更新全局配置
        if 'global' in env_config:
            env_global = env_config['global']
            if 'default_provider' in env_global:
                global_config.default_provider = env_global['default_provider']
            if 'log_level' in env_global:
                global_config.log_level = env_global['log_level']
        
        # 更新提供商配置
        if 'providers' in env_config:
            for provider_name, provider_overrides in env_config['providers'].items():
                if provider_name in providers:
                    provider = providers[provider_name]
                    if 'enabled' in provider_overrides:
                        provider.enabled = provider_overrides['enabled']
                    if 'models' in provider_overrides:
                        for model_name, model_overrides in provider_overrides['models'].items():
                            if model_name in provider.models:
                                model = provider.models[model_name]
                                for key, value in model_overrides.items():
                                    setattr(model, key, value)
        
        return global_config, providers
    
    def _validate_config(self, config: LLMConfig) -> None:
        """验证配置"""
        # 验证默认提供商存在且启用
        default_provider = config.global_config.default_provider
        if default_provider not in config.providers:
            raise ConfigurationError(f"默认提供商 '{default_provider}' 未配置")
        
        if not config.providers[default_provider].enabled:
            raise ConfigurationError(f"默认提供商 '{default_provider}' 未启用")
        
        # 验证每个启用的提供商都有有效配置
        for provider_name, provider_config in config.providers.items():
            if not provider_config.enabled:
                continue
            
            # 验证默认模型存在
            if provider_config.default_model not in provider_config.models:
                raise ConfigurationError(
                    f"提供商 '{provider_name}' 的默认模型 '{provider_config.default_model}' 未配置"
                )
            
            # 验证API密钥（除了mock提供商）
            if provider_name != 'mock' and not provider_config.api_key:
                logger.warning(f"提供商 '{provider_name}' 未配置API密钥")
        
        logger.info("配置验证通过")


# 全局配置加载器实例
_config_loader: Optional[ConfigLoader] = None


def get_config_loader(config_path: Optional[Union[str, Path]] = None) -> ConfigLoader:
    """获取全局配置加载器实例"""
    global _config_loader
    if _config_loader is None or config_path is not None:
        _config_loader = ConfigLoader(config_path)
    return _config_loader


def get_config_path(config_name: str = "llm_config.yaml") -> Path:
    """
    获取配置文件的默认路径
    
    Args:
        config_name: 配置文件名称
    
    Returns:
        Path: 配置文件路径
    """
    # 获取项目根目录
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent  # 从src/core/config/loader.py回到项目根目录
    
    # 配置文件路径
    config_path = project_root / "config" / config_name
    
    return config_path


def load_llm_config(environment: Optional[str] = None, 
                    config_path: Optional[Union[str, Path]] = None,
                    reload: bool = False) -> LLMConfig:
    """
    加载LLM配置的便捷函数
    
    Args:
        environment: 环境名称
        config_path: 配置文件路径
        reload: 是否强制重新加载
    
    Returns:
        LLMConfig: 解析后的配置对象
    """
    loader = get_config_loader(config_path)
    return loader.load_config(environment, reload)