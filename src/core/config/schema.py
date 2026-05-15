"""
Configuration Schema Validation

Provides Pydantic-based validation models for the YAML configuration system.
Integrates with ConfigLoader._validate_config() to provide detailed error
messages and catch misconfiguration early at load time.

All models mirror the dataclasses in loader.py but add runtime validation.
"""

import re
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Self


# ── Constants ──────────────────────────────────────────────────────

VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
VALID_PROVIDER_TYPES = {
    "openai",
    "litellm",
    "anthropic",
    "azure_openai",
    "ollama",
    "mock",
}
URL_PATTERN = re.compile(r"^https?://[^\s/$.?#].[^\s]*$", re.IGNORECASE)
TEMP_MIN = 0.0
TEMP_MAX = 2.0


# ── Sub-config models ──────────────────────────────────────────────


class RetryConfigSchema(BaseModel):
    """Validated retry configuration."""

    max_attempts: int = Field(default=3, ge=1, le=100)
    initial_delay: float = Field(default=1.0, ge=0.01, le=300.0)
    max_delay: float = Field(default=60.0, ge=0.1, le=3600.0)
    exponential_base: float = Field(default=2.0, ge=1.0, le=10.0)

    @field_validator("max_delay")
    @classmethod
    def max_delay_gte_initial(cls, v: float, info) -> float:
        """Warn if max_delay is less than initial_delay (partial check via model)."""
        return v

    @model_validator(mode="after")
    def validate_delays(self) -> Self:
        if self.max_delay < self.initial_delay:
            raise ValueError(
                f"max_delay ({self.max_delay}) must be >= initial_delay ({self.initial_delay})"
            )
        return self


class TimeoutConfigSchema(BaseModel):
    """Validated timeout configuration."""

    connect: float = Field(default=30.0, ge=0.1, le=600.0)
    read: float = Field(default=120.0, ge=0.1, le=3600.0)
    total: float = Field(default=180.0, ge=0.1, le=7200.0)

    @model_validator(mode="after")
    def validate_timeouts(self) -> Self:
        if self.total < self.connect:
            raise ValueError(
                f"total timeout ({self.total}) must be >= connect timeout ({self.connect})"
            )
        if self.total < self.read:
            raise ValueError(
                f"total timeout ({self.total}) must be >= read timeout ({self.read})"
            )
        return self


class RateLimitConfigSchema(BaseModel):
    """Validated rate limit configuration."""

    requests_per_minute: int = Field(default=60, ge=1, le=100000)
    tokens_per_minute: int = Field(default=90000, ge=1, le=100000000)


class ModelConfigSchema(BaseModel):
    """Validated per-model configuration."""

    max_tokens: int = Field(default=4096, ge=1, le=1048576)
    temperature: float = Field(default=0.7, ge=TEMP_MIN, le=TEMP_MAX)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    deployment_name: Optional[str] = None
    response_delay: float = Field(default=0.0, ge=0.0, le=60.0)


# ── Provider model ─────────────────────────────────────────────────


class ProviderConfigSchema(BaseModel):
    """Validated LLM provider configuration."""

    type: str = "openai"
    enabled: bool = True
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    default_model: str = "gpt-4"
    models: Dict[str, ModelConfigSchema] = Field(default_factory=dict)
    retry: Optional[RetryConfigSchema] = None
    timeout: Optional[TimeoutConfigSchema] = None
    rate_limit: Optional[RateLimitConfigSchema] = None
    azure_endpoint: Optional[str] = None
    api_version: Optional[str] = None
    organization: Optional[str] = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in VALID_PROVIDER_TYPES and v != v.lower():
            # Allow custom types as long as they're lowercase
            pass
        return v.lower()

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.strip() and not URL_PATTERN.match(v):
            raise ValueError(f"Invalid URL format: '{v}'")
        return v if (v and v.strip()) else None

    @field_validator("azure_endpoint")
    @classmethod
    def validate_azure_endpoint(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.strip() and not URL_PATTERN.match(v):
            raise ValueError(f"Invalid Azure endpoint URL: '{v}'")
        return v if (v and v.strip()) else None

    @model_validator(mode="after")
    def validate_default_model_exists(self) -> Self:
        if self.default_model not in self.models and self.models:
            raise ValueError(
                f"default_model '{self.default_model}' is not defined in models list. "
                f"Available models: {list(self.models.keys())}"
            )
        return self


# ── Top-level models ────────────────────────────────────────────────


class GlobalConfigSchema(BaseModel):
    """Validated global configuration."""

    default_provider: str = "openai"
    fallback_providers: List[str] = Field(default_factory=list)
    retry: RetryConfigSchema = Field(default_factory=RetryConfigSchema)
    timeout: TimeoutConfigSchema = Field(default_factory=TimeoutConfigSchema)
    log_level: str = "INFO"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        upper = v.upper()
        if upper not in VALID_LOG_LEVELS:
            raise ValueError(
                f"Invalid log_level: '{v}'. Must be one of {VALID_LOG_LEVELS}"
            )
        return upper


class FeatureConfigSchema(BaseModel):
    """Validated feature configuration."""

    streaming_enabled: bool = True
    streaming_chunk_size: int = Field(default=1024, ge=256, le=65536)
    context_max_history: int = Field(default=10, ge=0, le=1000)
    context_max_tokens: int = Field(default=8192, ge=128, le=524288)
    cache_enabled: bool = True
    cache_ttl: int = Field(default=3600, ge=0, le=86400 * 30)
    cache_max_size: int = Field(default=1000, ge=1, le=1000000)
    monitoring_enabled: bool = True
    monitoring_metrics_interval: int = Field(default=60, ge=1, le=3600)
    security_input_validation: bool = True
    security_output_filtering: bool = True
    security_max_input_length: int = Field(default=10000, ge=1, le=1000000)


class LLMConfigSchema(BaseModel):
    """Complete validated LLM configuration."""

    global_config: GlobalConfigSchema
    providers: Dict[str, ProviderConfigSchema]
    features: FeatureConfigSchema = Field(default_factory=FeatureConfigSchema)
    environment: str = "development"

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        valid = {"development", "testing", "production"}
        if v.lower() not in valid:
            raise ValueError(f"Invalid environment: '{v}'. Must be one of {valid}")
        return v.lower()

    @model_validator(mode="after")
    def validate_default_provider_exists(self) -> Self:
        default = self.global_config.default_provider
        if default not in self.providers:
            raise ValueError(
                f"default_provider '{default}' is not configured in providers list. "
                f"Available providers: {list(self.providers.keys())}"
            )
        return self

    @model_validator(mode="after")
    def validate_at_least_one_provider_enabled(self) -> Self:
        enabled = [n for n, p in self.providers.items() if p.enabled]
        if not enabled:
            raise ValueError(
                "At least one provider must be enabled. "
                "All providers are currently disabled."
            )
        return self

    @model_validator(mode="after")
    def validate_fallback_providers_exist(self) -> Self:
        fallbacks = self.global_config.fallback_providers
        for fb in fallbacks:
            if fb not in self.providers:
                raise ValueError(
                    f"Fallback provider '{fb}' is not defined in providers list. "
                    f"Available providers: {list(self.providers.keys())}"
                )
        return self


# ── Validation result ──────────────────────────────────────────────


class ValidationResult:
    """Result of a configuration validation, with warnings and errors."""

    def __init__(self, is_valid: bool = True):
        self.is_valid = is_valid
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def add_error(self, message: str) -> None:
        self.is_valid = False
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def __bool__(self) -> bool:
        return self.is_valid

    def __str__(self) -> str:
        parts = []
        if self.errors:
            parts.append(
                f"Errors ({len(self.errors)}):\n"
                + "\n".join(f"  - {e}" for e in self.errors)
            )
        if self.warnings:
            parts.append(
                f"Warnings ({len(self.warnings)}):\n"
                + "\n".join(f"  - {w}" for w in self.warnings)
            )
        return "\n".join(parts) if parts else "Valid"


def validate_config_dict(
    raw_config: dict,
    provider_configs: dict,
    features: dict,
    environment: str = "development",
) -> ValidationResult:
    """Validate a parsed configuration dictionary using Pydantic models.

    This function can be called from ConfigLoader._validate_config() to add
    comprehensive schema validation on top of the existing business-rule checks.

    Args:
        raw_config: The raw parsed configuration dict (used for cross-field checks).
        provider_configs: Parsed provider configurations (dict of ProviderConfig).
        features: Parsed feature configuration dict.
        environment: Environment name.

    Returns:
        A ValidationResult with errors and warnings.
    """
    result = ValidationResult()

    # Build the Pydantic models from the parsed config
    try:
        # Convert dataclass objects to dicts for Pydantic
        providers_dict = {}
        for name, pc in provider_configs.items():
            pd = {
                "type": pc.type,
                "enabled": pc.enabled,
                "api_key": pc.api_key,
                "base_url": pc.base_url,
                "default_model": pc.default_model,
                "models": {
                    mname: {
                        "max_tokens": mcfg.max_tokens,
                        "temperature": mcfg.temperature,
                        "top_p": mcfg.top_p,
                        "frequency_penalty": mcfg.frequency_penalty,
                        "presence_penalty": mcfg.presence_penalty,
                        "deployment_name": mcfg.deployment_name,
                        "response_delay": mcfg.response_delay,
                    }
                    for mname, mcfg in pc.models.items()
                },
                "azure_endpoint": pc.azure_endpoint,
                "api_version": pc.api_version,
                "organization": pc.organization,
            }
            if pc.retry:
                pd["retry"] = {
                    "max_attempts": pc.retry.max_attempts,
                    "initial_delay": pc.retry.initial_delay,
                    "max_delay": pc.retry.max_delay,
                    "exponential_base": pc.retry.exponential_base,
                }
            if pc.timeout:
                pd["timeout"] = {
                    "connect": pc.timeout.connect,
                    "read": pc.timeout.read,
                    "total": pc.timeout.total,
                }
            if pc.rate_limit:
                pd["rate_limit"] = {
                    "requests_per_minute": pc.rate_limit.requests_per_minute,
                    "tokens_per_minute": pc.rate_limit.tokens_per_minute,
                }
            providers_dict[name] = pd

        # Build global config dict from raw_config
        global_raw = raw_config.get("global", {})
        global_dict = {
            "default_provider": global_raw.get("default_provider", "openai"),
            "fallback_providers": global_raw.get("fallback_providers", []),
            "retry": global_raw.get("retry", {}),
            "timeout": global_raw.get("timeout", {}),
            "log_level": global_raw.get("log_level", "INFO"),
        }

        # Build features dict
        features_dict = {
            "streaming_enabled": features.get("streaming", {}).get("enabled", True),
            "streaming_chunk_size": features.get("streaming", {}).get(
                "chunk_size", 1024
            ),
            "context_max_history": features.get("context", {}).get("max_history", 10),
            "context_max_tokens": features.get("context", {}).get("max_tokens", 8192),
            "cache_enabled": features.get("cache", {}).get("enabled", True),
            "cache_ttl": features.get("cache", {}).get("ttl", 3600),
            "cache_max_size": features.get("cache", {}).get("max_size", 1000),
            "monitoring_enabled": features.get("monitoring", {}).get("enabled", True),
            "monitoring_metrics_interval": features.get("monitoring", {}).get(
                "metrics_interval", 60
            ),
            "security_input_validation": features.get("security", {}).get(
                "input_validation", True
            ),
            "security_output_filtering": features.get("security", {}).get(
                "output_filtering", True
            ),
            "security_max_input_length": features.get("security", {}).get(
                "max_input_length", 10000
            ),
        }

        # Validate with Pydantic
        LLMConfigSchema(
            global_config=global_dict,
            providers=providers_dict,
            features=features_dict,
            environment=environment,
        )

    except ValueError as e:
        result.add_error(str(e))
        return result

    # Additional cross-field checks that Pydantic can't easily express
    _check_api_key_warnings(result, providers_dict)

    return result


def _check_api_key_warnings(
    result: ValidationResult,
    providers_dict: dict,
) -> None:
    """Check for missing API keys on non-mock, non-local providers."""
    for name, pd in providers_dict.items():
        if not pd.get("enabled", True):
            continue
        if name == "mock":
            continue
        if name == "ollama":
            continue  # Local models don't need API keys
        if not pd.get("api_key"):
            result.add_warning(
                f"Provider '{name}' has no API key configured. "
                "Set via environment variable or api_key field."
            )
