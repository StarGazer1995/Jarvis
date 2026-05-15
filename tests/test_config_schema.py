"""
Tests for Configuration Schema Validation

Validates that the Pydantic schema layer correctly catches
misconfigured values before they cause runtime errors.
"""

import pytest

from src.core.config.schema import (
    RetryConfigSchema,
    TimeoutConfigSchema,
    ModelConfigSchema,
    GlobalConfigSchema,
    ProviderConfigSchema,
    FeatureConfigSchema,
    LLMConfigSchema,
    validate_config_dict,
    ValidationResult,
)
from src.core.config.loader import (
    RetryConfig,
    TimeoutConfig,
    ModelConfig,
    ProviderConfig,
    GlobalConfig,
    FeatureConfig,
)


# ═══════════════════════════════════════════════════════════════════
# 1. RetryConfig Validation
# ═══════════════════════════════════════════════════════════════════


class TestRetryConfigSchema:
    def test_valid_retry_config(self):
        config = RetryConfigSchema(max_attempts=5, initial_delay=1.0, max_delay=30.0)
        assert config.max_attempts == 5

    def test_max_attempts_must_be_positive(self):
        with pytest.raises(ValueError):
            RetryConfigSchema(max_attempts=0)

    def test_max_attempts_capped_at_100(self):
        with pytest.raises(ValueError):
            RetryConfigSchema(max_attempts=101)

    def test_max_delay_gte_initial_delay(self):
        with pytest.raises(ValueError, match="max_delay.*>=.*initial_delay"):
            RetryConfigSchema(max_attempts=3, initial_delay=10.0, max_delay=1.0)


# ═══════════════════════════════════════════════════════════════════
# 2. TimeoutConfig Validation
# ═══════════════════════════════════════════════════════════════════


class TestTimeoutConfigSchema:
    def test_valid_timeout_config(self):
        config = TimeoutConfigSchema(connect=5.0, read=30.0, total=60.0)
        assert config.connect == 5.0

    def test_timeouts_must_be_positive(self):
        with pytest.raises(ValueError):
            TimeoutConfigSchema(connect=-1.0)

    def test_total_must_be_gte_connect(self):
        with pytest.raises(ValueError, match="total.*>=.*connect"):
            TimeoutConfigSchema(connect=60.0, read=30.0, total=10.0)

    def test_total_must_be_gte_read(self):
        with pytest.raises(ValueError, match="total.*>=.*read"):
            TimeoutConfigSchema(connect=5.0, read=120.0, total=30.0)


# ═══════════════════════════════════════════════════════════════════
# 3. ModelConfig Validation
# ═══════════════════════════════════════════════════════════════════


class TestModelConfigSchema:
    def test_valid_model_config(self):
        config = ModelConfigSchema(max_tokens=2048, temperature=0.5)
        assert config.max_tokens == 2048

    def test_temperature_range(self):
        with pytest.raises(ValueError):
            ModelConfigSchema(temperature=3.0)

    def test_max_tokens_positive(self):
        with pytest.raises(ValueError):
            ModelConfigSchema(max_tokens=0)

    def test_top_p_range(self):
        with pytest.raises(ValueError):
            ModelConfigSchema(top_p=1.5)


# ═══════════════════════════════════════════════════════════════════
# 4. ProviderConfig Validation
# ═══════════════════════════════════════════════════════════════════


class TestProviderConfigSchema:
    def test_valid_provider(self):
        config = ProviderConfigSchema(
            type="openai",
            default_model="gpt-4",
            models={"gpt-4": ModelConfigSchema()},
        )
        assert config.type == "openai"

    def test_invalid_base_url(self):
        with pytest.raises(ValueError, match="Invalid URL"):
            ProviderConfigSchema(base_url="not-a-url")

    def test_valid_base_url(self):
        config = ProviderConfigSchema(base_url="https://api.openai.com/v1")
        assert config.base_url == "https://api.openai.com/v1"

    def test_default_model_not_in_models(self):
        with pytest.raises(ValueError, match="default_model.*not defined"):
            ProviderConfigSchema(
                default_model="nonexistent-model",
                models={"gpt-4": ModelConfigSchema()},
            )


# ═══════════════════════════════════════════════════════════════════
# 5. GlobalConfig Validation
# ═══════════════════════════════════════════════════════════════════


class TestGlobalConfigSchema:
    def test_valid_global_config(self):
        config = GlobalConfigSchema(
            default_provider="openai",
            log_level="INFO",
        )
        assert config.log_level == "INFO"

    def test_invalid_log_level(self):
        with pytest.raises(ValueError, match="Invalid log_level"):
            GlobalConfigSchema(log_level="TRACE")


# ═══════════════════════════════════════════════════════════════════
# 6. LLMConfigSchema — Integration Validation
# ═══════════════════════════════════════════════════════════════════


class TestLLMConfigSchema:
    """End-to-end validation of the full config schema."""

    def test_valid_full_config(self):
        config = LLMConfigSchema(
            global_config={
                "default_provider": "test_provider",
                "log_level": "DEBUG",
            },
            providers={
                "test_provider": {
                    "type": "openai",
                    "enabled": True,
                    "api_key": "${API_KEY}",
                    "default_model": "model-x",
                    "models": {
                        "model-x": {"max_tokens": 4096},
                    },
                },
            },
            environment="development",
        )
        assert config.environment == "development"

    def test_default_provider_not_in_providers(self):
        with pytest.raises(ValueError, match="default_provider.*not configured"):
            LLMConfigSchema(
                global_config={"default_provider": "missing"},
                providers={
                    "other_provider": {
                        "type": "openai",
                        "default_model": "model-x",
                        "models": {"model-x": {}},
                    },
                },
            )

    def test_all_providers_disabled(self):
        with pytest.raises(ValueError, match="At least one provider.*enabled"):
            LLMConfigSchema(
                global_config={"default_provider": "p1"},
                providers={
                    "p1": {
                        "type": "openai",
                        "enabled": False,
                        "default_model": "m1",
                        "models": {"m1": {}},
                    },
                },
            )

    def test_fallback_provider_not_defined(self):
        with pytest.raises(ValueError, match="Fallback provider.*not defined"):
            LLMConfigSchema(
                global_config={
                    "default_provider": "p1",
                    "fallback_providers": ["ghost"],
                },
                providers={
                    "p1": {
                        "type": "openai",
                        "default_model": "m1",
                        "models": {"m1": {}},
                    },
                },
            )

    def test_invalid_environment(self):
        with pytest.raises(ValueError, match="Invalid environment"):
            LLMConfigSchema(
                global_config={"default_provider": "p1"},
                providers={"p1": {"type": "openai", "models": {"m1": {}}}},
                environment="staging",
            )


# ═══════════════════════════════════════════════════════════════════
# 7. validate_config_dict — Integration with real dataclasses
# ═══════════════════════════════════════════════════════════════════


class TestValidateConfigDict:
    """Test the integration function that bridges dataclasses -> Pydantic."""

    def test_valid_config(self):
        provider = ProviderConfig(
            type="openai",
            enabled=True,
            api_key="sk-test",
            default_model="gpt-4",
            models={"gpt-4": ModelConfig(max_tokens=4096)},
        )

        result = validate_config_dict(
            raw_config={
                "global": {
                    "default_provider": "test",
                    "log_level": "INFO",
                },
            },
            provider_configs={"test": provider},
            features={},
            environment="development",
        )

        assert result.is_valid, f"Expected valid, got: {result}"

    def test_invalid_retry_delays(self):
        """Validation should catch max_delay < initial_delay."""
        provider = ProviderConfig(
            type="openai",
            enabled=True,
            api_key="sk-test",
            default_model="gpt-4",
            models={"gpt-4": ModelConfig()},
            retry=RetryConfig(max_attempts=3, initial_delay=60.0, max_delay=5.0),
        )

        result = validate_config_dict(
            raw_config={
                "global": {
                    "default_provider": "test",
                    "log_level": "INFO",
                },
            },
            provider_configs={"test": provider},
            features={},
            environment="development",
        )

        assert not result.is_valid
        assert any("max_delay" in e for e in result.errors)

    def test_warns_missing_api_key(self):
        provider = ProviderConfig(
            type="openai",
            enabled=True,
            api_key=None,
            default_model="gpt-4",
            models={"gpt-4": ModelConfig()},
        )

        result = validate_config_dict(
            raw_config={
                "global": {
                    "default_provider": "test",
                    "log_level": "INFO",
                },
            },
            provider_configs={"test": provider},
            features={},
            environment="development",
        )

        assert result.is_valid  # Missing API key is a warning, not an error
        assert len(result.warnings) > 0
        assert any("API key" in w for w in result.warnings)
