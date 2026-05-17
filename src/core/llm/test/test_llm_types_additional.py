"""Additional LLM type tests migrated from legacy `src/test` files."""


class TestLLMTypesFinalPush:
    """Migrated methods from TestFinalPush."""

    def test_litellm_config(self):
        from src.core.llm.types import LLMConfig, LLMProvider

        cfg = LLMConfig(provider=LLMProvider.LITELLM, model="gpt-4", api_key="k")
        assert cfg.provider == LLMProvider.LITELLM
        assert cfg.model == "gpt-4"


class TestLiteLLMFinal:
    """LiteLLM 最后覆盖"""

    def test_litellm_config_creation(self):
        from src.core.llm.types import LLMConfig, LLMProvider

        cfg = LLMConfig(provider=LLMProvider.LITELLM, model="gpt-4")
        assert cfg.provider == LLMProvider.LITELLM
        assert cfg.model == "gpt-4"
        assert cfg.provider_name is None
