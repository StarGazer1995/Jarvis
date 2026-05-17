"""Additional LLM factory tests migrated from legacy `src/test` files."""

from unittest.mock import MagicMock

import pytest


class TestFactoryDetailedCoverage:
    """LLM Factory 补充测试"""

    def test_provider_info_str(self):
        from src.core.config.loader import ProviderConfig
        from src.core.llm.factory import ProviderInfo, ProviderType

        config = ProviderConfig(type="mock", enabled=True)
        info = ProviderInfo(
            name="test",
            type=ProviderType.OPENAI,
            client_class=MagicMock,
            enabled=True,
            config=config,
        )
        s = str(info)
        assert "test" in s or "ProviderInfo" in s

    def test_registry_not_registered(self):
        from src.core.llm.factory import LLMProviderRegistry

        registry = LLMProviderRegistry()
        assert registry.is_provider_registered("nonexistent") is False

    def test_registry_list_providers(self):
        from src.core.llm.factory import LLMProviderRegistry

        registry = LLMProviderRegistry()
        providers = registry.list_providers()
        assert isinstance(providers, list)


class TestFactoryMoreCoverage:
    """LLM Factory 更多测试"""

    def test_registry_register_duplicate(self):
        from src.core.llm.client import BaseLLMClient
        from src.core.llm.factory import LLMProviderRegistry
        from src.core.llm.types import LLMConfig, LLMProvider

        class MockClient(BaseLLMClient):
            async def initialize(self):
                return True

            async def generate_response(self, messages, **kwargs):
                pass

            async def stream_response(self, messages, **kwargs):
                yield ""

        config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4")
        MockClient(config)

        registry = LLMProviderRegistry()
        registry.register_provider("mock", MockClient)
        providers_before = len(registry.list_providers())
        # Register same name again should not increase count
        registry.register_provider("mock", MockClient)
        assert len(registry.list_providers()) == providers_before

    def test_list_providers_returns_correctly(self):
        from src.core.llm.client import BaseLLMClient
        from src.core.llm.factory import LLMProviderRegistry

        class MockClient(BaseLLMClient):
            async def initialize(self):
                return True

            async def generate_response(self, messages, **kwargs):
                pass

            async def stream_response(self, messages, **kwargs):
                yield ""

        registry = LLMProviderRegistry()
        registry.register_provider("mock", MockClient)
        providers = registry.list_providers()
        assert "mock" in providers
        assert isinstance(providers, list)


class TestFactoryRemainingCoverage:
    """Factory 剩余边界测试"""

    def test_get_provider_class_nonexistent(self):
        from src.core.llm.factory import LLMProviderRegistry

        registry = LLMProviderRegistry()
        result = registry.get_provider_class("nonexistent")
        assert result is None

    def test_unregister_provider(self):
        from src.core.llm.client import BaseLLMClient
        from src.core.llm.factory import LLMProviderRegistry

        class MockClient(BaseLLMClient):
            async def initialize(self):
                return True

            async def generate_response(self, m, **k):
                pass

            async def stream_response(self, m, **k):
                yield ""

        registry = LLMProviderRegistry()
        registry.register_provider("test_client", MockClient)
        assert registry.is_provider_registered("test_client") is True
        registry.unregister_provider("test_client")
        assert registry.is_provider_registered("test_client") is False


class TestFactoryLastBatch:
    """factory 剩余边界测试"""

    def test_register_provider_invalid_class(self):
        from src.core.llm.factory import LLMProviderRegistry

        registry = LLMProviderRegistry()
        with pytest.raises(ValueError, match="必须继承自BaseLLMClient"):
            registry.register_provider("bad", str)

    def test_unregister_nonexistent(self):
        from src.core.llm.factory import LLMProviderRegistry

        registry = LLMProviderRegistry()
        # Should not raise
        registry.unregister_provider("nonexistent")


class TestFactoryFinalPush:
    """Migrated methods from TestFinalPush."""

    def test_factory_get_provider_class(self):
        from src.core.llm.client import BaseLLMClient
        from src.core.llm.factory import LLMProviderRegistry

        class MockClient(BaseLLMClient):
            async def initialize(self):
                return True

            async def generate_response(self, m, **k):
                pass

            async def stream_response(self, m, **k):
                yield ""

        registry = LLMProviderRegistry()
        registry.register_provider("mock", MockClient)
        cls = registry.get_provider_class("mock")
        assert cls is MockClient
        assert registry.get_provider_class("nonexistent") is None

    def test_factory_unregister(self):
        from src.core.llm.factory import LLMProviderRegistry

        registry = LLMProviderRegistry()
        registry.unregister_provider("nonexistent")
