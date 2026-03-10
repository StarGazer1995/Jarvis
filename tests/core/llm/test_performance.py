import pytest
import time
from src.core.llm.client import (
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMProvider,
    CachedLLMClient,
    MonitoredLLMClient,
    BaseLLMClient,
)
from src.core.llm.utils.cache_manager import CacheManager
from src.core.llm.utils.metrics_collector import MetricsCollector, global_metrics


class MockClient(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.call_count = 0

    async def initialize(self) -> bool:
        self._initialized = True
        return True

    async def generate_response(self, messages, **kwargs):
        self.call_count += 1
        return LLMResponse(
            content=f"Response {self.call_count}",
            usage={"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
        )

    async def stream_response(self, messages, **kwargs):
        yield "chunk"


@pytest.fixture
def mock_config():
    return LLMConfig(provider=LLMProvider.OPENAI, model="test")


@pytest.fixture
def mock_client(mock_config):
    return MockClient(mock_config)


class TestCacheManager:
    def test_cache_set_get(self):
        manager = CacheManager(ttl=60)
        messages = [LLMMessage(role="user", content="Hello")]
        response = LLMResponse(content="Hi")

        manager.set(messages, response)
        cached = manager.get(messages)

        assert cached is not None
        assert cached.content == "Hi"

    def test_cache_miss(self):
        manager = CacheManager(ttl=60)
        messages = [LLMMessage(role="user", content="Hello")]

        cached = manager.get(messages)
        assert cached is None

    def test_cache_expiry(self):
        manager = CacheManager(ttl=0.1)
        messages = [LLMMessage(role="user", content="Hello")]
        response = LLMResponse(content="Hi")

        manager.set(messages, response)
        time.sleep(0.2)
        cached = manager.get(messages)

        assert cached is None


class TestCachedLLMClient:
    @pytest.mark.asyncio
    async def test_caching_behavior(self, mock_client):
        cached_client = CachedLLMClient(mock_client)
        messages = [LLMMessage(role="user", content="Hello")]

        # First call - should hit underlying client
        response1 = await cached_client.generate_response(messages)
        assert response1.content == "Response 1"
        assert mock_client.call_count == 1

        # Second call - should hit cache
        response2 = await cached_client.generate_response(messages)
        assert response2.content == "Response 1"
        assert mock_client.call_count == 1

        # Different message - should hit underlying client
        messages2 = [LLMMessage(role="user", content="Hi")]
        response3 = await cached_client.generate_response(messages2)
        assert response3.content == "Response 2"
        assert mock_client.call_count == 2


class TestMetricsCollector:
    def test_metrics_recording(self):
        collector = MetricsCollector()
        collector.record_request(
            success=True,
            latency=0.5,
            tokens={"total_tokens": 100, "prompt_tokens": 40, "completion_tokens": 60},
        )

        metrics = collector.get_metrics()
        assert metrics["total_requests"] == 1
        assert metrics["successful_requests"] == 1
        assert metrics["total_tokens"] == 100
        assert metrics["average_latency"] == 0.5

        collector.record_request(success=False, latency=0.1)
        metrics = collector.get_metrics()
        assert metrics["total_requests"] == 2
        assert metrics["failed_requests"] == 1
        assert metrics["error_rate"] == 0.5


class TestMonitoredLLMClient:
    @pytest.mark.asyncio
    async def test_monitoring_behavior(self, mock_client):
        # Reset global metrics
        global_metrics.reset()

        monitored_client = MonitoredLLMClient(mock_client)
        messages = [LLMMessage(role="user", content="Hello")]

        await monitored_client.generate_response(messages)

        metrics = global_metrics.get_metrics()
        assert metrics["total_requests"] == 1
        assert metrics["successful_requests"] == 1
        assert metrics["total_tokens"] == 10
