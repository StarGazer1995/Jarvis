"""Cache manager tests migrated from `src/test`."""


class TestCacheManagerEdgeCases:
    """覆盖 cache_manager.py 中未覆盖的边界"""

    def test_cache_expiry(self):
        """测试缓存过期"""
        from src.core.llm.types import LLMMessage
        from src.core.llm.utils.cache_manager import CacheManager

        cache = CacheManager(ttl=0)
        msgs = [LLMMessage(role="user", content="test")]
        cache.set(msgs, "response", model="gpt-4")
        result = cache.get(msgs, model="gpt-4")
        assert result is None

    def test_cache_set_with_different_args(self):
        """测试使用不同参数缓存"""
        from src.core.llm.types import LLMMessage
        from src.core.llm.utils.cache_manager import CacheManager

        cache = CacheManager(ttl=3600)
        msgs = [LLMMessage(role="user", content="hello")]
        cache.set(msgs, "response")
        result = cache.get(msgs)
        assert result is not None
