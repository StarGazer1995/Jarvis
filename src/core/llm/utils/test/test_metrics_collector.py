"""Metrics collector tests migrated from `src/test`."""


class TestMetricsCollectorEdgeCases:
    """覆盖 metrics_collector.py 中未覆盖的边界"""

    def test_error_rate_zero(self):
        """测试无请求时错误率为0"""
        from src.core.llm.utils.metrics_collector import LLMMetrics

        metrics = LLMMetrics()
        assert metrics.error_rate == 0.0
