"""
指标收集模块

收集和报告LLM性能指标。
"""

import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager


@dataclass
class LLMMetrics:
    """LLM性能指标"""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency: float = 0.0

    @property
    def average_latency(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency / self.successful_requests

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.metrics = LLMMetrics()
        self.logger = logging.getLogger(__name__)

    def record_request(
        self, success: bool, latency: float, tokens: Dict[str, int] = None
    ):
        """
        记录请求指标

        Args:
            success: 是否成功
            latency: 耗时（秒）
            tokens: token使用情况
        """
        self.metrics.total_requests += 1

        if success:
            self.metrics.successful_requests += 1
            self.metrics.total_latency += latency

            if tokens:
                self.metrics.total_tokens += tokens.get("total_tokens", 0)
                self.metrics.total_prompt_tokens += tokens.get("prompt_tokens", 0)
                self.metrics.total_completion_tokens += tokens.get(
                    "completion_tokens", 0
                )
        else:
            self.metrics.failed_requests += 1

    def get_metrics(self) -> Dict[str, Any]:
        """获取指标汇总"""
        return {
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "error_rate": self.metrics.error_rate,
            "average_latency": self.metrics.average_latency,
            "total_tokens": self.metrics.total_tokens,
            "prompt_tokens": self.metrics.total_prompt_tokens,
            "completion_tokens": self.metrics.total_completion_tokens,
        }

    def reset(self):
        """重置指标"""
        self.metrics = LLMMetrics()


# 全局指标收集器
global_metrics = MetricsCollector()
