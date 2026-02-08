"""
缓存管理模块

提供LLM响应缓存机制。
"""

import hashlib
import json
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import asdict

from ..types import LLMResponse, LLMMessage

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, ttl: int = 3600, max_size: int = 1000):
        """
        初始化缓存管理器
        
        Args:
            ttl: 缓存有效期（秒）
            max_size: 最大缓存条目数
        """
        self.ttl = ttl
        self.max_size = max_size
        self._cache: Dict[str, Tuple[LLMResponse, float]] = {}
        self.logger = logging.getLogger(__name__)
        
    def get(self, messages: List[LLMMessage], **kwargs) -> Optional[LLMResponse]:
        """
        获取缓存的响应
        
        Args:
            messages: 消息列表
            **kwargs: 额外参数
            
        Returns:
            缓存的响应或None
        """
        key = self._generate_key(messages, **kwargs)
        
        if key in self._cache:
            response, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl:
                self.logger.debug(f"缓存命中: {key[:8]}")
                return response
            else:
                # 过期删除
                del self._cache[key]
                
        return None
        
    def set(self, messages: List[LLMMessage], response: LLMResponse, **kwargs) -> None:
        """
        设置缓存
        
        Args:
            messages: 消息列表
            response: LLM响应
            **kwargs: 额外参数
        """
        if len(self._cache) >= self.max_size:
            # 简单的清理策略：删除最早的
            # 实际生产中可能需要LRU
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
            
        key = self._generate_key(messages, **kwargs)
        self._cache[key] = (response, time.time())
        self.logger.debug(f"缓存设置: {key[:8]}")
        
    def _generate_key(self, messages: List[LLMMessage], **kwargs) -> str:
        """生成缓存键"""
        # 序列化消息
        msgs_data = [{"role": m.role, "content": m.content} for m in messages]
        msgs_str = json.dumps(msgs_data, sort_keys=True)
        
        # 序列化参数 (只取影响生成的参数)
        params = {k: v for k, v in kwargs.items() if k in ['temperature', 'max_tokens', 'model', 'top_p']}
        params_str = json.dumps(params, sort_keys=True)
        
        content = f"{msgs_str}|{params_str}"
        return hashlib.md5(content.encode()).hexdigest()
        
    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()
        
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl": self.ttl
        }
