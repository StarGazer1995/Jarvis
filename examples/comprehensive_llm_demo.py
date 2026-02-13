#!/usr/bin/env python3
"""
Comprehensive LLM Integration Demo Script

This script demonstrates all aspects of the LLM integration in the Jarvis system:
- LLM client initialization and configuration
- Error handling and retry mechanisms
- Mock and real LLM provider testing
- Streaming responses
- Performance monitoring
"""

import asyncio
import logging
import sys
import os
import time
from typing import List, Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.llm_client import LLMConfig, LLMProvider, LLMMessage, LLMResponse
from core.llm_providers.openai_client import OpenAILLMClient
from core.llm_utils.error_handler import (
    LLMError, LLMAPIError, LLMAuthenticationError,
    LLMTimeoutError, LLMConfigurationError
)
from core.llm_utils.retry_handler import RateLimitHandler


def setup_logging():
    """
    设置日志配置
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


async def test_openai_llm_client():
    """
    测试OpenAI LLM客户端功能（需要API密钥）
    """
    print("\n" + "="*70)
    print("🌐 OpenAI LLM Client Testing")
    print("="*70)
    
    # 检查API密钥
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  未设置OPENAI_API_KEY环境变量，跳过OpenAI测试")
        return
    
    # 创建OpenAI LLM配置
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-3.5-turbo",
        api_key=api_key,
        max_tokens=100,
        temperature=0.7,
        timeout=30.0
    )
    
    client = OpenAILLMClient(config)
    
    try:
        # 初始化客户端
        print("📋 初始化OpenAI LLM客户端...")
        success = await client.initialize()
        print(f"✅ 初始化结果: {success}")
        
        if not success:
            print("❌ OpenAI客户端初始化失败")
            return
        
        # 测试简单对话
        print("\n💬 测试简单对话...")
        messages = [
            LLMMessage(role="user", content="Say hello in Chinese")
        ]
        
        start_time = time.time()
        response = await client.generate_response(messages)
        duration = time.time() - start_time
        
        print(f"🤖 响应: {response.content}")
        print(f"⏱️  耗时: {duration:.2f}秒")
        print(f"🔢 Token使用: {response.usage}")
        
        # 测试流式响应
        print("\n🌊 测试流式响应...")
        stream_messages = [
            LLMMessage(role="user", content="Count from 1 to 5 in English")
        ]
        
        chunks = []
        start_time = time.time()
        
        async for chunk in client.stream_response(stream_messages):
            chunks.append(chunk)
            print(f"📦 收到块: '{chunk}'", end="", flush=True)
        
        duration = time.time() - start_time
        print(f"\n⏱️  流式耗时: {duration:.2f}秒")
        
        # 获取统计信息
        stats = client.get_stats()
        print(f"\n📊 客户端统计: {stats}")
        
    except Exception as e:
        print(f"❌ OpenAI LLM测试失败: {e}")
    finally:
        await client.close()


async def test_error_handling():
    """
    测试错误处理机制
    """
    print("\n" + "="*70)
    print("🛡️  Error Handling Testing")
    print("="*70)
    
    # 测试配置错误
    print("\n📋 测试配置错误...")
    try:
        invalid_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-3.5-turbo",
            api_key="",  # 空API密钥
            max_tokens=-1,  # 无效token数
            temperature=2.0  # 无效温度
        )
        client = OpenAILLMClient(invalid_config)
        client._validate_config()
    except LLMConfigurationError as e:
        print(f"✅ 成功捕获配置错误: {e}")
    
    # 测试认证错误
    print("\n🔐 测试认证错误...")
    try:
        auth_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-3.5-turbo",
            api_key="invalid-key",
            max_tokens=100,
            temperature=0.7
        )
        client = OpenAILLMClient(auth_config)
        await client.initialize()
    except LLMAuthenticationError as e:
        print(f"✅ 成功捕获认证错误: {e}")
    except Exception as e:
        print(f"⚠️  其他错误: {e}")


async def test_rate_limiting():
    """
    测试速率限制处理
    """
    print("\n" + "="*70)
    print("⏱️  Rate Limiting Testing")
    print("="*70)
    
    # 创建速率限制处理器
    logger = logging.getLogger("rate_limit_test")
    rate_handler = RateLimitHandler(logger=logger)
    
    print("📋 测试速率限制等待...")
    
    # 模拟多次快速请求
    for i in range(3):
        start_time = time.time()
        await rate_handler.wait_if_needed()
        wait_time = time.time() - start_time
        print(f"🔄 请求 {i+1}: 等待时间 {wait_time:.2f}秒")
        
        # 模拟请求处理时间
        await asyncio.sleep(0.1)


async def main():
    """
    主函数：运行所有演示测试
    """
    print("🎯 Comprehensive LLM Integration Demo")
    print("="*70)
    
    setup_logging()
    
    try:
        # 运行所有测试
        await test_openai_llm_client()
        await test_error_handling()
        await test_rate_limiting()
        
        print("\n" + "="*70)
        print("✅ 所有LLM集成测试完成！")
        print("="*70)
        print("\n📋 测试总结:")
        print("• OpenAI LLM客户端集成测试")
        print("• 错误处理和恢复机制")
        print("• 速率限制处理")
        print("\n🎯 下一步:")
        print("• 配置真实的LLM API密钥")
        print("• 自定义提示模板")
        print("• 集成到生产环境")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)