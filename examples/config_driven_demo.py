#!/usr/bin/env python3
"""
配置驱动的LLM演示脚本

该脚本演示如何使用配置文件来驱动LLM提供商的选择和使用，包括：
- 加载不同的配置文件
- 使用不同的提供商
- 动态切换提供商
- 配置验证和错误处理
- 多环境支持
"""

import os
import sys
import asyncio
import argparse
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.llm_factory import LLMProviderFactory, create_llm_client
from src.core.config_loader import load_llm_config
from src.core.llm_client import LLMMessage, LLMProvider
from src.core.exceptions import LLMError, ConfigurationError


def setup_logging(level: str = "INFO") -> None:
    """
    设置日志配置
    
    Args:
        level: 日志级别
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def print_section(title: str) -> None:
    """
    打印章节标题
    
    Args:
        title: 标题文本
    """
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_subsection(title: str) -> None:
    """
    打印子章节标题
    
    Args:
        title: 标题文本
    """
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")


async def demo_basic_usage(config_path: Optional[str] = None) -> None:
    """
    演示基本使用方法
    
    Args:
        config_path: 配置文件路径
    """
    print_section("基本使用演示")
    
    try:
        # 创建LLM客户端（使用默认提供商）
        print("🔧 创建默认LLM客户端...")
        client = create_llm_client(config_path=config_path)
        
        # 发送简单消息
        print("💬 发送测试消息...")
        messages = [
            LLMMessage(role="user", content="你好，请简单介绍一下你自己。")
        ]
        
        response = await client.chat_completion(messages)
        print(f"🤖 回复: {response.content}")
        print(f"📊 使用模型: {response.model}")
        print(f"⏱️  响应时间: {response.response_time:.2f}秒")
        print(f"🔢 Token使用: {response.usage}")
        
    except Exception as e:
        print(f"❌ 基本使用演示失败: {e}")


async def demo_provider_switching(config_path: Optional[str] = None) -> None:
    """
    演示提供商切换
    
    Args:
        config_path: 配置文件路径
    """
    print_section("提供商切换演示")
    
    try:
        # 创建工厂实例
        factory = LLMProviderFactory(config_path)
        
        # 获取可用的提供商列表
        enabled_providers = factory.list_enabled_providers()
        print(f"🔍 可用提供商: {enabled_providers}")
        
        # 测试每个提供商
        test_message = [
            LLMMessage(role="user", content="请用一句话介绍人工智能。")
        ]
        
        for provider_name in enabled_providers[:3]:  # 限制测试前3个提供商
            print_subsection(f"测试提供商: {provider_name}")
            
            try:
                # 创建特定提供商的客户端
                client = factory.get_provider(provider_name)
                
                # 发送消息
                response = await client.chat_completion(test_message)
                print(f"🤖 {provider_name} 回复: {response.content}")
                print(f"📊 模型: {response.model}")
                print(f"⏱️  响应时间: {response.response_time:.2f}秒")
                
            except Exception as e:
                print(f"❌ 提供商 {provider_name} 测试失败: {e}")
        
        # 关闭工厂
        factory.close()
        
    except Exception as e:
        print(f"❌ 提供商切换演示失败: {e}")


async def demo_environment_configs() -> None:
    """演示不同环境配置"""
    print_section("环境配置演示")
    
    # 测试不同环境的配置文件
    environments = [
        ("minimal", "config/examples/minimal.yaml"),
        ("development", "config/examples/development.yaml"),
        ("testing", "config/examples/testing.yaml")
    ]
    
    for env_name, config_file in environments:
        print_subsection(f"环境: {env_name}")
        
        if not os.path.exists(config_file):
            print(f"⚠️  配置文件不存在: {config_file}")
            continue
        
        try:
            # 加载配置
            config = load_llm_config(config_path=config_file)
            print(f"🔧 默认提供商: {config.global_config.default_provider}")
            print(f"📝 日志级别: {config.global_config.logging.level}")
            print(f"🔄 最大重试次数: {config.global_config.retry.max_attempts}")
            
            # 列出启用的提供商
            enabled_providers = [
                name for name, provider in config.providers.items()
                if provider.enabled
            ]
            print(f"✅ 启用的提供商: {enabled_providers}")
            
            # 测试默认提供商
            if enabled_providers:
                client = create_llm_client(config_path=config_file)
                messages = [
                    LLMMessage(role="user", content=f"这是来自{env_name}环境的测试消息。")
                ]
                response = await client.chat_completion(messages)
                print(f"🤖 回复: {response.content[:100]}...")
            
        except Exception as e:
            print(f"❌ 环境 {env_name} 测试失败: {e}")


async def demo_model_comparison(config_path: Optional[str] = None) -> None:
    """
    演示模型对比
    
    Args:
        config_path: 配置文件路径
    """
    print_section("模型对比演示")
    
    try:
        factory = LLMProviderFactory(config_path)
        
        # 获取第一个可用提供商
        enabled_providers = factory.list_enabled_providers()
        if not enabled_providers:
            print("❌ 没有可用的提供商")
            return
        
        provider_name = enabled_providers[0]
        print(f"🔍 使用提供商: {provider_name}")
        
        # 获取提供商支持的模型
        models = factory.get_provider_models(provider_name)
        print(f"📋 可用模型: {models}")
        
        # 测试不同模型
        test_prompt = "请用一句话解释什么是机器学习。"
        
        for model in models[:2]:  # 限制测试前2个模型
            print_subsection(f"模型: {model}")
            
            try:
                client = factory.get_provider(provider_name)
                messages = [
                    LLMMessage(role="user", content=test_prompt)
                ]
                
                # 这里应该支持指定模型，但当前实现可能需要扩展
                response = await client.chat_completion(messages)
                print(f"🤖 回复: {response.content}")
                print(f"📊 实际使用模型: {response.model}")
                print(f"⏱️  响应时间: {response.response_time:.2f}秒")
                
            except Exception as e:
                print(f"❌ 模型 {model} 测试失败: {e}")
        
        factory.close()
        
    except Exception as e:
        print(f"❌ 模型对比演示失败: {e}")


async def demo_config_validation(config_path: Optional[str] = None) -> None:
    """
    演示配置验证
    
    Args:
        config_path: 配置文件路径
    """
    print_section("配置验证演示")
    
    try:
        factory = LLMProviderFactory(config_path)
        
        # 获取所有提供商
        config = factory.load_config()
        all_providers = list(config.providers.keys())
        print(f"🔍 所有配置的提供商: {all_providers}")
        
        # 验证每个提供商的配置
        for provider_name in all_providers:
            print_subsection(f"验证提供商: {provider_name}")
            
            is_valid = factory.validate_provider_config(provider_name)
            if is_valid:
                print(f"✅ {provider_name} 配置有效")
            else:
                print(f"❌ {provider_name} 配置无效")
        
        # 获取提供商信息
        provider_infos = factory.list_available_providers()
        print_subsection("提供商详细信息")
        
        for info in provider_infos:
            print(f"📋 提供商: {info.name}")
            print(f"   类型: {info.type}")
            print(f"   启用: {info.enabled}")
            print(f"   客户端类: {info.client_class.__name__}")
            print(f"   默认模型: {info.config.default_model}")
        
        factory.close()
        
    except Exception as e:
        print(f"❌ 配置验证演示失败: {e}")


async def demo_error_handling(config_path: Optional[str] = None) -> None:
    """
    演示错误处理
    
    Args:
        config_path: 配置文件路径
    """
    print_section("错误处理演示")
    
    # 测试不存在的提供商
    print_subsection("测试不存在的提供商")
    try:
        client = create_llm_client(provider_name="nonexistent_provider", config_path=config_path)
        print("❌ 应该抛出异常但没有")
    except Exception as e:
        print(f"✅ 正确捕获异常: {e}")
    
    # 测试无效的配置文件
    print_subsection("测试无效的配置文件")
    try:
        client = create_llm_client(config_path="nonexistent_config.yaml")
        print("❌ 应该抛出异常但没有")
    except Exception as e:
        print(f"✅ 正确捕获异常: {e}")
    
    # 测试禁用的提供商
    print_subsection("测试禁用的提供商")
    try:
        factory = LLMProviderFactory(config_path)
        config = factory.load_config()
        
        # 查找禁用的提供商
        disabled_providers = [
            name for name, provider in config.providers.items()
            if not provider.enabled
        ]
        
        if disabled_providers:
            provider_name = disabled_providers[0]
            print(f"🔍 测试禁用的提供商: {provider_name}")
            client = factory.get_provider(provider_name)
            print("❌ 应该抛出异常但没有")
        else:
            print("ℹ️  没有禁用的提供商可供测试")
        
        factory.close()
        
    except Exception as e:
        print(f"✅ 正确捕获异常: {e}")


async def demo_performance_comparison(config_path: Optional[str] = None) -> None:
    """
    演示性能对比
    
    Args:
        config_path: 配置文件路径
    """
    print_section("性能对比演示")
    
    try:
        factory = LLMProviderFactory(config_path)
        enabled_providers = factory.list_enabled_providers()
        
        if len(enabled_providers) < 2:
            print("ℹ️  需要至少2个启用的提供商进行性能对比")
            return
        
        # 测试消息
        test_messages = [
            LLMMessage(role="user", content="请简单介绍一下Python编程语言。")
        ]
        
        results = []
        
        # 测试每个提供商的性能
        for provider_name in enabled_providers[:3]:  # 限制测试前3个
            print_subsection(f"性能测试: {provider_name}")
            
            try:
                client = factory.get_provider(provider_name)
                
                # 记录开始时间
                import time
                start_time = time.time()
                
                # 发送请求
                response = await client.chat_completion(test_messages)
                
                # 计算总时间
                total_time = time.time() - start_time
                
                results.append({
                    'provider': provider_name,
                    'model': response.model,
                    'response_time': response.response_time,
                    'total_time': total_time,
                    'tokens': response.usage.get('total_tokens', 0) if response.usage else 0,
                    'content_length': len(response.content)
                })
                
                print(f"⏱️  响应时间: {response.response_time:.2f}秒")
                print(f"🔢 总时间: {total_time:.2f}秒")
                print(f"📝 内容长度: {len(response.content)}字符")
                
            except Exception as e:
                print(f"❌ {provider_name} 性能测试失败: {e}")
        
        # 显示性能对比结果
        if results:
            print_subsection("性能对比结果")
            print(f"{'提供商':<15} {'模型':<20} {'响应时间':<10} {'总时间':<10} {'内容长度':<10}")
            print("-" * 75)
            
            for result in sorted(results, key=lambda x: x['response_time']):
                print(f"{result['provider']:<15} {result['model']:<20} "
                      f"{result['response_time']:<10.2f} {result['total_time']:<10.2f} "
                      f"{result['content_length']:<10}")
        
        factory.close()
        
    except Exception as e:
        print(f"❌ 性能对比演示失败: {e}")


async def interactive_demo(config_path: Optional[str] = None) -> None:
    """
    交互式演示
    
    Args:
        config_path: 配置文件路径
    """
    print_section("交互式演示")
    
    try:
        factory = LLMProviderFactory(config_path)
        enabled_providers = factory.list_enabled_providers()
        
        if not enabled_providers:
            print("❌ 没有可用的提供商")
            return
        
        print("🎯 可用的提供商:")
        for i, provider in enumerate(enabled_providers, 1):
            print(f"  {i}. {provider}")
        
        # 选择提供商
        while True:
            try:
                choice = input(f"\n请选择提供商 (1-{len(enabled_providers)}, 或输入 'q' 退出): ").strip()
                
                if choice.lower() == 'q':
                    break
                
                provider_index = int(choice) - 1
                if 0 <= provider_index < len(enabled_providers):
                    provider_name = enabled_providers[provider_index]
                    break
                else:
                    print("❌ 无效的选择，请重试")
            except ValueError:
                print("❌ 请输入有效的数字")
        
        if choice.lower() == 'q':
            print("👋 退出交互式演示")
            return
        
        # 创建客户端
        print(f"🔧 使用提供商: {provider_name}")
        client = factory.get_provider(provider_name)
        
        # 对话循环
        conversation_history = []
        
        print("\n💬 开始对话 (输入 'quit' 退出):")
        
        while True:
            user_input = input("\n👤 您: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            try:
                # 添加用户消息到历史
                conversation_history.append(LLMMessage(role="user", content=user_input))
                
                # 发送消息
                print("🤖 正在思考...")
                response = await client.chat_completion(conversation_history)
                
                # 显示回复
                print(f"🤖 {provider_name}: {response.content}")
                print(f"   (模型: {response.model}, 响应时间: {response.response_time:.2f}秒)")
                
                # 添加助手回复到历史
                conversation_history.append(LLMMessage(role="assistant", content=response.content))
                
                # 限制历史长度
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-10:]
                
            except Exception as e:
                print(f"❌ 发送消息失败: {e}")
        
        print("👋 对话结束")
        factory.close()
        
    except Exception as e:
        print(f"❌ 交互式演示失败: {e}")


async def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(description="配置驱动的LLM演示脚本")
    parser.add_argument("--config", "-c", type=str, help="配置文件路径")
    parser.add_argument("--demo", "-d", type=str, 
                       choices=["basic", "switching", "environments", "models", 
                               "validation", "errors", "performance", "interactive", "all"],
                       default="all", help="要运行的演示类型")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    print("🚀 配置驱动的LLM演示脚本")
    print(f"📁 配置文件: {args.config or '默认配置'}")
    print(f"🎯 演示类型: {args.demo}")
    
    # 运行演示
    demos = {
        "basic": demo_basic_usage,
        "switching": demo_provider_switching,
        "environments": demo_environment_configs,
        "models": demo_model_comparison,
        "validation": demo_config_validation,
        "errors": demo_error_handling,
        "performance": demo_performance_comparison,
        "interactive": interactive_demo
    }
    
    if args.demo == "all":
        # 运行所有演示（除了交互式）
        for demo_name, demo_func in demos.items():
            if demo_name != "interactive":
                try:
                    if demo_name == "environments":
                        await demo_func()
                    else:
                        await demo_func(args.config)
                except Exception as e:
                    print(f"❌ 演示 {demo_name} 失败: {e}")
                    
        # 询问是否运行交互式演示
        if input("\n🤔 是否运行交互式演示? (y/N): ").strip().lower() == 'y':
            await interactive_demo(args.config)
    else:
        # 运行指定的演示
        demo_func = demos[args.demo]
        if args.demo == "environments":
            await demo_func()
        else:
            await demo_func(args.config)
    
    print("\n✅ 演示完成！")


if __name__ == "__main__":
    asyncio.run(main())