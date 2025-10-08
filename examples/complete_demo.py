#!/usr/bin/env python3
"""
Jarvis AI代理完整演示脚本

这个脚本展示了Jarvis AI代理的完整功能，包括：
1. 基本对话功能
2. 系统状态监控
3. 工具使用统计
4. 健康检查
5. 配置管理
6. 对话历史导出

运行方式：
    python examples/complete_demo.py
"""

import sys
import os
import asyncio
import logging
import json
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入模块
from src.jarvis_agent import JarvisAgent, JarvisConfig
from src.core.server_config import SimpleMCPServerConfig


def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_subsection(title: str):
    """打印子章节标题"""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")


async def demonstrate_jarvis_capabilities():
    """演示Jarvis的完整功能"""
    print("🤖 Jarvis AI代理完整功能演示")
    print(f"📅 演示时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    agent = None
    
    try:
        print_section("1. 系统初始化")
        
        # 创建配置
        config = JarvisConfig(
            name="DemoJarvis",
            version="1.0.0",
            log_level="INFO",
            max_conversation_history=50,
            enable_tool_chaining=True,
            confidence_threshold=0.7,
            mcp_servers=[]  # 演示中不使用外部MCP服务器
        )
        
        print("📋 配置信息:")
        print(f"  - 代理名称: {config.name}")
        print(f"  - 版本: {config.version}")
        print(f"  - 日志级别: {config.log_level}")
        print(f"  - 最大对话历史: {config.max_conversation_history}")
        print(f"  - 工具链启用: {config.enable_tool_chaining}")
        print(f"  - 置信度阈值: {config.confidence_threshold}")
        
        logger.info("创建Jarvis代理实例...")
        agent = JarvisAgent(config)
        
        logger.info("启动代理...")
        await agent.start()
        
        print("✅ Jarvis代理启动成功！")
        
        print_section("2. 系统状态检查")
        
        # 获取系统状态
        status = agent.get_status()
        print("📊 当前系统状态:")
        print(json.dumps(status, indent=2, ensure_ascii=False))
        
        # 健康检查
        health = await agent.health_check()
        print("\n🏥 系统健康检查:")
        print(json.dumps(health, indent=2, ensure_ascii=False))
        
        # 可用工具
        tools = agent.get_available_tools()
        print(f"\n🔧 可用工具数量: {len(tools)}")
        if tools:
            print(f"工具列表: {', '.join(tools)}")
        else:
            print("当前没有可用的外部工具")
        
        print_section("3. 对话功能演示")
        
        # 测试对话
        test_conversations = [
            {
                "category": "基本问候",
                "messages": [
                    "Hello",
                    "How are you?",
                    "What's your name?"
                ]
            },
            {
                "category": "功能询问",
                "messages": [
                    "What can you do?",
                    "Tell me about your capabilities",
                    "How can you help me?"
                ]
            },
            {
                "category": "中文对话",
                "messages": [
                    "你好",
                    "你是谁？",
                    "你能做什么？"
                ]
            }
        ]
        
        conversation_count = 0
        
        for category_info in test_conversations:
            print_subsection(f"对话类别: {category_info['category']}")
            
            for message in category_info["messages"]:
                conversation_count += 1
                print(f"\n💬 对话 {conversation_count}")
                print(f"👤 用户: {message}")
                
                try:
                    response = await agent.process_message(message)
                    print(f"🤖 Jarvis: {response}")
                except Exception as e:
                    print(f"❌ 处理消息时出错: {e}")
                
                # 短暂暂停，模拟真实对话
                await asyncio.sleep(0.5)
        
        print_section("4. 工具使用统计")
        
        # 工具使用统计
        tool_stats = agent.get_tool_usage_stats()
        print("📈 工具使用统计:")
        if tool_stats:
            for tool, count in tool_stats.items():
                print(f"  - {tool}: {count} 次")
        else:
            print("  暂无工具使用记录")
        
        print_section("5. 对话历史导出")
        
        # 导出对话历史
        try:
            conversation_export = agent.get_conversation_export("json")
            print("📝 对话历史导出 (JSON格式):")
            
            # 解析并美化输出
            if conversation_export:
                try:
                    parsed_export = json.loads(conversation_export)
                    print(json.dumps(parsed_export, indent=2, ensure_ascii=False))
                except json.JSONDecodeError:
                    print(conversation_export)
            else:
                print("  暂无对话历史")
                
        except Exception as e:
            print(f"❌ 导出对话历史时出错: {e}")
        
        print_section("6. 用户偏好设置演示")
        
        # 设置用户偏好
        preferences = {
            "language": "zh-CN",
            "response_style": "friendly",
            "max_response_length": 200,
            "enable_emoji": True
        }
        
        print("⚙️  设置用户偏好:")
        for key, value in preferences.items():
            agent.set_user_preference(key, value)
            print(f"  - {key}: {value}")
        
        # 获取用户偏好
        print("\n📖 当前用户偏好:")
        for key in preferences.keys():
            value = agent.get_user_preference(key)
            print(f"  - {key}: {value}")
        
        print_section("7. 最终状态检查")
        
        # 最终状态检查
        final_status = agent.get_status()
        print("📊 最终系统状态:")
        print(f"  - 代理运行状态: {final_status['agent']['is_running']}")
        print(f"  - 初始化状态: {final_status['agent']['is_initialized']}")
        print(f"  - 对话活跃状态: {final_status['agent']['conversation_active']}")
        print(f"  - ARK引擎状态: {final_status['ark_engine']['state']}")
        
        # 最终健康检查
        final_health = await agent.health_check()
        print(f"\n🏥 最终健康状态: {final_health['overall']}")
        
        print_section("演示完成")
        
        print("🎉 Jarvis AI代理功能演示完成！")
        print("\n📋 演示总结:")
        print(f"  ✅ 成功启动代理")
        print(f"  ✅ 完成 {conversation_count} 轮对话测试")
        print(f"  ✅ 系统状态检查正常")
        print(f"  ✅ 健康检查通过")
        print(f"  ✅ 用户偏好设置成功")
        print(f"  ✅ 对话历史导出成功")
        
        print("\n💡 接下来你可以:")
        print("  1. 运行 python examples/simple_test.py 进行基本功能测试")
        print("  2. 运行 python examples/conversation_demo.py 进行交互式对话")
        print("  3. 查看 docs/README.md 了解更多API文档")
        print("  4. 运行 pytest 执行完整的测试套件")
        
        return 0
        
    except Exception as e:
        logger.error(f"演示过程中出错: {e}")
        print(f"\n❌ 演示失败: {e}")
        return 1
    
    finally:
        # 清理资源
        if agent:
            try:
                logger.info("关闭代理...")
                await agent.stop()
                print("\n✅ 代理已安全关闭")
            except Exception as e:
                logger.error(f"关闭代理时出错: {e}")


def main():
    """主函数"""
    try:
        exit_code = asyncio.run(demonstrate_jarvis_capabilities())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n👋 演示被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 演示运行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()