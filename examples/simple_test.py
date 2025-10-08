#!/usr/bin/env python3
"""
简单的Jarvis功能测试脚本

这个脚本测试Jarvis的基本功能，不依赖外部MCP服务器。
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入模块
from src.jarvis_agent import JarvisAgent, JarvisConfig


def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


async def test_basic_functionality():
    """测试基本功能"""
    print("🧪 Jarvis基本功能测试")
    print("=" * 40)
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 创建简单配置（不使用MCP服务器）
        config = JarvisConfig(
            name="TestJarvis",
            log_level="INFO",
            max_conversation_history=10,
            enable_tool_chaining=False,
            confidence_threshold=0.7,
            mcp_servers=[]  # 空的MCP服务器列表
        )
        
        logger.info("创建Jarvis代理...")
        agent = JarvisAgent(config)
        
        logger.info("启动代理...")
        await agent.start()
        
        print("✅ Jarvis代理启动成功！")
        
        # 测试状态
        status = agent.get_status()
        print(f"📊 代理状态: {status}")
        
        # 测试工具
        tools = agent.get_available_tools()
        print(f"🔧 可用工具: {tools}")
        
        # 测试简单对话
        print("\n🗣️  测试对话功能:")
        test_messages = [
            "你好",
            "你是谁？",
            "你能做什么？"
        ]
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n测试 {i}: {message}")
            try:
                response = await agent.process_message(message)
                print(f"回复: {response}")
            except Exception as e:
                print(f"❌ 处理消息出错: {e}")
        
        # 测试健康检查
        health = await agent.health_check()
        print(f"\n🏥 健康检查: {health}")
        
        print("\n✅ 所有基本功能测试完成！")
        
    except Exception as e:
        logger.error(f"测试过程中出错: {e}")
        print(f"❌ 测试失败: {e}")
        return 1
    
    finally:
        # 清理资源
        try:
            if 'agent' in locals():
                logger.info("关闭代理...")
                await agent.stop()
                print("✅ 代理已安全关闭")
        except Exception as e:
            logger.error(f"关闭代理时出错: {e}")
    
    return 0


def main():
    """主函数"""
    try:
        exit_code = asyncio.run(test_basic_functionality())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 测试被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 测试运行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()