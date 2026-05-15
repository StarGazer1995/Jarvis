#!/usr/bin/env python3
"""
Jarvis对话演示脚本

这个脚本演示如何启动Jarvis AI代理并进行对话测试。
它会正确设置Python路径并启动交互式对话模式。
"""

import asyncio
import logging
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 现在可以正确导入模块
from src.core.config.server import SimpleMCPServerConfig
from src.jarvis_agent import JarvisAgent, JarvisConfig


def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


async def run_conversation_demo():
    """运行对话演示"""
    print("🤖 Jarvis AI代理对话演示")
    print("=" * 50)

    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # 创建配置
        config = JarvisConfig(
            name="Jarvis",
            log_level="INFO",
            max_conversation_history=100,
            enable_tool_chaining=True,
            confidence_threshold=0.7,
            mcp_servers=[
                SimpleMCPServerConfig(
                    name="time", command="uvx", args=["mcp-server-time"], env={}
                )
            ],
        )

        # 创建代理
        logger.info("正在初始化Jarvis代理...")
        agent = JarvisAgent(config)

        # 启动代理
        logger.info("正在启动代理...")
        await agent.start()

        print("\n✅ Jarvis代理已成功启动！")
        print("💡 你可以开始与Jarvis对话了。输入 'quit' 或 'exit' 退出。")
        print("💡 输入 'help' 查看可用命令。")
        print("-" * 50)

        # 开始对话循环
        conversation_count = 0
        while True:
            try:
                # 获取用户输入
                user_input = input("\n👤 你: ").strip()

                # 检查退出命令
                if user_input.lower() in ["quit", "exit", "退出", "q"]:
                    print("\n👋 再见！感谢使用Jarvis！")
                    break

                # 检查帮助命令
                if user_input.lower() in ["help", "帮助", "h"]:
                    print("\n📖 可用命令:")
                    print("  - help/帮助: 显示此帮助信息")
                    print("  - status/状态: 显示代理状态")
                    print("  - tools/工具: 显示可用工具")
                    print("  - quit/exit/退出: 退出程序")
                    print("  - 或者直接输入任何问题与Jarvis对话")
                    continue

                # 检查状态命令
                if user_input.lower() in ["status", "状态"]:
                    status = await agent.get_status()
                    print(f"\n📊 代理状态: {status}")
                    continue

                # 检查工具命令
                if user_input.lower() in ["tools", "工具"]:
                    tools = agent.get_available_tools()
                    print(f"\n🔧 可用工具: {', '.join(tools) if tools else '无'}")
                    continue

                # 如果输入为空，跳过
                if not user_input:
                    continue

                # 处理用户消息
                print("🤖 Jarvis: ", end="", flush=True)

                response = await agent.process_message(user_input)
                print(response)

                conversation_count += 1

                # 每5轮对话显示一次统计
                if conversation_count % 5 == 0:
                    print(f"\n📈 已完成 {conversation_count} 轮对话")

            except KeyboardInterrupt:
                print("\n\n⚠️  检测到中断信号，正在退出...")
                break
            except Exception as e:
                logger.error(f"处理消息时出错: {e}")
                print(f"\n❌ 抱歉，处理您的消息时出现错误: {e}")
                print("请重试或输入 'quit' 退出。")

    except Exception as e:
        logger.error(f"启动代理时出错: {e}")
        print(f"\n❌ 启动Jarvis代理时出现错误: {e}")
        return 1

    finally:
        # 清理资源
        try:
            if "agent" in locals():
                logger.info("正在关闭代理...")
                await agent.stop()
                print("✅ 代理已安全关闭")
        except Exception as e:
            logger.error(f"关闭代理时出错: {e}")

    return 0


def main():
    """主函数"""
    try:
        # 运行异步主函数
        exit_code = asyncio.run(run_conversation_demo())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
