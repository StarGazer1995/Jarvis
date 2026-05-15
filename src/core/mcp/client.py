"""
ARK引擎的MCP客户端实现
使用官方modelcontextprotocol SDK提供标准化的MCP客户端功能
"""

import json
import logging
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolRequest, Tool

from ..config.server import ServerStatus, SimpleMCPServerConfig


class ARKMCPClient:
    """
    ARK引擎的MCP客户端
    基于官方modelcontextprotocol SDK实现，提供标准化的MCP客户端功能
    """

    def __init__(self):
        """初始化MCP客户端"""
        self.logger = logging.getLogger(__name__)
        self.servers: dict[str, SimpleMCPServerConfig] = {}
        self.sessions: dict[str, ClientSession] = {}
        self.server_exit_stacks: dict[str, AsyncExitStack] = {}
        self.available_tools: dict[str, Tool] = {}
        self.tool_schemas: dict[str, dict[str, Any]] = {}
        self.is_initialized = False
        self._running = False

        # 内置工具
        self._builtin_tools = {
            "echo": {
                "name": "echo",
                "description": "Echo back the input message",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to echo back",
                        }
                    },
                    "required": ["message"],
                },
            },
            "get_time": {
                "name": "get_time",
                "description": "Get current time",
                "inputSchema": {"type": "object", "properties": {}, "required": []},
            },
        }

    async def initialize_from_config(self, config_path: str) -> None:
        """
        从配置文件初始化MCP客户端

        Args:
            config_path: 配置文件路径
        """
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, encoding="utf-8") as f:
                    config_data = json.load(f)

                # 加载服务器配置
                for server_data in config_data.get("servers", []):
                    server_config = SimpleMCPServerConfig.from_dict(server_data)
                    self.servers[server_config.name] = server_config

                self.logger.info(f"从配置文件加载了 {len(self.servers)} 个服务器配置")
            else:
                self.logger.warning(f"配置文件不存在: {config_path}")
                await self._initialize_demo_servers()

            # 连接到启用的服务器
            await self._connect_to_servers()
            self.is_initialized = True

        except Exception as e:
            self.logger.error(f"初始化MCP客户端失败: {e}")
            raise

    async def _initialize_demo_servers(self) -> None:
        """初始化演示服务器配置"""
        demo_servers = [
            SimpleMCPServerConfig(
                name="file_operations",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                description="文件系统操作服务器",
            ),
            SimpleMCPServerConfig(
                name="web_search",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-brave-search"],
                description="网页搜索服务器",
                enabled=False,  # 默认禁用，需要API密钥
            ),
        ]

        for server in demo_servers:
            self.servers[server.name] = server

        self.logger.info("初始化了演示服务器配置")

    async def _connect_to_servers(self) -> None:
        """连接到所有启用的服务器"""
        for server_name, server_config in self.servers.items():
            if server_config.enabled:
                try:
                    await self._connect_to_server_internal(server_name, server_config)
                except Exception as e:
                    self.logger.error(f"连接到服务器 {server_name} 失败: {e}")

    async def list_tools(self) -> list[dict[str, Any]]:
        """
        列出所有可用的工具

        Returns:
            工具列表，每个工具包含名称、描述和输入模式
        """
        tools = []
        all_tools = self.get_all_tools()

        for tool_key, tool_info in all_tools.items():
            # Determine server name
            if tool_key in self._builtin_tools:
                server_name = "builtin"
                tool_entry = {
                    "name": tool_key,
                    "description": tool_info["description"],
                    "inputSchema": tool_info["inputSchema"],
                    "server": server_name,
                }
            else:
                # MCP server tool
                server_name = tool_key.split(":")[0] if ":" in tool_key else "unknown"
                tool_entry = {
                    "name": tool_info.get("name", tool_key),
                    "description": tool_info.get("description", ""),
                    "inputSchema": tool_info.get(
                        "parameters", tool_info.get("inputSchema", {})
                    ),
                    "server": server_name,
                    "full_name": tool_key,
                }

            tools.append(tool_entry)

        return tools

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """
        调用指定的工具

        Args:
            tool_name: 工具名称（可以是简单名称或完整名称）
            arguments: 工具参数

        Returns:
            工具执行结果
        """
        try:
            if tool_name in self._builtin_tools:
                return await self._call_builtin_tool(tool_name, arguments)

            tool_key, tool = self._resolve_tool(tool_name)
            if tool_key is None or tool is None:
                raise ValueError(f"工具 '{tool_name}' 不存在")

            server_name = self._get_server_name(tool_key, tool)
            tool_real_name = tool.get("name") if isinstance(tool, dict) else tool.name
            raw_result = await self._execute_server_tool(
                server_name, tool_real_name, arguments
            )
            content = raw_result.get("result")
            if content is None:
                content = []

            return {
                "success": True,
                "result": content,
                "tool": tool_name,
                "server": server_name,
            }

        except Exception as e:
            self.logger.error(f"调用工具 {tool_name} 失败: {e}")
            return {"success": False, "error": str(e), "tool": tool_name}

    async def _call_builtin_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """
        调用内置工具

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            工具执行结果
        """
        if tool_name == "echo":
            message = arguments.get("message", "")
            return {
                "success": True,
                "result": [{"type": "text", "text": f"Echo: {message}"}],
                "tool": tool_name,
                "server": "builtin",
            }

        elif tool_name == "get_time":
            current_time = datetime.now().isoformat()
            return {
                "success": True,
                "result": [{"type": "text", "text": f"Current time: {current_time}"}],
                "tool": tool_name,
                "server": "builtin",
            }

        else:
            raise ValueError(f"未知的内置工具: {tool_name}")

    async def disconnect_server(self, server_name: str) -> bool:
        """
        断开与指定服务器的连接

        Args:
            server_name: 服务器名称

        Returns:
            是否成功断开连接
        """
        try:
            if server_name in self.sessions:
                # Close the exit stack which closes the session and stdio streams
                if server_name in self.server_exit_stacks:
                    await self.server_exit_stacks[server_name].aclose()
                    del self.server_exit_stacks[server_name]

                if server_name in self.sessions:
                    del self.sessions[server_name]

                # 移除该服务器的工具
                tools_to_remove = [
                    key
                    for key in self.available_tools.keys()
                    if key.startswith(f"{server_name}:")
                ]
                for tool_key in tools_to_remove:
                    del self.available_tools[tool_key]

                self.logger.info(f"成功断开与服务器 {server_name} 的连接")
                return True

            return False

        except Exception as e:
            self.logger.error(f"断开服务器 {server_name} 连接失败: {e}")
            return False

    async def close(self) -> None:
        """关闭所有连接"""
        try:
            for server_name in list(self.sessions.keys()):
                await self.disconnect_server(server_name)

            self.sessions.clear()
            self.available_tools.clear()
            self.is_initialized = False

            self.logger.info("MCP客户端已关闭")

        except Exception as e:
            self.logger.error(f"关闭MCP客户端失败: {e}")

    def get_server_status(self, server_name: str) -> ServerStatus:
        """
        获取服务器状态

        Args:
            server_name: 服务器名称

        Returns:
            服务器状态
        """
        if server_name in self.sessions:
            return ServerStatus.CONNECTED
        elif server_name in self.servers:
            return ServerStatus.DISCONNECTED
        else:
            return ServerStatus.UNKNOWN

    def get_all_server_status(self) -> dict[str, ServerStatus]:
        """
        获取所有服务器的状态

        Returns:
            Dict[str, ServerStatus]: 服务器名称到状态的映射
        """
        status_dict = {}
        for server_name in self.servers:
            status_dict[server_name] = self.get_server_status(server_name)

        return status_dict

    async def demo(self) -> None:
        """
        演示MCP客户端功能
        显示客户端状态和可用工具信息
        """
        print("🌐 ARK MCP Client Demo")
        print("=" * 30)

        print("📊 Client Status:")
        print(f"  - Initialized: {self.is_initialized}")
        print(f"  - Configured Servers: {len(self.servers)}")
        print(f"  - Active Sessions: {len(self.sessions)}")
        print(f"  - Available Tools: {len(self.available_tools)}")

        if self.servers:
            print("\n🔧 Configured Servers:")
            for name, config in self.servers.items():
                status = self.get_server_status(name)
                print(f"  - {name}: {status.value}")

        if self.available_tools:
            print("\n🛠️  Available Tools:")
            for tool_name in self.available_tools:
                print(f"  - {tool_name}")

        print("\n✅ Demo completed!")

    # 测试兼容性方法
    async def add_server(self, server_config: SimpleMCPServerConfig) -> bool:
        """
        添加服务器配置

        Args:
            server_config: 服务器配置

        Returns:
            是否添加成功
        """
        if server_config.name in self.servers:
            return False

        self.servers[server_config.name] = server_config
        return True

    async def connect_to_server(self, server_config: SimpleMCPServerConfig) -> bool:
        """
        连接到MCP服务器（公共方法）

        Args:
            server_config: 服务器配置

        Returns:
            是否连接成功
        """
        # 先添加服务器配置
        await self.add_server(server_config)
        # 然后连接到服务器
        return await self._connect_to_server(server_config)

    async def remove_server(self, server_name: str) -> bool:
        """
        移除服务器配置

        Args:
            server_name: 服务器名称

        Returns:
            是否移除成功
        """
        if server_name not in self.servers:
            return False

        # 断开连接
        if server_name in self.sessions:
            await self.disconnect_server(server_name)

        del self.servers[server_name]
        return True

    def get_server(self, server_name: str) -> SimpleMCPServerConfig | None:
        """
        获取服务器配置

        Args:
            server_name: 服务器名称

        Returns:
            服务器配置或None
        """
        return self.servers.get(server_name)

    def list_servers(self) -> list[SimpleMCPServerConfig]:
        """
        列出所有服务器配置

        Returns:
            服务器配置列表
        """
        return list(self.servers.values())

    async def start(self) -> None:
        """启动客户端"""
        self._running = True
        await self._connect_to_servers()
        await self._discover_tools()

    async def stop(self) -> None:
        """停止客户端"""
        self._running = False
        await self._disconnect_all_servers()

    async def discover_tools(self, server_name: str) -> list[dict[str, Any]]:
        """
        发现服务器工具

        Args:
            server_name: 服务器名称

        Returns:
            工具列表
        """
        if server_name not in self.sessions:
            return []

        session = self.sessions[server_name]
        tools_result = await session.list_tools()

        tools = []
        for tool in tools_result.tools:
            tool_info = {
                "name": tool.name,
                "description": tool.description or "",
                "inputSchema": tool.inputSchema,
                "server": server_name,
            }
            tools.append(tool_info)

            # 更新工具模式和可用工具
            tool_key = f"{server_name}:{tool.name}"
            self.tool_schemas[tool_key] = tool.inputSchema
            self.available_tools[tool_key] = tool_info

        return tools

    async def execute_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        执行工具（测试兼容性方法）

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            执行结果，如果工具不存在或执行失败则返回None
        """
        try:
            if tool_name in self._builtin_tools:
                return await self._call_builtin_tool(tool_name, arguments)

            tool_key, tool_info = self._resolve_tool(tool_name)
            if tool_key is None or tool_info is None:
                return None

            server_name = self._get_server_name(tool_key, tool_info)
            tool_real_name = (
                tool_info.get("name")
                if isinstance(tool_info, dict)
                else getattr(tool_info, "name", tool_name)
            )
            return await self._execute_server_tool(
                server_name, tool_real_name, arguments
            )
        except Exception as e:
            logging.error(f"执行工具 {tool_name} 失败: {e}")
            return None

    def _resolve_tool(self, tool_name: str) -> tuple[str | None, Any]:
        if tool_name in self.available_tools:
            return tool_name, self.available_tools[tool_name]
        for key, tool in self.available_tools.items():
            candidate_name = (
                tool.get("name")
                if isinstance(tool, dict)
                else getattr(tool, "name", None)
            )
            if candidate_name == tool_name:
                return key, tool
        return None, None

    def _get_server_name(self, tool_key: str, tool_info: Any) -> str:
        if isinstance(tool_info, dict) and tool_info.get("server"):
            return str(tool_info["server"])
        return tool_key.split(":")[0]

    async def _call_session_tool(
        self, session: ClientSession, tool_name: str, arguments: dict[str, Any]
    ) -> Any:
        try:
            return await session.call_tool(tool_name, arguments)
        except TypeError:
            request = CallToolRequest(
                method="tools/call", params={"name": tool_name, "arguments": arguments}
            )
            return await session.call_tool(request)

    async def _execute_server_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """
        执行服务器工具

        Args:
            server_name: 服务器名称
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            执行结果
        """
        if server_name not in self.sessions:
            raise ValueError(f"服务器 {server_name} 未连接")

        session = self.sessions[server_name]
        result = await self._call_session_tool(session, tool_name, arguments)

        content = getattr(result, "content", None)
        normalized_content = None
        if content is not None:
            normalized_content = [
                c.model_dump() if hasattr(c, "model_dump") else c for c in content
            ]

        return {
            "success": not result.isError,
            "result": normalized_content
            if normalized_content is not None
            else str(result),
            "error": str(result) if result.isError else None,
        }

    def get_available_tools(self) -> dict[str, dict[str, Any]]:
        """
        获取可用的MCP服务器工具（不包含内置工具）

        Returns:
            工具字典
        """
        # 直接返回available_tools，因为它已经是正确的格式
        return self.available_tools.copy()

    def get_all_tools(self) -> dict[str, dict[str, Any]]:
        """
        获取所有可用工具（包含内置工具和MCP服务器工具）

        Returns:
            工具字典
        """
        tools = {}

        # 添加内置工具
        for tool_name, tool_info in self._builtin_tools.items():
            tools[tool_name] = tool_info

        # 添加MCP服务器工具 (available_tools already contains dictionaries)
        tools.update(self.available_tools)

        return tools

    def get_tool_schema(self, tool_name: str) -> dict[str, Any] | None:
        """
        获取工具模式

        Args:
            tool_name: 工具名称

        Returns:
            工具模式或None
        """
        # 检查内置工具
        if tool_name in self._builtin_tools:
            return self._builtin_tools[tool_name]["inputSchema"]

        # 检查MCP工具
        return self.tool_schemas.get(tool_name)

    def get_client_stats(self) -> dict[str, Any]:
        """
        获取客户端统计信息

        Returns:
            统计信息
        """
        return {
            "total_servers": len(self.servers),
            "connected_servers": len(self.sessions),
            "total_tools": len(self.available_tools),
            "running": self._running,
        }

    async def _discover_tools(self) -> None:
        """发现所有服务器的工具"""
        for server_name in self.sessions:
            await self.discover_tools(server_name)

    async def _connect_to_server(
        self, server_name_or_config, server_config=None
    ) -> bool:
        """
        连接到指定的MCP服务器（重载方法以支持测试）

        Args:
            server_name_or_config: 服务器名称或配置对象
            server_config: 服务器配置（当第一个参数是名称时使用）

        Returns:
            是否连接成功
        """
        if isinstance(server_name_or_config, SimpleMCPServerConfig):
            # 测试兼容性：直接传入配置对象
            config = server_name_or_config
            try:
                await self._connect_to_server_internal(config.name, config)
                return True
            except Exception:
                return False
        else:
            # 正常调用：传入名称和配置
            server_name = server_name_or_config
            try:
                await self._connect_to_server_internal(server_name, server_config)
                return True
            except Exception:
                return False

    async def _connect_to_server_internal(
        self, server_name: str, server_config: SimpleMCPServerConfig
    ) -> None:
        """
        连接到指定的MCP服务器（内部实现）

        Args:
            server_name: 服务器名称
            server_config: 服务器配置
        """
        try:
            # 创建服务器参数
            # 处理命令格式：如果command是列表，第一个元素是命令，其余是参数
            if isinstance(server_config.command, list):
                command = server_config.command[0]
                args = server_config.command[1:] + (server_config.args or [])
            else:
                command = server_config.command
                args = server_config.args or []

            server_params = StdioServerParameters(
                command=command, args=args, env=server_config.env
            )

            # Create a new exit stack for this server
            stack = AsyncExitStack()
            self.server_exit_stacks[server_name] = stack

            # 建立连接
            # Enter the stdio context
            read_stream, write_stream = await stack.enter_async_context(
                stdio_client(server_params)
            )

            session = ClientSession(read_stream, write_stream)
            # Enter the session context
            await stack.enter_async_context(session)

            # 初始化会话
            await session.initialize()

            # 存储会话
            self.sessions[server_name] = session

            # 获取工具列表
            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                tool_info = {
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": tool.inputSchema,
                    "server": server_name,
                }
                tool_key = f"{server_name}:{tool.name}"
                self.available_tools[tool_key] = tool_info
                self.tool_schemas[tool_key] = tool.inputSchema

            self.logger.info(
                f"成功连接到服务器 {server_name}，发现 {len(tools_result.tools)} 个工具"
            )

        except Exception as e:
            self.logger.error(f"连接到服务器 {server_name} 失败: {e}")
            # Clean up the stack if connection failed
            if server_name in self.server_exit_stacks:
                await self.server_exit_stacks[server_name].aclose()
                del self.server_exit_stacks[server_name]
            raise

    async def _disconnect_all_servers(self) -> None:
        """
        断开所有服务器连接
        """
        for server_name in list(self.sessions.keys()):
            await self.disconnect_server(server_name)

    async def _get_server_tools(self, server_name: str) -> dict[str, dict[str, Any]]:
        """
        获取指定服务器的工具列表

        Args:
            server_name: 服务器名称

        Returns:
            服务器工具字典
        """
        if server_name not in self.sessions:
            return {}

        try:
            session = self.sessions[server_name]
            tools_response = await session.list_tools()

            server_tools = {}
            for tool in tools_response.tools:
                server_tools[tool.name] = {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                    "server": server_name,
                }

            return server_tools
        except Exception as e:
            logging.error(f"获取服务器 {server_name} 工具失败: {e}")
            return {}

    def __repr__(self) -> str:
        """返回客户端的字符串表示"""
        return f"ARKMCPClient(servers={len(self.servers)}, tools={len(self.available_tools)})"
