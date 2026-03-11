import logging
from typing import Any

from ..config.server import SimpleMCPServerConfig
from ..mcp.client import ARKMCPClient


async def connect_servers(
    mcp_client: ARKMCPClient,
    logger: logging.Logger,
    servers: list[SimpleMCPServerConfig] | None,
    connected_template: str,
    failed_template: str,
) -> None:
    if not servers:
        return
    for server_config in servers:
        success = await mcp_client.connect_to_server(server_config)
        if success:
            logger.info(connected_template.format(name=server_config.name))
        else:
            logger.warning(failed_template.format(name=server_config.name))


async def discover_tools(
    mcp_client: ARKMCPClient,
    logger: logging.Logger,
    unknown_name_template: str,
    failed_template: str,
) -> dict[str, Any]:
    available_tools: dict[str, Any] = {}
    for server_name in mcp_client.sessions.keys():
        try:
            server_tools = await mcp_client.discover_tools(server_name)
            for tool in server_tools:
                tool_name = tool.get(
                    "name", unknown_name_template.format(index=len(available_tools))
                )
                full_name = f"{server_name}:{tool_name}"
                if full_name in available_tools:
                    logger.warning(
                        "Duplicate tool detected and skipped: %s",
                        full_name,
                    )
                    continue
                available_tools[full_name] = tool
        except Exception as e:
            logger.warning(failed_template.format(server_name=server_name, error=e))
    return available_tools


async def close_mcp_client(mcp_client: ARKMCPClient) -> None:
    close = getattr(mcp_client, "close", None)
    if callable(close):
        await close()
        return
    disconnect_all = getattr(mcp_client, "disconnect_all", None)
    if callable(disconnect_all):
        await disconnect_all()
