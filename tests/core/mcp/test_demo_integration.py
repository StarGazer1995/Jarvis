import pytest
import sys
from pathlib import Path

from src.core.mcp.client import ARKMCPClient
from src.core.config.server import SimpleMCPServerConfig


@pytest.mark.asyncio
async def test_demo_server_integration():
    """Test integration with the local demo MCP server."""

    # Path to the demo server script
    project_root = Path(__file__).parent.parent.parent.parent
    demo_server_path = project_root / "examples" / "mcp_servers" / "demo_server.py"

    if not demo_server_path.exists():
        pytest.skip(f"Demo server script not found at {demo_server_path}")

    # Configure the client to use the demo server
    client = ARKMCPClient()

    # Create server config
    # We use python to run the script. Ensure we use the same python interpreter.
    server_config = SimpleMCPServerConfig(
        name="demo_integration",
        command=sys.executable,
        args=[str(demo_server_path)],
        description="Integration Test Demo Server",
    )

    try:
        # Add server config
        await client.add_server(server_config)

        # Connect
        connected = await client.connect_to_server(server_config)
        assert connected, "Failed to connect to demo server"

        # List tools
        tools = await client.list_tools()
        tool_names = [t["name"] for t in tools]
        print(f"Discovered tools: {tool_names}")
        assert "add" in tool_names
        assert "echo" in tool_names

        # Call 'add' tool
        result = await client.call_tool("add", {"a": 10, "b": 20})
        assert result["success"] is True

        # The result['result'] is a list of Content objects (TextContent, ImageContent, etc.)
        # We need to verify the value.
        # Assuming TextContent for simple return values
        content = result["result"]
        assert len(content) > 0
        assert content[0]["text"] == "30"

        # Call 'echo' tool
        echo_result = await client.call_tool("echo", {"message": "Hello MCP"})
        assert echo_result["success"] is True
        content = echo_result["result"]
        assert len(content) > 0
        assert content[0]["text"] == "Echo: Hello MCP"

    finally:
        await client.stop()
