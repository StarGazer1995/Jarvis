# Creating New Capabilities in Project Jarvis

This guide explains how to extend Project Jarvis with new capabilities, specifically focusing on adding new tools and integrations via the Model Context Protocol (MCP).

## Overview

Project Jarvis uses **ARK (Agent Reactor Kernel)** as its core engine, which leverages **MCP (Model Context Protocol)** to interact with external tools and services. To add a new capability, you typically:

1.  Create or identify an MCP server that provides the desired tools.
2.  Configure Jarvis to connect to this MCP server.
3.  (Optional) Implement custom logic in Jarvis if complex coordination is needed.

## Option 1: Adding an Existing MCP Server

The easiest way to add capabilities is to use an existing MCP server. The MCP ecosystem has many pre-built servers for file systems, databases, APIs, etc.

### Configuration

Add the server configuration to your `config/mcp_servers.json` (or equivalent config file loaded by `ARKMCPClient`):

```json
{
  "servers": [
    {
      "name": "filesystem",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/allowed/path"],
      "env": {}
    }
  ]
}
```

Or programmatically in your code:

```python
from src.core.config.server import SimpleMCPServerConfig

server_config = SimpleMCPServerConfig(
    name="filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/allowed/path"]
)
agent.add_mcp_server(server_config)
```

## Option 2: Creating a Custom MCP Server

If you need custom tools, you can build your own MCP server using Python or TypeScript.

### Using Python (FastMCP)

We recommend using `fastmcp` for quick development, or the low-level `mcp` library for more control (and to avoid banner output in stdio).

**Example (using `mcp` low-level library):**

```python
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

app = Server("My Custom Server")

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="my_tool",
            description="Description of what my tool does",
            inputSchema={
                "type": "object",
                "properties": {
                    "param1": {"type": "string"}
                },
                "required": ["param1"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if name == "my_tool":
        # Implement your logic here
        result = f"Processed {arguments['param1']}"
        return [types.TextContent(type="text", text=result)]
    
    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

### Integration Steps

1.  Save your server script (e.g., `src/capabilities/my_server.py`).
2.  Add it to Jarvis configuration:
    ```python
    import sys
    config = SimpleMCPServerConfig(
        name="my_capability",
        command=sys.executable,
        args=["src/capabilities/my_server.py"]
    )
    ```

## Best Practices

*   **Stdio Isolation**: Ensure your server script ONLY prints JSON-RPC messages to `stdout`. Use `sys.stderr` for logs.
*   **Error Handling**: Handle exceptions gracefully within your tools and return meaningful error messages.
*   **Security**: Validate inputs rigorously. MCP provides a secure channel, but your tool logic must be safe.
*   **Type Safety**: Define clear JSON schemas for your tools to help LLMs understand how to use them.

## Testing Your Capability

1.  **Unit Tests**: Test your tool logic in isolation.
2.  **Integration Tests**: Use `ARKMCPClient` to connect to your server and call tools programmatically (see `tests/core/mcp/test_demo_integration.py` for an example).
3.  **E2E Tests**: Verify that Jarvis can use the tool in a conversation.
