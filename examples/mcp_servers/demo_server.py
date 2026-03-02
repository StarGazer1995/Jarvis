import asyncio
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

# Create server
app = Server("Demo Server")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="add",
            description="Add two numbers",
            inputSchema={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        ),
        types.Tool(
            name="echo",
            description="Echo a message",
            inputSchema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if name == "add":
        return [
            types.TextContent(type="text", text=str(arguments["a"] + arguments["b"]))
        ]
    elif name == "echo":
        return [types.TextContent(type="text", text=f"Echo: {arguments['message']}")]

    raise ValueError(f"Unknown tool: {name}")


async def main():
    # Run the server using stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
