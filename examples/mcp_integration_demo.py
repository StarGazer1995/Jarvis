#!/usr/bin/env python3
"""
MCP Integration Demonstration.

This script demonstrates the Model Context Protocol (MCP) integration
capabilities of the ARK system, including:
- MCP server configuration and management
- Tool discovery and registration
- Tool execution with security validation
- Server health monitoring
- Configuration persistence

Run from project root: python examples/mcp_integration_demo.py --verbose
"""

import argparse
import asyncio
import json
import logging
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.mcp_client import ARKMCPClient
from core.security_manager import ARKSecurityManager, PermissionType
from core.server_config import ARKServerConfigManager, SimpleMCPServerConfig
from core.tool_registry import ARKToolRegistry, ToolCategory


async def demonstrate_mcp_server_management():
    """
    Demonstrate MCP server configuration and management.
    """
    print("🔧 Demonstrating MCP Server Management")
    print("=" * 45)

    # Create MCP client
    mcp_client = ARKMCPClient()
    print("✅ MCP Client initialized")

    # Create sample server configurations
    sample_servers = [
        SimpleMCPServerConfig(
            name="file_operations",
            command=["python", "-m", "mcp_servers.filesystem"],
            description="File system operations server",
            env={"MCP_LOG_LEVEL": "INFO"},
        ),
        SimpleMCPServerConfig(
            name="web_search",
            command=["python", "-m", "mcp_servers.web_search"],
            description="Web search and information retrieval",
            env={"API_KEY": "demo_key"},
        ),
        SimpleMCPServerConfig(
            name="system_tools",
            command=["python", "-m", "mcp_servers.system"],
            description="System administration tools",
            env={"SYSTEM_ACCESS": "restricted"},
        ),
    ]

    # Add servers to client
    print("\n📦 Adding MCP Servers...")
    for server_config in sample_servers:
        success = await mcp_client.add_server(server_config)
        status = "✅" if success else "❌"
        print(f"{status} Added server: {server_config.name}")
        print(f"   Description: {server_config.description}")
        print(f"   Command: {' '.join(server_config.command)}")

    # List configured servers
    print(f"\n📋 Configured Servers: {len(mcp_client.servers)}")
    for name, config in mcp_client.servers.items():
        print(f"  - {name}: {config.description}")

    # Demonstrate server operations (simulated)
    print("\n🚀 Server Operations (Simulated)")
    print("-" * 30)

    for server_name in ["file_operations", "web_search"]:
        print(f"\n🔄 Testing server: {server_name}")

        # Simulate server startup
        print(f"  ⏳ Starting {server_name}...")
        await asyncio.sleep(0.5)  # Simulate startup time
        print(f"  ✅ {server_name} started successfully")

        # Simulate tool discovery
        print(f"  🔍 Discovering tools from {server_name}...")
        await asyncio.sleep(0.3)  # Simulate discovery time

        # Mock discovered tools
        if server_name == "file_operations":
            tools = ["read_file", "write_file", "list_directory", "create_directory"]
        else:  # web_search
            tools = ["search_web", "fetch_url", "extract_content"]

        print(f"  🔧 Discovered {len(tools)} tools: {', '.join(tools)}")

        # Simulate health check
        print(f"  💓 Health check for {server_name}...")
        await asyncio.sleep(0.2)
        print(f"  ✅ {server_name} is healthy")

    print("\n📊 MCP Client Statistics:")
    print(f"  - Total servers: {len(mcp_client.servers)}")
    print(f"  - Auto-start servers: {len([s for s in sample_servers if s.auto_start])}")
    print(f"  - Security levels: {set(s.security_level for s in sample_servers)}")


async def demonstrate_tool_discovery_and_registry():
    """
    Demonstrate tool discovery and registry management.
    """
    print("\n🔍 Demonstrating Tool Discovery and Registry")
    print("=" * 50)

    # Create tool registry
    tool_registry = ARKToolRegistry()
    print("✅ Tool Registry initialized")

    # Simulate tool discovery from MCP servers
    print("\n📡 Simulating Tool Discovery...")

    # Mock tools from different servers
    mock_tools = [
        {
            "name": "read_file",
            "description": "Read contents of a file",
            "server": "file_operations",
            "category": ToolCategory.FILESYSTEM,
            "permissions": [PermissionType.READ, PermissionType.FILESYSTEM],
            "parameters": {
                "file_path": {"type": "string", "required": True},
                "encoding": {"type": "string", "default": "utf-8"},
            },
        },
        {
            "name": "write_file",
            "description": "Write contents to a file",
            "server": "file_operations",
            "category": ToolCategory.FILESYSTEM,
            "permissions": [PermissionType.WRITE, PermissionType.FILESYSTEM],
            "parameters": {
                "file_path": {"type": "string", "required": True},
                "content": {"type": "string", "required": True},
                "encoding": {"type": "string", "default": "utf-8"},
            },
        },
        {
            "name": "search_web",
            "description": "Search the web for information",
            "server": "web_search",
            "category": ToolCategory.INFORMATION,
            "permissions": [PermissionType.NETWORK],
            "parameters": {
                "query": {"type": "string", "required": True},
                "max_results": {"type": "integer", "default": 10},
            },
        },
        {
            "name": "process_list",
            "description": "List running system processes",
            "server": "system_tools",
            "category": ToolCategory.SYSTEM,
            "permissions": [PermissionType.SYSTEM],
            "parameters": {"filter": {"type": "string", "required": False}},
        },
        {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "server": "math_tools",
            "category": ToolCategory.UTILITY,
            "permissions": [],
            "parameters": {"expression": {"type": "string", "required": True}},
        },
    ]

    # Register tools
    for tool_info in mock_tools:
        from core.tool_registry import ToolMetadata

        metadata = ToolMetadata(
            name=tool_info["name"],
            description=tool_info["description"],
            version="1.0.0",
            category=tool_info["category"],
            required_permissions=set(tool_info["permissions"]),
            parameters_schema=tool_info["parameters"],
            server_source=tool_info["server"],
        )

        success = await tool_registry.register_tool(metadata)
        status = "✅" if success else "❌"
        print(
            f"{status} Registered: {tool_info['name']} ({tool_info['category'].value})"
        )

    # Show registry statistics
    print("\n📊 Tool Registry Statistics:")
    stats = await tool_registry.get_registry_stats()
    print(f"  - Total tools: {stats['total_tools']}")
    print(f"  - By category: {stats['by_category']}")
    print(f"  - By status: {stats['by_status']}")
    print(f"  - By server: {stats['by_server']}")

    # Demonstrate tool filtering
    print("\n🔍 Tool Filtering Examples:")

    # Filter by category
    filesystem_tools = await tool_registry.discover_tools(
        category_filter=ToolCategory.FILESYSTEM
    )
    print(f"  - Filesystem tools: {len(filesystem_tools)} found")
    for tool in filesystem_tools:
        print(f"    • {tool.name}: {tool.description}")

    # Filter by permissions
    network_tools = await tool_registry.discover_tools(
        permission_filter={PermissionType.NETWORK}
    )
    print(f"  - Network tools: {len(network_tools)} found")
    for tool in network_tools:
        print(f"    • {tool.name}: {tool.description}")

    # Search by keyword
    search_results = await tool_registry.search_tools("file")
    print(f"  - Tools matching 'file': {len(search_results)} found")
    for tool in search_results:
        print(f"    • {tool.name}: {tool.description}")


async def demonstrate_tool_execution_with_security():
    """
    Demonstrate tool execution with security validation.
    """
    print("\n🔒 Demonstrating Tool Execution with Security")
    print("=" * 50)

    # Create security manager
    security_manager = ARKSecurityManager()
    print("✅ Security Manager initialized")

    # Create MCP client
    ARKMCPClient()

    # Simulate tool execution scenarios
    import time

    from core.security_manager import SecurityContext, ValidationRequest

    execution_scenarios = [
        {
            "tool": "read_file",
            "parameters": {"file_path": "/tmp/safe_file.txt"},
            "permissions": {PermissionType.READ, PermissionType.FILESYSTEM},
            "policy": "low",
            "expected": "ALLOWED",
        },
        {
            "tool": "write_file",
            "parameters": {"file_path": "/home/user/document.txt", "content": "Hello"},
            "permissions": {PermissionType.WRITE, PermissionType.FILESYSTEM},
            "policy": "medium",
            "expected": "ALLOWED",
        },
        {
            "tool": "search_web",
            "parameters": {"query": "Python programming", "max_results": 5},
            "permissions": {PermissionType.NETWORK},
            "policy": "low",
            "expected": "ALLOWED",
        },
        {
            "tool": "process_list",
            "parameters": {"filter": "python"},
            "permissions": {PermissionType.SYSTEM},
            "policy": "high",
            "expected": "REQUIRES_APPROVAL",
        },
        {
            "tool": "system_shutdown",
            "parameters": {},
            "permissions": {PermissionType.SYSTEM},
            "policy": "critical",
            "expected": "REQUIRES_APPROVAL",
        },
    ]

    print("\n🧪 Testing Tool Execution Scenarios:")

    for i, scenario in enumerate(execution_scenarios, 1):
        print(f"\n📝 Scenario {i}: {scenario['tool']}")
        print(f"   Parameters: {scenario['parameters']}")
        print(f"   Policy: {scenario['policy']}")

        # Create security context
        context = SecurityContext(
            user_id="demo_user",
            session_id="demo_session",
            tool_name=scenario["tool"],
            tool_version="1.0.0",
            execution_id=f"exec_{i}",
            timestamp=time.time(),
        )

        # Create validation request
        request = ValidationRequest(
            context=context,
            tool_parameters=scenario["parameters"],
            requested_permissions=scenario["permissions"],
        )

        # Validate tool execution
        response = await security_manager.validate_tool_execution(
            request, scenario["policy"]
        )

        print(f"   🔍 Validation Result: {response.result}")
        print(f"   💬 Message: {response.message}")

        # Check if result matches expectation
        expected_result = scenario["expected"]
        if expected_result in str(response.result):
            print(f"   ✅ Result matches expectation: {expected_result}")
        else:
            print(f"   ⚠️  Expected {expected_result}, got {response.result}")

        # Simulate tool execution if allowed
        if response.result.value == "allowed":
            print(f"   🚀 Executing tool: {scenario['tool']}")
            await asyncio.sleep(0.3)  # Simulate execution time
            print("   ✅ Tool execution completed successfully")
        elif response.result.value == "requires_approval":
            print("   ⏳ Tool execution pending approval")
        else:
            print("   ❌ Tool execution blocked")


async def demonstrate_configuration_persistence():
    """
    Demonstrate configuration loading and saving.
    """
    print("\n💾 Demonstrating Configuration Persistence")
    print("=" * 45)

    # Create server config manager
    config_manager = ARKServerConfigManager()
    print("✅ Server Config Manager initialized")

    # Create sample configuration
    sample_config = {
        "servers": {
            "file_ops": {
                "name": "file_operations",
                "description": "File system operations",
                "command": "python",
                "args": ["-m", "mcp_servers.filesystem"],
                "env": {"LOG_LEVEL": "INFO"},
                "capabilities": ["file_read", "file_write"],
                "security_level": "medium",
                "auto_start": True,
            },
            "web_search": {
                "name": "web_search",
                "description": "Web search capabilities",
                "command": "python",
                "args": ["-m", "mcp_servers.web"],
                "env": {"API_KEY": "demo"},
                "capabilities": ["search", "fetch"],
                "security_level": "low",
                "auto_start": False,
            },
        },
        "global_settings": {
            "max_concurrent_servers": 10,
            "default_timeout": 30,
            "log_level": "INFO",
        },
    }

    # Save configuration to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_config, f, indent=2)
        config_file = f.name

    try:
        print(f"\n💾 Saving configuration to: {config_file}")

        # Load configuration
        success = await config_manager.load_configuration(config_file)
        status = "✅" if success else "❌"
        print(f"{status} Configuration loaded")

        if success:
            # Show loaded configuration
            configs = await config_manager.list_configurations()
            print(f"📋 Loaded configurations: {len(configs)}")

            for config_name in configs:
                config = await config_manager.get_configuration(config_name)
                if config:
                    print(f"  - {config_name}: {config.description}")

        # Demonstrate configuration validation
        print("\n🔍 Validating configurations...")

        for config_name in ["file_ops", "web_search"]:
            config = await config_manager.get_configuration(config_name)
            if config:
                is_valid = await config_manager.validate_configuration(config)
                status = "✅" if is_valid else "❌"
                print(f"{status} {config_name}: {'Valid' if is_valid else 'Invalid'}")

        # Show health status
        print("\n💓 Health Status:")
        health_status = await config_manager.get_health_status()
        print(f"  - Total configurations: {health_status.get('total_configs', 0)}")
        print(f"  - Valid configurations: {health_status.get('valid_configs', 0)}")
        print(f"  - Auto-start enabled: {health_status.get('auto_start_count', 0)}")

    finally:
        # Clean up temporary file
        Path(config_file).unlink(missing_ok=True)
        print("🧹 Cleaned up temporary configuration file")


async def main():
    """
    Main demonstration function.
    """
    parser = argparse.ArgumentParser(description="MCP Integration Demonstration")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--demo",
        choices=["all", "servers", "tools", "security", "config"],
        default="all",
        help="Choose which demonstration to run",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("🌐 MCP Integration Demonstration Suite")
    print("=" * 50)

    try:
        if args.demo in ["all", "servers"]:
            await demonstrate_mcp_server_management()

        if args.demo in ["all", "tools"]:
            await demonstrate_tool_discovery_and_registry()

        if args.demo in ["all", "security"]:
            await demonstrate_tool_execution_with_security()

        if args.demo in ["all", "config"]:
            await demonstrate_configuration_persistence()

        print("\n🎉 All MCP demonstrations completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        logging.exception("MCP demonstration error")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
