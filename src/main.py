#!/usr/bin/env python3
"""
Jarvis AI Agent - Main Entry Point

This is the main entry point for the Jarvis AI Agent application.
It provides a command-line interface for running the agent and
demonstrates basic usage of the ARK-powered conversational system.
"""

import asyncio
import argparse
import logging
import sys
import signal
from typing import Optional

from .jarvis_agent import JarvisAgent, JarvisConfig
from .core.config.server import SimpleMCPServerConfig


class JarvisApp:
    """
    Main application class for Jarvis AI Agent.
    
    Handles command-line interface, signal handling, and
    the main conversation loop.
    """
    
    def __init__(self):
        """Initialize the Jarvis application."""
        self.agent: Optional[JarvisAgent] = None
        self.running = False
        self.logger = logging.getLogger('jarvis.app')
    
    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def parse_arguments(self) -> argparse.Namespace:
        """
        Parse command-line arguments.
        
        Returns:
            Parsed arguments namespace
        """
        parser = argparse.ArgumentParser(
            description="Jarvis AI Agent - ARK-Powered Conversational Assistant",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python src/main.py                    # Start with default settings
  python src/main.py --log-level DEBUG # Start with debug logging
  python src/main.py --interactive     # Start interactive mode
  python src/main.py --demo            # Run demonstration mode
            """
        )
        
        parser.add_argument(
            '--log-level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            default='INFO',
            help='Set logging level (default: INFO)'
        )
        
        parser.add_argument(
            '--interactive',
            action='store_true',
            help='Run in interactive conversation mode'
        )
        
        parser.add_argument(
            '--demo',
            action='store_true',
            help='Run demonstration mode with sample interactions'
        )
        
        parser.add_argument(
            '--config-file',
            type=str,
            help='Path to configuration file (JSON format)'
        )
        
        parser.add_argument(
            '--max-history',
            type=int,
            default=100,
            help='Maximum conversation history length (default: 100)'
        )
        
        parser.add_argument(
            '--confidence-threshold',
            type=float,
            default=0.7,
            help='Confidence threshold for tool execution (default: 0.7)'
        )
        
        parser.add_argument(
            '--version',
            action='version',
            version='Jarvis AI Agent v1.0.0'
        )
        
        return parser.parse_args()
    
    def load_config(self, args: argparse.Namespace) -> JarvisConfig:
        """
        Load configuration from arguments and optional config file.
        
        Args:
            args: Parsed command-line arguments
            
        Returns:
            Jarvis configuration object
        """
        config = JarvisConfig(
            log_level=args.log_level,
            max_conversation_history=args.max_history,
            confidence_threshold=args.confidence_threshold
        )
        
        # Load from config file if provided
        if args.config_file:
            try:
                import json
                with open(args.config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Update config with file data
                for key, value in config_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                
                self.logger.info(f"Loaded configuration from {args.config_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load config file: {e}")
        
        # Add default MCP servers
        config.mcp_servers = [
            SimpleMCPServerConfig(
                name="demo_server",
                command=["python", "-m", "demo_mcp_server"],
                description="Demonstration MCP server with basic tools"
            )
        ]
        
        return config
    
    async def run_interactive_mode(self) -> None:
        """Run the agent in interactive conversation mode."""
        print("\n" + "="*60)
        print("🤖 Jarvis AI Agent - Interactive Mode")
        print("="*60)
        print("Type 'quit', 'exit', or 'bye' to end the conversation")
        print("Type 'help' for available commands")
        print("Type 'status' to see agent status")
        print("-"*60)
        
        # Start conversation
        welcome = await self.agent.start_conversation()
        print(f"\n🤖 Jarvis: {welcome}\n")
        
        self.running = True
        
        while self.running:
            try:
                # Get user input
                user_input = input("👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    goodbye = await self.agent.end_conversation()
                    print(f"\n🤖 Jarvis: {goodbye}\n")
                    break
                
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                elif user_input.lower() == 'status':
                    await self._show_status()
                    continue
                
                elif user_input.lower() == 'tools':
                    self._show_tools()
                    continue
                
                elif user_input.lower() == 'export':
                    self._export_conversation()
                    continue
                
                # Process message through agent
                response = await self.agent.process_message(user_input)
                print(f"\n🤖 Jarvis: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nReceived interrupt signal...")
                break
            except EOFError:
                print("\n\nEnd of input received...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                self.logger.error(f"Interactive mode error: {e}")
    
    async def run_demo_mode(self) -> None:
        """Run the agent in demonstration mode."""
        print("\n" + "="*60)
        print("🚀 Jarvis AI Agent - Demonstration Mode")
        print("="*60)
        
        # Start conversation
        welcome = await self.agent.start_conversation()
        print(f"\n🤖 Jarvis: {welcome}\n")
        
        # Demo interactions
        demo_messages = [
            "Hello! What can you help me with?",
            "What tools do you have available?",
            "Can you tell me about your capabilities?",
            "What's the current status of your systems?",
            "Thank you for the demonstration!"
        ]
        
        for i, message in enumerate(demo_messages, 1):
            print(f"👤 Demo User: {message}")
            
            response = await self.agent.process_message(message)
            print(f"🤖 Jarvis: {response}\n")
            
            # Add delay between messages for readability
            if i < len(demo_messages):
                await asyncio.sleep(2)
        
        # End conversation
        goodbye = await self.agent.end_conversation()
        print(f"🤖 Jarvis: {goodbye}\n")
        
        print("="*60)
        print("🎉 Demonstration completed!")
        print("="*60)
    
    def _show_help(self) -> None:
        """Show help information."""
        help_text = """
Available Commands:
  help     - Show this help message
  status   - Show agent and system status
  tools    - List available tools
  export   - Export conversation history
  quit     - End conversation and exit
  exit     - End conversation and exit
  bye      - End conversation and exit

You can also ask me questions or request assistance with various tasks!
        """
        print(help_text)
    
    async def _show_status(self) -> None:
        """Show agent status."""
        try:
            status = self.agent.get_status()
            health = await self.agent.health_check()
            
            print("\n📊 Agent Status:")
            print(f"  Name: {status['agent']['name']}")
            print(f"  Version: {status['agent']['version']}")
            print(f"  Running: {status['agent']['is_running']}")
            print(f"  Conversation Active: {status['agent']['conversation_active']}")
            print(f"  Overall Health: {health['overall']}")
            print(f"  Available Tools: {len(self.agent.get_available_tools())}")
            print(f"  MCP Servers: {status['configuration']['mcp_servers_count']}")
            
        except Exception as e:
            print(f"❌ Error getting status: {e}")
    
    def _show_tools(self) -> None:
        """Show available tools."""
        try:
            tools = self.agent.get_available_tools()
            usage_stats = self.agent.get_tool_usage_stats()
            
            print("\n🔧 Available Tools:")
            if tools:
                for tool in tools:
                    usage_count = usage_stats.get(tool, 0)
                    print(f"  • {tool} (used {usage_count} times)")
            else:
                print("  No tools currently available")
            
        except Exception as e:
            print(f"❌ Error getting tools: {e}")
    
    def _export_conversation(self) -> None:
        """Export conversation history."""
        try:
            conversation = self.agent.get_conversation_export("text")
            print("\n📄 Conversation Export:")
            print("-" * 40)
            print(conversation)
            print("-" * 40)
            
        except Exception as e:
            print(f"❌ Error exporting conversation: {e}")
    
    async def run(self) -> int:
        """
        Main application entry point.
        
        Returns:
            Exit code (0 for success, 1 for error)
        """
        try:
            # Parse arguments
            args = self.parse_arguments()
            
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Load configuration
            config = self.load_config(args)
            
            # Create and initialize agent
            self.agent = JarvisAgent(config)
            
            print(f"🚀 Starting Jarvis AI Agent v{config.version}...")
            
            # Initialize agent
            success = await self.agent.initialize()
            if not success:
                print("❌ Failed to initialize Jarvis Agent")
                return 1
            
            # Start agent
            await self.agent.start()
            
            print("✅ Jarvis Agent started successfully!")
            
            # Run in appropriate mode
            if args.demo:
                await self.run_demo_mode()
            elif args.interactive:
                await self.run_interactive_mode()
            else:
                # Default: show status and available commands
                print("\n📋 Jarvis Agent is ready!")
                print("Use --interactive for conversation mode")
                print("Use --demo for demonstration mode")
                print("Use --help for more options")
                
                status = self.agent.get_status()
                print(f"\n📊 Status: {status['agent']['name']} v{status['agent']['version']}")
                print(f"🔧 Available Tools: {len(self.agent.get_available_tools())}")
                
                # Perform health check
                health = await self.agent.health_check()
                print(f"💚 Health: {health['overall']}")
            
            return 0
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            return 0
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            self.logger.error(f"Fatal error in main: {e}")
            return 1
        finally:
            # Cleanup
            if self.agent:
                try:
                    await self.agent.stop()
                except Exception as e:
                    self.logger.error(f"Error during cleanup: {e}")


async def main() -> int:
    """
    Async main function.
    
    Returns:
        Exit code
    """
    app = JarvisApp()
    return await app.run()


def cli_main() -> None:
    """
    Command-line interface main function.
    
    This function is the entry point when running the module directly
    or through setuptools console scripts.
    """
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli_main()