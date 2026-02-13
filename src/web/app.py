
import chainlit as cl
import sys
import os

# Add project root to path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.jarvis_agent import JarvisAgent, JarvisConfig
from src.core.config.server import SimpleMCPServerConfig
from src.core.config.loader import load_llm_config as load_yaml_config
from src.core.llm.config import convert_to_client_config

@cl.on_chat_start
async def start():
    """Initialize the agent when a new chat session starts."""
    # Initialize configuration
    config = JarvisConfig()
    
    # Add default MCP servers (matching main.py behavior)
    config.mcp_servers = [
        SimpleMCPServerConfig(
            name="demo_server",
            command=["python", "-m", "demo_mcp_server"],
            description="Demonstration MCP server with basic tools"
        )
    ]
    
    # Initialize agent
    agent = JarvisAgent(config)
    
    # Load LLM Configuration from YAML
    try:
        # Default to production if not specified, to avoid Mock LLM in default run
        env = os.getenv("JARVIS_ENV", "production")
        yaml_config = load_yaml_config(environment=env)
        client_config = convert_to_client_config(yaml_config)
        
        # Inject configuration into LLM Manager
        # Note: We must do this before agent.initialize() is called
        agent.ark_engine.llm_manager._default_config = client_config
        
        print(f"Loaded LLM Config: {client_config.provider_name} ({client_config.model})")
        
    except Exception as e:
        print(f"Warning: Failed to load YAML configuration: {e}")
        # Continue with default configuration (which might be Mock)
    
    success = await agent.initialize()
    
    if not success:
        await cl.Message(content="❌ Failed to initialize Jarvis Agent. Please check the logs.").send()
        return

    await agent.start()
    
    # Store agent in user session
    cl.user_session.set("agent", agent)
    
    # Send welcome message
    welcome_message = await agent.start_conversation()
    await cl.Message(content=f"🤖 **Jarvis**: {welcome_message}").send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming user messages."""
    agent = cl.user_session.get("agent")
    
    if not agent:
        await cl.Message(content="❌ Agent not initialized. Please restart the session.").send()
        return
    
    # Show typing indicator
    async with cl.Step(name="Jarvis Processing") as step:
        step.input = message.content
        
        # Process message
        response = await agent.process_message(message.content)
        
        step.output = response

    # Send response
    await cl.Message(content=response).send()

@cl.on_stop
async def stop():
    """Cleanup when the session ends."""
    agent = cl.user_session.get("agent")
    if agent:
        await agent.stop()
