
import chainlit as cl
import sys
import os
import asyncio

# Add project root to path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.jarvis_agent import JarvisAgent, JarvisConfig
from src.core.agent.deep_research import DeepResearchAgent
from src.core.config.server import SimpleMCPServerConfig
from src.core.config.loader import load_llm_config as load_yaml_config
from src.core.llm.config import convert_to_client_config
from src.core.llm.client import LLMManager
from src.core.ark.utils import clean_llm_response

class StreamHandler:
    """Handler for streaming agent thoughts and responses."""
    
    def __init__(self):
        self.thought_step = None
        self.final_message = None
        
    def on_thought_start(self):
        """Called when a thought block starts."""
        self.thought_step = cl.Step(name="Thinking")
        asyncio.create_task(self.thought_step.send())
        
    def on_thought_token(self, token: str):
        """Called when a token is generated within a thought block."""
        if self.thought_step:
            asyncio.create_task(self.thought_step.stream_token(token))
            
    def on_thought_end(self):
        """Called when a thought block ends."""
        if self.thought_step:
            asyncio.create_task(self.thought_step.update())
            self.thought_step = None
            
    def on_token(self, token: str):
        """Called when a token is generated for the final answer."""
        if not self.final_message:
            self.final_message = cl.Message(content="")
            asyncio.create_task(self.final_message.send())
        asyncio.create_task(self.final_message.stream_token(token))

    def to_callbacks(self):
        """Convert handler methods to a callbacks dictionary."""
        return {
            "on_thought_start": self.on_thought_start,
            "on_thought_token": self.on_thought_token,
            "on_thought_end": self.on_thought_end,
            "on_token": self.on_token
        }

@cl.on_chat_start
async def start():
    """Initialize the agent when a new chat session starts."""
    
    # Set chat profiles
    cl.set_chat_profiles([
        cl.ChatProfile(name="Default", markdown_description="Standard Jarvis Agent"),
        cl.ChatProfile(name="Deep Research", markdown_description="Deep Research Agent for complex queries")
    ])
    
    chat_profile = cl.user_session.get("chat_profile")
    
    # Load LLM Configuration from YAML
    client_config = None
    try:
        # Default to production if not specified, to avoid Mock LLM in default run
        env = os.getenv("JARVIS_ENV", "production")
        yaml_config = load_yaml_config(environment=env)
        client_config = convert_to_client_config(yaml_config)
        print(f"Loaded LLM Config: {client_config.provider_name} ({client_config.model})")
    except Exception as e:
        print(f"Warning: Failed to load YAML configuration: {e}")

    if chat_profile == "Deep Research":
        # Initialize Deep Research Agent
        # DeepResearchAgent takes a config dict
        agent_config = {}
        agent = DeepResearchAgent(agent_config)
        
        if client_config:
             agent.llm_manager = LLMManager(client_config)
             # Update tools with the new LLM manager to ensure they use the configured client
             if hasattr(agent, 'tools') and hasattr(agent.tools, 'llm_manager'):
                 agent.tools.llm_manager = agent.llm_manager

        
        # Initialize agent
        if hasattr(agent, "initialize"):
             if asyncio.iscoroutinefunction(agent.initialize):
                 await agent.initialize()
             else:
                 agent.initialize()
        
        cl.user_session.set("agent", agent)
        cl.user_session.set("agent_type", "deep_research")
        
        await cl.Message(content=f"🧠 **Deep Research Agent** initialized. Ask me anything!").send()

    else:
        # Initialize Jarvis Agent (Default)
        config = JarvisConfig()
        
        # Add default MCP servers (matching main.py behavior)
        config.mcp_servers = [
            SimpleMCPServerConfig(
                name="demo_server",
                command=["python", "-m", "demo_mcp_server"],
                description="Demonstration MCP server with basic tools"
            )
        ]
        
        agent = JarvisAgent(config)
        
        # Inject configuration into LLM Manager
        if client_config:
            agent.ark_engine.llm_manager._default_config = client_config
        
        success = await agent.initialize()
        
        if not success:
            await cl.Message(content="❌ Failed to initialize Jarvis Agent. Please check the logs.").send()
            return

        await agent.start()
        
        # Store agent in user session
        cl.user_session.set("agent", agent)
        cl.user_session.set("agent_type", "jarvis")
        
        # Send welcome message
        welcome_message = await agent.start_conversation()
        await cl.Message(content=f"🤖 **Jarvis**: {welcome_message}").send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming user messages."""
    agent = cl.user_session.get("agent")
    agent_type = cl.user_session.get("agent_type")
    
    if not agent:
        await cl.Message(content="❌ Agent not initialized. Please restart the session.").send()
        return
    
    if agent_type == "deep_research":
        handler = StreamHandler()
        # DeepResearchAgent uses process_input with callbacks
        response = await agent.process_input(message.content, callbacks=handler.to_callbacks())
        
        # Clean response before sending
        cleaned_response = clean_llm_response(response)
        
        # If the handler didn't stream anything (e.g. error or short response), send the response directly
        if not handler.final_message:
             await cl.Message(content=cleaned_response).send()
        else:
             # Ensure the final message is updated with any remaining content
             # Note: StreamHandler might have already streamed tokens. 
             # If we clean it now, we might replace content.
             # However, StreamHandler currently streams 'on_token'.
             # If 'on_token' received raw chunks, the user might have seen raw output.
             # But 'on_token' usually comes from the LLM stream.
             # The 'response' from process_input is the final aggregated string.
             # Ideally we should clean it.
             # Let's update the final message with the cleaned content to be sure.
             handler.final_message.content = cleaned_response
             await handler.final_message.update()
             
    else:
        # Default Jarvis Agent
        # Show typing indicator
        async with cl.Step(name="Jarvis Processing") as step:
            step.input = message.content
            
            # Process message
            response = await agent.process_message(message.content)
            
            # Clean response
            cleaned_response = clean_llm_response(response)
            
            step.output = cleaned_response


        # Send response
        await cl.Message(content=cleaned_response).send()

@cl.on_stop
async def stop():
    """Cleanup when the session ends."""
    agent = cl.user_session.get("agent")
    if agent:
        if hasattr(agent, "stop"):
            if asyncio.iscoroutinefunction(agent.stop):
                await agent.stop()
            else:
                agent.stop()
