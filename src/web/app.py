
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

class StreamHandler:
    """Handler for streaming agent thoughts and responses."""
    
    def __init__(self):
        self.thought_step = None
        self.final_message = None
        self.send_task = None
        
    def on_thought_start(self):
        """Called when a thought block starts."""
        # Create a new Step for the thought process
        # name="Thinking" will be the title of the step
        # type="tool" renders it as a collapsible step in Chainlit
        self.thought_step = cl.Step(name="Thinking", type="tool")
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
            # Only start streaming if we have substantial content or specific tokens
            # But we want to show *something* eventually.
            # Let's create the message but with content.
            # Avoid sending empty content initially
            if not token:
                return
                
            self.final_message = cl.Message(content=token)
            self.send_task = asyncio.create_task(self.final_message.send())
        else:
             asyncio.create_task(self.final_message.stream_token(token))

    def to_callbacks(self):
        """Convert handler methods to a callbacks dictionary."""
        return {
            "on_thought_start": self.on_thought_start,
            "on_thought_token": self.on_thought_token,
            "on_thought_end": self.on_thought_end,
            "on_token": self.on_token
        }

@cl.set_chat_profiles
async def chat_profiles(current_user: cl.User):
    return [
        cl.ChatProfile(name="Default", markdown_description="Standard Jarvis Agent"),
        cl.ChatProfile(name="Deep Research", markdown_description="Deep Research Agent for complex queries")
    ]

@cl.on_chat_start
async def start():
    """Initialize the agent when a new chat session starts."""
    
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
    session_id = cl.user_session.get("id")
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
        # cleaned_response = clean_llm_response(response)
        
        # If the handler didn't stream anything (e.g. error or short response), send the response directly
        if not handler.final_message:
             # 如果没有流式输出，且不使用 clean_llm_response，则直接发送原始 response
             await cl.Message(content=response).send()
        else:
             # Wait for the send task to complete
             if handler.send_task:
                 await handler.send_task
             # Ensure the final message is updated with any remaining content
             if handler.final_message.content != response:
                 handler.final_message.content = response
                 await handler.final_message.update()
             
    else:
        # Default Jarvis Agent
        # Show typing indicator (removed to prevent premature scrolling)
        # handler = StreamHandler()
        
        # Use cl.Step to wrap the thinking process if needed, or just let StreamHandler manage the output.
        # But Chainlit's default behavior is to scroll to bottom on new message.
        # We need to make sure we don't emit ANY message until we have content.
        
        handler = StreamHandler()
        response = await agent.process_message(message.content, callbacks=handler.to_callbacks())
        # Clean response before sending
        # cleaned_response = clean_llm_response(response)
        
        # If the handler didn't stream anything (e.g. error or short response), send the response directly
        if not handler.final_message:
            # 如果没有流式输出，且不使用 clean_llm_response，则直接发送原始 response
            await cl.Message(content=response).send()
        else:
            # Wait for the send task to complete
            if handler.send_task:
                await handler.send_task
             # Ensure the final message is updated with any remaining content
            if handler.final_message.content != response:
                handler.final_message.content = response
                await handler.final_message.update()


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
