
import sys
import os
import asyncio
from dotenv import load_dotenv
import chainlit as cl
from sqlalchemy import text

# Load environment variables from .env file
load_dotenv()

# Add project root to path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.jarvis_agent import JarvisAgent, JarvisConfig
from src.core.agent.deep_research import DeepResearchAgent
from src.core.config.server import SimpleMCPServerConfig
from src.core.config.loader import load_llm_config as load_yaml_config
from src.core.llm.config import convert_to_client_config
from src.core.llm.client import LLMManager
from src.web.data_layer import get_data_layer
from src.web.auth import auth_callback

import json
import re

def clean_llm_response(response: str) -> str:
    """
    Clean LLM response by attempting to parse JSON and extract 'content'.
    Handles cases where response is wrapped in markdown code blocks.
    """
    if not response:
        return ""
        
    cleaned = response.strip()
    
    # Remove markdown code blocks if present
    # e.g. ```json ... ```
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1)
    
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "content" in parsed:
            return parsed["content"]
    except json.JSONDecodeError:
        pass
        
    return response

# Configure Chainlit authentication
@cl.password_auth_callback
def auth(username, password):
    return auth_callback(username, password)

# Configure Chainlit data layer
@cl.data_layer
def db():
    return get_data_layer()

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
        self.thought_step = cl.Step(name="Thinking", type="tool", parent_id=None)
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
                
            self.final_message = cl.Message(content=token, parent_id=None)
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

@cl.on_chat_resume
async def on_chat_resume(thread):
    """
    Called when a user resumes a chat session from the history.
    """
    # Initialize the agent (similar to on_chat_start)
    try:
        env = os.getenv("JARVIS_ENV", "production")
        yaml_config = load_yaml_config(environment=env)
        client_config = convert_to_client_config(yaml_config)
    except Exception as e:
        print(f"Warning: Failed to load YAML configuration: {e}")
        client_config = None

    # Initialize Jarvis Agent (Default)
    # TODO: In the future, store agent_type in thread metadata to restore correct agent
    config = JarvisConfig()
    
    # Add default MCP servers
    config.mcp_servers = [
        SimpleMCPServerConfig(
            name="demo_server",
            command=["python", "-m", "demo_mcp_server"],
            description="Demonstration MCP server with basic tools"
        )
    ]
    
    agent = JarvisAgent(config)
    
    if client_config:
        agent.ark_engine.llm_manager._default_config = client_config
    
    success = await agent.initialize()
    if not success:
        await cl.Message(content="❌ Failed to restore Jarvis Agent session.").send()
        return

    await agent.start()

    # Restore conversation history
    # We need to fetch steps from the thread and populate the agent's memory
    # Chainlit passes the 'thread' object which can be a dict or object
    
    steps = []
    if isinstance(thread, dict):
        steps = thread.get('steps', [])
    elif hasattr(thread, 'steps'):
        steps = thread.steps
        
    if steps:
        # Sort steps by createdAt just in case
        steps = sorted(steps, key=lambda x: x.get('createdAt', ''))
        
        # Filter only user and assistant messages
        # Chainlit steps are flat list. We need to iterate and reconstruct conversation.
        # User message usually comes as a step with type="user_message"
        # Assistant response is a step with type="assistant_message"
        
        current_user_msg = None
        
        for step in steps:
            step_type = step.get('type')
            step_name = step.get('name')
            step_output = step.get('output', '')
            
            # Chainlit 1.x uses different types for messages
            # user_message: The message sent by the user
            # assistant_message: The message sent by the assistant (final answer)
            # run: Intermediate steps (thinking, tools)
            
            if step_type == 'user_message':
                current_user_msg = step_output
            elif step_type == 'assistant_message':
                if current_user_msg:
                    agent_response = step_output
                    # Add exchange to context
                    agent.ark_engine.context_manager.add_exchange(
                        user_input=current_user_msg,
                        agent_response=agent_response,
                        intent="restored_history",
                        metadata={"restored": True, "step_id": step.get('id')}
                    )
                    current_user_msg = None # Reset for next turn
        
        print(f"Restored {len(agent.ark_engine.context_manager.conversation_history)} conversation turns.")
        
        if len(agent.ark_engine.context_manager.conversation_history) > 0:
            # Cleanup old restore messages before sending a new one
            try:
                dl = get_data_layer()
                if dl and hasattr(dl, 'engine'):
                    thread_id = thread.get('id') if isinstance(thread, dict) else thread.id
                    async with dl.engine.begin() as conn:
                        await conn.execute(
                            text("DELETE FROM steps WHERE threadId = :tid AND name = 'System' AND output LIKE '🔄 **Context Restored**%'"),
                            {"tid": thread_id}
                        )
            except Exception as e:
                print(f"Warning: Failed to cleanup old restore messages: {e}")

            await cl.Message(
                content=f"🔄 **Context Restored**: Loaded {len(agent.ark_engine.context_manager.conversation_history)} messages from history.",
                author="System",
                parent_id=None
            ).send()

    # Check if LLM is enabled in the engine
    if hasattr(agent, 'ark_engine') and hasattr(agent.ark_engine, 'llm_enabled'):
        if not agent.ark_engine.llm_enabled:
            await cl.Message(
                content="⚠️ **Warning**: LLM initialization failed (likely due to API rate limits). The agent may not be able to respond.",
                author="System",
                parent_id=None
            ).send()
    
    # Store agent in user session
    cl.user_session.set("agent", agent)
    cl.user_session.set("agent_type", "jarvis")

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
        await cl.Message(content=f"🤖 **Jarvis**: {welcome_message}", parent_id=None).send()
        
    # Explicitly update thread with user info to ensure persistence
    # user = cl.user_session.get("user")
    # if user:
    #     try:
    #         thread_id = cl.context.session.thread_id
    #         dl = get_data_layer()
    #         if dl:
    #             await dl.update_thread(thread_id=thread_id, user_id=user.id)
    #     except Exception as e:
    #         print(f"Warning: Failed to manually update thread: {e}")

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming user messages."""
    session_id = cl.user_session.get("id")
    agent = cl.user_session.get("agent")
    agent_type = cl.user_session.get("agent_type")
    
    if not agent:
        await cl.Message(content="❌ Agent not initialized. Please restart the session.").send()
        return
    
    handler = StreamHandler()
    response = ""
    
    if agent_type == "deep_research":
        # DeepResearchAgent uses process_input with callbacks
        response = await agent.process_input(message.content, callbacks=handler.to_callbacks())
    else:
        # Default Jarvis Agent
        response = await agent.process_message(message.content, callbacks=handler.to_callbacks())

    # Clean the response (extract content if JSON)
    cleaned_response = clean_llm_response(response)

    # Handle the response sending
    if not handler.final_message:
        # If no streaming happened, send the full response
        await cl.Message(content=cleaned_response, parent_id=None).send()
    else:
        # If streaming happened, ensure final content matches cleaned response
        if handler.send_task:
            await handler.send_task
            
        # Update content if it differs (e.g. streaming missed something or raw vs clean)
        # We prefer the cleaned response if it's different from what was streamed
        if handler.final_message.content != cleaned_response:
            # Only update if the difference is significant (not just whitespace)
            if handler.final_message.content.strip() != cleaned_response.strip():
                handler.final_message.content = cleaned_response
                await handler.final_message.update()

    # Post-message processing: Check if we need to rename the thread and associate with user
    # This ensures only non-empty, renamed threads are saved to history
    try:
        context_manager = agent.ark_engine.context_manager if hasattr(agent, 'ark_engine') else None
        
        # Only rename if we have a short history (e.g. first turn)
        # Note: context_manager.conversation_history stores exchanges. 
        # 1 exchange = 1 user message + 1 assistant response.
        if context_manager and len(context_manager.conversation_history) == 1:
            user = cl.user_session.get("user")
            if user:
                # Generate a simple title based on the first user message
                # We can try to use LLM to generate a better title if possible, 
                # but for now let's use a truncated version of the user message.
                first_exchange = context_manager.conversation_history[0]
                user_input = first_exchange.user_input
                
                # Simple truncation
                title = user_input[:30] + "..." if len(user_input) > 30 else user_input
                
                # Rename thread
                await cl.rename_thread(cl.context.session.thread_id, title)
                
                # Associate with user (Persist to history)
                dl = get_data_layer()
                if dl:
                    await dl.update_thread(thread_id=cl.context.session.thread_id, user_id=user.id)
                    print(f"Thread {cl.context.session.thread_id} renamed to '{title}' and associated with user {user.id}")
                    
    except Exception as e:
        print(f"Warning: Failed to rename/associate thread: {e}")


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
