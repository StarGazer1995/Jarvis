import sys
import os
import asyncio
from dotenv import load_dotenv
import chainlit as cl
import logging
from chainlit.input_widget import TextInput

# Load environment variables from .env file
load_dotenv()

# Add project root to path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.web.data_layer import get_data_layer, get_database_url
from src.web.auth import auth_callback
from src.web.agent_factory import create_agent
from src.web.history_manager import restore_agent_history, cleanup_system_messages
from src.web.utils import clean_llm_response
from src.web.settings_repository import UserSettingsRepository, TOKEN_FIELDS

# Configure logger
logger = logging.getLogger(__name__)
_settings_repository = None


def _get_user_id() -> str:
    user = cl.user_session.get("user")
    if not user:
        return "anonymous"
    return getattr(user, "id", None) or getattr(user, "identifier", "anonymous")


def _extract_user_settings(raw_settings: dict | None) -> dict:
    settings = raw_settings or {}
    return {field: settings.get(field, "") for field in TOKEN_FIELDS}


def _merge_user_settings(current: dict | None, updates: dict | None) -> dict:
    merged = _extract_user_settings(current)
    for field in TOKEN_FIELDS:
        value = (updates or {}).get(field, "")
        if isinstance(value, str) and value.strip():
            merged[field] = value
    return merged


def _get_settings_repository() -> UserSettingsRepository | None:
    global _settings_repository
    if _settings_repository is not None:
        return _settings_repository

    database_url = get_database_url() or "sqlite+aiosqlite:///./.chainlit/chainlit.db"
    try:
        repository = UserSettingsRepository(database_url)
        repository.init_table()
        _settings_repository = repository
        return _settings_repository
    except Exception as exc:
        logger.warning(f"Failed to initialize settings repository: {exc}")
        return None


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
            "on_token": self.on_token,
        }


@cl.set_chat_profiles
async def chat_profiles(current_user: cl.User):
    return [
        cl.ChatProfile(name="Default", markdown_description="Standard Jarvis Agent"),
        cl.ChatProfile(
            name="Deep Research",
            markdown_description="Deep Research Agent for complex queries",
        ),
    ]


@cl.on_chat_resume
async def on_chat_resume(thread):
    """
    Called when a user resumes a chat session from the history.
    """
    env = os.getenv("JARVIS_ENV", "production")

    repository = _get_settings_repository()
    user_settings = {}
    if repository:
        user_settings = repository.load_settings(_get_user_id())

    agent = await create_agent("jarvis", env, user_settings=user_settings)

    if not agent:
        await cl.Message(content="❌ Failed to restore Jarvis Agent session.").send()
        return

    # Restore conversation history
    restored_count = restore_agent_history(agent, thread)

    if restored_count > 0:
        # Cleanup old restore messages
        thread_id = getattr(
            thread, "id", thread.get("id") if isinstance(thread, dict) else None
        )
        await cleanup_system_messages(thread_id)

        await cl.Message(
            content=f"🔄 **Context Restored**: Loaded {restored_count} messages from history.",
            author="System",
            parent_id=None,
        ).send()

    # Check if LLM is enabled in the engine
    if hasattr(agent, "ark_engine") and hasattr(agent.ark_engine, "llm_enabled"):
        if not agent.ark_engine.llm_enabled:
            await cl.Message(
                content="⚠️ **Warning**: LLM initialization failed (likely due to API rate limits). The agent may not be able to respond.",
                author="System",
                parent_id=None,
            ).send()

    # Store agent in user session
    cl.user_session.set("agent", agent)
    cl.user_session.set("agent_type", "jarvis")
    cl.user_session.set("user_settings", user_settings)


@cl.on_chat_start
async def start():
    """Initialize the agent when a new chat session starts."""
    chat_profile = cl.user_session.get("chat_profile")
    env = os.getenv("JARVIS_ENV", "production")

    agent_type = "deep_research" if chat_profile == "Deep Research" else "jarvis"
    repository = _get_settings_repository()
    user_id = _get_user_id()
    user_settings = {}
    if repository:
        user_settings = repository.load_settings(user_id)

    agent = await create_agent(agent_type, env, user_settings=user_settings)

    if not agent:
        await cl.Message(
            content="❌ Failed to initialize Agent. Please check the logs."
        ).send()
        return

    cl.user_session.set("agent", agent)
    cl.user_session.set("agent_type", agent_type)
    cl.user_session.set("user_settings", user_settings)

    await cl.ChatSettings(
        [
            TextInput(
                id="openai_api_key",
                label="OpenAI API Key",
                initial="",
                placeholder="已保存则留空保持不变，输入新值可覆盖",
            ),
            TextInput(
                id="tavily_api_key",
                label="Tavily API Key",
                initial="",
                placeholder="已保存则留空保持不变，输入新值可覆盖",
            ),
            TextInput(
                id="confluence_page_token",
                label="Confluence Page Token",
                initial="",
                placeholder="已保存则留空保持不变，输入新值可覆盖",
            ),
            TextInput(
                id="beacon_model_token",
                label="Beacon Model Token",
                initial="",
                placeholder="已保存则留空保持不变，输入新值可覆盖",
            ),
        ]
    ).send()

    if agent_type == "deep_research":
        await cl.Message(
            content=f"🧠 **Deep Research Agent** initialized. Ask me anything!"
        ).send()
    else:
        # Send welcome message
        welcome_message = await agent.start_conversation()
        await cl.Message(
            content=f"🤖 **Jarvis**: {welcome_message}", parent_id=None
        ).send()

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
        await cl.Message(
            content="❌ Agent not initialized. Please restart the session."
        ).send()
        return

    handler = StreamHandler()
    response = ""

    if agent_type == "deep_research":
        # DeepResearchAgent uses process_input with callbacks
        response = await agent.process_input(
            message.content, callbacks=handler.to_callbacks()
        )
    else:
        # Default Jarvis Agent
        response = await agent.process_message(
            message.content, callbacks=handler.to_callbacks()
        )

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
        context_manager = (
            agent.ark_engine.context_manager if hasattr(agent, "ark_engine") else None
        )

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
                    await dl.update_thread(
                        thread_id=cl.context.session.thread_id, user_id=user.id
                    )
                    logger.info(
                        f"Thread {cl.context.session.thread_id} renamed to '{title}' and associated with user {user.id}"
                    )

    except Exception as e:
        logger.warning(f"Failed to rename/associate thread: {e}")


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


@cl.on_settings_update
async def on_settings_update(settings):
    current_settings = cl.user_session.get("user_settings") or {}
    sanitized_settings = _merge_user_settings(current_settings, settings)
    cl.user_session.set("user_settings", sanitized_settings)
    repository = _get_settings_repository()
    if repository:
        repository.save_settings(_get_user_id(), sanitized_settings)
        await cl.Message(content="✅ Settings saved.", parent_id=None).send()
    else:
        await cl.Message(content="⚠️ Settings not persisted.", parent_id=None).send()
