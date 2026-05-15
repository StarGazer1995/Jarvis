import logging

from sqlalchemy import text

from src.jarvis_agent import JarvisAgent
from src.web.data_layer import get_data_layer

logger = logging.getLogger(__name__)


def restore_agent_history(agent: JarvisAgent, thread):
    """
    Restore conversation history from Chainlit thread steps into the agent's memory.

    Args:
        agent: The initialized JarvisAgent instance.
        thread: The Chainlit thread object or dictionary.
    """
    if not thread:
        return 0

    steps = []
    thread_id = None

    if isinstance(thread, dict):
        steps = thread.get("steps", [])
        thread_id = thread.get("id")
    elif hasattr(thread, "steps"):
        steps = thread.steps
        thread_id = getattr(thread, "id", None)

    if not steps:
        return 0

    # Sort steps by createdAt
    try:
        steps = sorted(
            steps,
            key=lambda x: (
                x.get("createdAt", "")
                if isinstance(x, dict)
                else getattr(x, "createdAt", "")
            ),
        )
    except Exception as e:
        logger.warning(f"Failed to sort steps: {e}")

    current_user_msg = None
    restored_count = 0

    for step in steps:
        if isinstance(step, dict):
            step_type = step.get("type")
            step_output = step.get("output", "")
            step_id = step.get("id")
        else:
            step_type = getattr(step, "type", None)
            step_output = getattr(step, "output", "")
            step_id = getattr(step, "id", None)

        if step_type == "user_message":
            current_user_msg = step_output
        elif step_type == "assistant_message":
            if current_user_msg:
                agent_response = step_output
                # Add exchange to context
                if hasattr(agent, "ark_engine") and hasattr(
                    agent.ark_engine, "context_manager"
                ):
                    agent.ark_engine.context_manager.add_exchange(
                        user_input=current_user_msg,
                        agent_response=agent_response,
                        intent="restored_history",
                        metadata={"restored": True, "step_id": step_id},
                    )
                    restored_count += 1
                current_user_msg = None  # Reset for next turn

    logger.info(
        f"Restored {restored_count} conversation turns from thread {thread_id}."
    )
    return restored_count


async def cleanup_system_messages(thread_id: str):
    """
    Remove old 'Context Restored' system messages from the database for a given thread.
    """
    if not thread_id:
        return

    try:
        dl = get_data_layer()
        if dl and hasattr(dl, "engine"):
            async with dl.engine.begin() as conn:
                await conn.execute(
                    text(
                        "DELETE FROM steps WHERE threadId = :tid AND name = 'System' AND output LIKE '🔄 **Context Restored**%'"
                    ),
                    {"tid": thread_id},
                )
            logger.info(f"Cleaned up old system messages for thread {thread_id}.")
    except Exception as e:
        logger.warning(f"Failed to cleanup old restore messages: {e}")
