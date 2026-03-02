import sys
from unittest.mock import MagicMock, patch, AsyncMock
import pytest

# 1. Create a mock for chainlit module
mock_cl = MagicMock()
mock_cl.__path__ = []  # Mark as package
mock_cl.password_auth_callback = lambda x: x
mock_cl.data_layer = lambda x: x
mock_cl.set_chat_profiles = lambda x: x
mock_cl.on_chat_resume = lambda x: x
mock_cl.on_chat_start = lambda x: x
mock_cl.on_message = lambda x: x
mock_cl.on_stop = lambda x: x
mock_cl.Step = MagicMock()
mock_cl.Message = MagicMock()
# Make Message().send() awaitable
mock_cl.Message.return_value.send = AsyncMock()
mock_cl.Message.return_value.update = AsyncMock()
mock_cl.Message.return_value.stream_token = AsyncMock()

mock_cl.user_session = MagicMock()
mock_cl.context = MagicMock()
mock_cl.rename_thread = AsyncMock()

# Mock submodules
mock_cl_data = MagicMock()
mock_cl_data_sa = MagicMock()
mock_cl_data.sql_alchemy = mock_cl_data_sa

# 2. Patch sys.modules to inject the mock
# We need to ensure we patch it BEFORE importing app
with patch.dict(
    sys.modules,
    {
        "chainlit": mock_cl,
        "chainlit.data": mock_cl_data,
        "chainlit.data.sql_alchemy": mock_cl_data_sa,
    },
):
    import src.web.app as app_module
    from src.web.app import start, on_chat_resume, main, stop


@pytest.mark.asyncio
async def test_start():
    # Use patch.object to ensure we patch the exact module object
    with patch.object(
        app_module, "create_agent", new_callable=AsyncMock
    ) as mock_create:
        mock_agent = MagicMock()
        mock_agent.start_conversation = AsyncMock(return_value="Welcome")
        mock_create.return_value = mock_agent

        # Mock session
        mock_cl.user_session.get.return_value = "Default"

        await start()

        mock_create.assert_called_once()
        mock_cl.user_session.set.assert_any_call("agent", mock_agent)


@pytest.mark.asyncio
async def test_on_chat_resume():
    with (
        patch.object(app_module, "create_agent", new_callable=AsyncMock) as mock_create,
        patch.object(app_module, "restore_agent_history") as mock_restore,
        patch.object(
            app_module, "cleanup_system_messages", new_callable=AsyncMock
        ) as mock_cleanup,
    ):
        mock_agent = MagicMock()
        mock_create.return_value = mock_agent
        mock_restore.return_value = 5  # 5 messages restored

        thread = MagicMock()
        thread.id = "t1"

        await on_chat_resume(thread)

        mock_create.assert_called_once()
        mock_restore.assert_called_once()
        mock_cleanup.assert_called_once()
        # Should send "Context Restored" message
        # We can check if cl.Message was instantiated with specific content
        call_args = mock_cl.Message.call_args_list
        found = False
        for call in call_args:
            if "Context Restored" in call.kwargs.get("content", "") or (
                len(call.args) > 0 and "Context Restored" in call.args[0]
            ):
                found = True
                break
        assert found


@pytest.mark.asyncio
async def test_main_message():
    with patch("src.web.app.clean_llm_response") as mock_clean:
        mock_agent = MagicMock()
        mock_agent.process_message = AsyncMock(return_value="response")

        mock_cl.user_session.get.side_effect = lambda k: {
            "agent": mock_agent,
            "agent_type": "jarvis",
            "id": "session1",
        }.get(k)

        mock_clean.return_value = "cleaned response"

        message = MagicMock()
        message.content = "hello"

        await main(message)

        mock_agent.process_message.assert_called_once()
        mock_cl.Message.assert_called()  # Should send response
