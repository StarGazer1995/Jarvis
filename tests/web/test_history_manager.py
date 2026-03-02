import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.web.history_manager import restore_agent_history, cleanup_system_messages


def test_restore_agent_history_dict():
    agent = MagicMock()
    # deeply nested mocks are annoying, let's just check if add_exchange is called
    agent.ark_engine.context_manager.add_exchange = MagicMock()

    thread = {
        "id": "t1",
        "steps": [
            {
                "type": "user_message",
                "output": "hi",
                "createdAt": "2023-01-01T10:00:00",
            },
            {
                "type": "assistant_message",
                "output": "hello",
                "createdAt": "2023-01-01T10:01:00",
            },
        ],
    }

    count = restore_agent_history(agent, thread)
    assert count == 1
    agent.ark_engine.context_manager.add_exchange.assert_called_once()


def test_restore_agent_history_object():
    agent = MagicMock()
    agent.ark_engine.context_manager.add_exchange = MagicMock()

    step1 = MagicMock()
    step1.type = "user_message"
    step1.output = "hi"
    step1.createdAt = "1"

    step2 = MagicMock()
    step2.type = "assistant_message"
    step2.output = "hello"
    step2.createdAt = "2"

    thread = MagicMock()
    thread.id = "t1"
    thread.steps = [step1, step2]

    count = restore_agent_history(agent, thread)
    assert count == 1
    agent.ark_engine.context_manager.add_exchange.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup_system_messages():
    with patch("src.web.history_manager.get_data_layer") as mock_get_dl:
        mock_dl = MagicMock()
        mock_conn = AsyncMock()
        mock_dl.engine.begin.return_value.__aenter__.return_value = mock_conn
        mock_get_dl.return_value = mock_dl

        await cleanup_system_messages("t1")

        mock_conn.execute.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup_system_messages_no_dl():
    with patch("src.web.history_manager.get_data_layer") as mock_get_dl:
        mock_get_dl.return_value = None

        # Should not raise exception
        await cleanup_system_messages("t1")
