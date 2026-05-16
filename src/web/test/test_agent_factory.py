import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.web.agent_factory
from src.web.agent_factory import create_agent, get_llm_config


@pytest.fixture
def mock_load_llm_config():
    with patch("src.web.agent_factory.load_llm_config") as mock:
        yield mock


@pytest.fixture
def mock_convert_config():
    with patch("src.web.agent_factory.convert_to_client_config") as mock:
        yield mock


@pytest.mark.asyncio
async def test_create_agent_jarvis(mock_load_llm_config, mock_convert_config):
    with patch("src.web.agent_factory.JarvisAgent") as MockAgent:
        mock_instance = MockAgent.return_value
        mock_instance.initialize = AsyncMock(return_value=True)
        mock_instance.start = AsyncMock()

        agent = await create_agent("jarvis")

        assert agent == mock_instance
        mock_instance.initialize.assert_called_once()
        mock_instance.start.assert_called_once()


@pytest.mark.asyncio
async def test_create_agent_deep_research(mock_load_llm_config, mock_convert_config):
    with patch("src.web.agent_factory.DeepResearchAgent") as MockAgent:
        mock_instance = MockAgent.return_value
        # Mock initialize (can be async or sync, factory handles it)
        mock_instance.initialize = AsyncMock()

        agent = await create_agent("deep_research")

        assert agent == mock_instance
        mock_instance.initialize.assert_called_once()


def test_get_llm_config_cache(mock_load_llm_config, mock_convert_config):
    # Reset cache
    src.web.agent_factory._llm_config_cache = {}

    mock_config = MagicMock()
    mock_config.provider_name = "test_provider"
    mock_config.model = "test_model"
    mock_convert_config.return_value = mock_config

    c1 = get_llm_config("env1")
    c2 = get_llm_config("env1")

    assert c1 == mock_config
    assert c2 == mock_config

    # load should be called only once due to cache
    mock_load_llm_config.assert_called_once()


@pytest.mark.asyncio
async def test_create_agent_uses_user_settings_for_api_key():
    with (
        patch("src.web.agent_factory.JarvisAgent") as MockAgent,
        patch("src.web.agent_factory.get_llm_config") as mock_get_config,
    ):
        mock_instance = MockAgent.return_value
        mock_instance.initialize = AsyncMock(return_value=True)
        mock_instance.start = AsyncMock()

        config = MagicMock()
        config.api_key = "system-key"
        mock_get_config.return_value = config

        await create_agent(
            "jarvis",
            user_settings={
                "openai_api_key": "user-key",
                "tavily_api_key": "tavily-key",
                "confluence_page_token": "confluence-token",
                "beacon_model_token": "beacon-token",
            },
        )

        effective = mock_instance.ark_engine.llm_manager._default_config
        assert effective.api_key == "user-key"
        assert os.environ["TAVILY_API_KEY"] == "tavily-key"
        assert os.environ["CONFLUENCE_PAGE_TOKEN"] == "confluence-token"
        assert os.environ["BEACON_MODEL_TOKEN"] == "beacon-token"
