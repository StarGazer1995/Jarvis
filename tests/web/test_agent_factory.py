import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.web.agent_factory import create_agent, get_llm_config
import src.web.agent_factory


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
