from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.config.loader import GlobalConfig, LLMConfig, ProviderConfig
from src.core.context.manager import ConversationContext, ConversationTurn


@pytest.fixture
def mock_llm_config():
    config = MagicMock(spec=LLMConfig)
    config.global_config = MagicMock(spec=GlobalConfig)
    config.global_config.default_provider = "openai"

    provider_config = MagicMock(spec=ProviderConfig)
    provider_config.type = "openai"
    provider_config.default_model = "gpt-3.5-turbo"
    provider_config.api_key = "test_key"
    provider_config.base_url = None

    config.providers = {"openai": provider_config}
    return config


@pytest.mark.asyncio
async def test_compress_history_triggers_compression(mock_llm_config):
    # Setup context with many turns
    context = ConversationContext(session_id="test_session")
    for i in range(25):
        context.add_turn(ConversationTurn(f"Input {i}", f"Response {i}"))

    # Mock dependencies
    with (
        patch("src.core.context.manager.load_llm_config", return_value=mock_llm_config),
        patch("src.core.context.manager.ChatOpenAI") as MockChatOpenAI,
        patch("src.core.context.manager.PromptTemplate") as MockPromptTemplate,
        patch("src.core.context.manager.StrOutputParser") as MockStrOutputParser,
    ):
        # Setup chain mock
        mock_chain = AsyncMock()
        mock_chain.ainvoke.return_value = "Summarized conversation content"

        # Setup the pipe behavior to return our mock_chain
        # The chain is formed by: prompt | llm | StrOutputParser()
        # We need to ensure that the result of these operations is mock_chain

        # Mock the prompt instance
        mock_prompt_instance = MagicMock()
        MockPromptTemplate.from_template.return_value = mock_prompt_instance

        # Mock the LLM instance
        mock_llm_instance = MagicMock()
        MockChatOpenAI.return_value = mock_llm_instance

        # Mock the StrOutputParser instance
        mock_parser_instance = MagicMock()
        MockStrOutputParser.return_value = mock_parser_instance

        # Define the pipe behavior
        # prompt | llm -> intermediate
        # intermediate | parser -> chain
        mock_intermediate = MagicMock()
        mock_prompt_instance.__or__.return_value = mock_intermediate
        mock_intermediate.__or__.return_value = mock_chain

        # Alternatively, if the pipe order or implementation details vary,
        # we might need to be more flexible, but this follows the code structure.

        # Execute compression
        await context.compress_history(threshold=20, keep_recent=5)

        # Verification

        # 1. Check if chain was invoked
        mock_chain.ainvoke.assert_called_once()

        # 2. Check conversation history length
        # Should be 1 (summary) + 5 (kept recent) = 6
        assert len(context.conversation_history) == 6

        # 3. Check the first turn is the summary
        summary_turn = context.conversation_history[0]
        assert summary_turn.user_input == "System: Previous conversation summary"
        assert summary_turn.agent_response == "Summarized conversation content"
        assert summary_turn.metadata.get("is_summary") is True
        assert summary_turn.metadata.get("compressed_turns") == 20  # 25 total - 5 kept

        # 4. Check the last turn is the most recent one
        assert context.conversation_history[-1].user_input == "Input 24"


@pytest.mark.asyncio
async def test_compress_history_below_threshold(mock_llm_config):
    context = ConversationContext()
    for i in range(5):
        context.add_turn(ConversationTurn(f"Input {i}", f"Response {i}"))

    with patch(
        "src.core.context.manager.load_llm_config", return_value=mock_llm_config
    ) as mock_load:
        await context.compress_history(threshold=10)

        # Should not attempt to load config or compress
        mock_load.assert_not_called()
        assert len(context.conversation_history) == 5


@pytest.mark.asyncio
async def test_compress_history_provider_config_error(mock_llm_config):
    context = ConversationContext()
    for i in range(25):
        context.add_turn(ConversationTurn(f"Input {i}", f"Response {i}"))

    # Mock config to return empty providers or missing default provider
    mock_llm_config.global_config.default_provider = "missing_provider"
    mock_llm_config.providers = {}

    with patch(
        "src.core.context.manager.load_llm_config", return_value=mock_llm_config
    ):
        await context.compress_history(threshold=20)

        # Should log warning and return without changing history
        assert len(context.conversation_history) == 25


@pytest.mark.asyncio
async def test_summarize_with_llm_manager_requires_manager():
    context = ConversationContext()

    with pytest.raises(ValueError, match="LLM manager is not configured"):
        await context._summarize_with_llm_manager("conversation")


@pytest.mark.asyncio
async def test_summarize_with_llm_manager_rejects_empty_summary():
    llm_manager = MagicMock()
    llm_manager.generate_response = AsyncMock(return_value=MagicMock(content="   "))
    context = ConversationContext(llm_manager=llm_manager)

    with pytest.raises(ValueError, match="Compression summary cannot be empty"):
        await context._summarize_with_llm_manager("conversation")


@pytest.mark.asyncio
async def test_compress_history_falls_back_to_openai_provider_with_base_url(
    mock_llm_config,
):
    context = ConversationContext(llm_manager=MagicMock())
    context.llm_manager.generate_response = AsyncMock(side_effect=RuntimeError("boom"))
    for i in range(25):
        context.add_turn(ConversationTurn(f"Input {i}", f"Response {i}"))

    mock_llm_config.providers["openai"].base_url = "https://openai.example.com"

    with (
        patch("src.core.context.manager.load_llm_config", return_value=mock_llm_config),
        patch("src.core.context.manager.ChatOpenAI") as mock_chat_openai,
        patch("src.core.context.manager.PromptTemplate") as mock_prompt_template,
        patch("src.core.context.manager.StrOutputParser") as mock_output_parser,
    ):
        mock_chain = AsyncMock()
        mock_chain.ainvoke.return_value = "OpenAI fallback summary"
        mock_prompt_instance = MagicMock()
        mock_intermediate = MagicMock()
        mock_prompt_template.from_template.return_value = mock_prompt_instance
        mock_prompt_instance.__or__.return_value = mock_intermediate
        mock_intermediate.__or__.return_value = mock_chain
        mock_output_parser.return_value = MagicMock()

        await context.compress_history(threshold=20, keep_recent=5)

        mock_chat_openai.assert_called_once()
        assert (
            mock_chat_openai.call_args.kwargs["openai_api_base"]
            == "https://openai.example.com"
        )
        assert (
            context.conversation_history[0].agent_response == "OpenAI fallback summary"
        )


@pytest.mark.asyncio
async def test_compress_history_supports_anthropic_provider_fallback():
    context = ConversationContext(llm_manager=MagicMock())
    context.llm_manager.generate_response = AsyncMock(side_effect=RuntimeError("boom"))
    for i in range(25):
        context.add_turn(ConversationTurn(f"Input {i}", f"Response {i}"))

    llm_config = MagicMock(spec=LLMConfig)
    llm_config.global_config = MagicMock(spec=GlobalConfig)
    llm_config.global_config.default_provider = "anthropic"
    provider_config = MagicMock(spec=ProviderConfig)
    provider_config.type = "anthropic"
    provider_config.default_model = "claude-test"
    provider_config.api_key = "anthropic-key"
    provider_config.base_url = "https://anthropic.example.com"
    llm_config.providers = {"anthropic": provider_config}

    with (
        patch("src.core.context.manager.load_llm_config", return_value=llm_config),
        patch("src.core.context.manager.ChatAnthropic") as mock_chat_anthropic,
        patch("src.core.context.manager.PromptTemplate") as mock_prompt_template,
        patch("src.core.context.manager.StrOutputParser") as mock_output_parser,
    ):
        mock_chain = AsyncMock()
        mock_chain.ainvoke.return_value = "Anthropic fallback summary"
        mock_prompt_instance = MagicMock()
        mock_intermediate = MagicMock()
        mock_prompt_template.from_template.return_value = mock_prompt_instance
        mock_prompt_instance.__or__.return_value = mock_intermediate
        mock_intermediate.__or__.return_value = mock_chain
        mock_output_parser.return_value = MagicMock()

        await context.compress_history(threshold=20, keep_recent=5)

        mock_chat_anthropic.assert_called_once()
        assert mock_chat_anthropic.call_args.kwargs["api_key"] == "anthropic-key"
        assert (
            mock_chat_anthropic.call_args.kwargs["anthropic_api_url"]
            == "https://anthropic.example.com"
        )
        assert (
            context.conversation_history[0].agent_response
            == "Anthropic fallback summary"
        )


@pytest.mark.asyncio
async def test_compress_history_supports_ollama_provider_fallback():
    context = ConversationContext(llm_manager=MagicMock())
    context.llm_manager.generate_response = AsyncMock(side_effect=RuntimeError("boom"))
    for i in range(25):
        context.add_turn(ConversationTurn(f"Input {i}", f"Response {i}"))

    llm_config = MagicMock(spec=LLMConfig)
    llm_config.global_config = MagicMock(spec=GlobalConfig)
    llm_config.global_config.default_provider = "ollama"
    provider_config = MagicMock(spec=ProviderConfig)
    provider_config.type = "ollama"
    provider_config.default_model = "llama-test"
    provider_config.api_key = None
    provider_config.base_url = "http://ollama.local"
    llm_config.providers = {"ollama": provider_config}

    with (
        patch("src.core.context.manager.load_llm_config", return_value=llm_config),
        patch("src.core.context.manager.ChatOllama") as mock_chat_ollama,
        patch("src.core.context.manager.PromptTemplate") as mock_prompt_template,
        patch("src.core.context.manager.StrOutputParser") as mock_output_parser,
    ):
        mock_chain = AsyncMock()
        mock_chain.ainvoke.return_value = "Ollama fallback summary"
        mock_prompt_instance = MagicMock()
        mock_intermediate = MagicMock()
        mock_prompt_template.from_template.return_value = mock_prompt_instance
        mock_prompt_instance.__or__.return_value = mock_intermediate
        mock_intermediate.__or__.return_value = mock_chain
        mock_output_parser.return_value = MagicMock()

        await context.compress_history(threshold=20, keep_recent=5)

        mock_chat_ollama.assert_called_once()
        assert mock_chat_ollama.call_args.kwargs["base_url"] == "http://ollama.local"
        assert (
            context.conversation_history[0].agent_response == "Ollama fallback summary"
        )


@pytest.mark.asyncio
async def test_compress_history_unsupported_provider_returns_without_changes():
    context = ConversationContext(llm_manager=MagicMock())
    context.llm_manager.generate_response = AsyncMock(side_effect=RuntimeError("boom"))
    for i in range(25):
        context.add_turn(ConversationTurn(f"Input {i}", f"Response {i}"))

    llm_config = MagicMock(spec=LLMConfig)
    llm_config.global_config = MagicMock(spec=GlobalConfig)
    llm_config.global_config.default_provider = "custom"
    provider_config = MagicMock(spec=ProviderConfig)
    provider_config.type = "custom"
    provider_config.default_model = "custom-model"
    provider_config.api_key = None
    provider_config.base_url = None
    llm_config.providers = {"custom": provider_config}

    with patch("src.core.context.manager.load_llm_config", return_value=llm_config):
        await context.compress_history(threshold=20, keep_recent=5)

    assert len(context.conversation_history) == 25
