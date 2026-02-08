
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.core.llm.providers.openai_client import OpenAILLMClient
from src.core.llm.client import LLMConfig, LLMMessage, LLMResponse, LLMProvider
from src.core.llm.utils.error_handler import LLMConfigurationError, LLMAuthenticationError, LLMAPIError

@pytest.fixture
def valid_config():
    return LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-3.5-turbo",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        timeout={"connect": 5.0, "read": 10.0, "total": 15.0},
        retry={"max_attempts": 3, "initial_delay": 0.1, "max_delay": 1.0, "exponential_base": 2.0}
    )

@pytest.fixture
def openai_client(valid_config):
    return OpenAILLMClient(valid_config)

class TestOpenAIClient:
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, openai_client):
        """Test successful initialization"""
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_instance = mock_openai.return_value
            # Mock the connection validation call
            mock_instance.chat.completions.create = AsyncMock(return_value=MagicMock())
            
            result = await openai_client.initialize()
            
            assert result is True
            assert openai_client._initialized is True
            mock_openai.assert_called_once()
            mock_instance.chat.completions.create.assert_called_once()  # Connection validation

    @pytest.mark.asyncio
    async def test_initialize_failure_no_api_key(self):
        """Test initialization failure due to missing API key"""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-3.5-turbo",
            api_key=""  # Empty API key
        )
        client = OpenAILLMClient(config)
        
        # Should raise LLMConfigurationError during initialization or return False?
        # Looking at implementation: _validate_config raises LLMConfigurationError, caught in initialize -> returns False
        result = await client.initialize()
        assert result is False
        assert client._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_failure_connection_error(self, openai_client):
        """Test initialization failure due to connection error"""
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_instance = mock_openai.return_value
            # Mock connection validation to raise an error
            mock_instance.chat.completions.create = AsyncMock(side_effect=Exception("Connection failed"))
            
            result = await openai_client.initialize()
            
            assert result is False
            assert openai_client._initialized is False

    @pytest.mark.asyncio
    async def test_generate_response_success(self, openai_client):
        """Test successful response generation"""
        # Manually set initialized to True to skip initialize call in test
        openai_client._initialized = True
        
        # Mock response
        mock_response = MagicMock()
        mock_response.id = "test-id"
        mock_response.created = 1234567890
        mock_response.object = "chat.completion"
        mock_response.model = "gpt-3.5-turbo"
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="Hello world"),
                finish_reason="stop"
            )
        ]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        
        with patch('openai.AsyncOpenAI') as mock_openai:
            openai_client._client = mock_openai.return_value
            openai_client._client.chat.completions.create = AsyncMock(return_value=mock_response)
            
            messages = [LLMMessage(role="user", content="Hello")]
            response = await openai_client.generate_response(messages)
            
            assert isinstance(response, LLMResponse)
            assert response.content == "Hello world"
            assert response.usage["total_tokens"] == 15
            assert response.metadata["id"] == "test-id"

    @pytest.mark.asyncio
    async def test_generate_response_retry(self, openai_client):
        """Test retry mechanism on API error"""
        openai_client._initialized = True
        
        # Mock API error then success
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Success"))]
        mock_response.usage = MagicMock()
        
        import openai
        error = openai.APIError("Temporary error", request=None, body=None)
        
        with patch('openai.AsyncOpenAI') as mock_openai:
            openai_client._client = mock_openai.return_value
            # First call fails, second succeeds
            openai_client._client.chat.completions.create = AsyncMock(side_effect=[error, mock_response])
            
            messages = [LLMMessage(role="user", content="Hello")]
            
            # Reduce retry delay for test speed
            openai_client.config.retry['initial_delay'] = 0.01
            
            response = await openai_client.generate_response(messages)
            
            assert response.content == "Success"
            assert openai_client._client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_stream_response(self, openai_client):
        """Test streaming response"""
        openai_client._initialized = True
        
        # Mock stream chunks
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock(delta=MagicMock(content="Hello"))]
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock(delta=MagicMock(content=" World"))]
        
        async def mock_stream_gen(**kwargs):
            yield chunk1
            yield chunk2
            
        with patch('openai.AsyncOpenAI') as mock_openai:
            openai_client._client = mock_openai.return_value
            openai_client._client.chat.completions.create = AsyncMock(side_effect=mock_stream_gen)
            
            messages = [LLMMessage(role="user", content="Hello")]
            chunks = []
            async for chunk in openai_client.stream_response(messages):
                chunks.append(chunk)
                
            assert "".join(chunks) == "Hello World"
            
            # Verify stream=True was passed
            call_kwargs = openai_client._client.chat.completions.create.call_args.kwargs
            assert call_kwargs["stream"] is True
