
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.core.llm_client import LLMManager, LLMConfig, LLMProvider, LLMMessage, LLMResponse, BaseLLMClient

class MockClient(BaseLLMClient):
    def __init__(self, config: LLMConfig, should_fail=False):
        super().__init__(config)
        self.should_fail = should_fail
        self.fail_count = 0
        
    async def initialize(self) -> bool:
        self._initialized = True
        return True
        
    async def generate_response(self, messages, **kwargs):
        if self.should_fail:
            raise Exception("Mock failure")
        return LLMResponse(content="Mock response")
        
    async def stream_response(self, messages, **kwargs):
        if self.should_fail:
            raise Exception("Mock failure")
        yield "Mock"
        yield " response"

@pytest.mark.asyncio
async def test_llm_manager_fallback():
    # Setup configs
    primary_config = LLMConfig(provider=LLMProvider.MOCK, model="primary")
    fallback_config1 = LLMConfig(provider=LLMProvider.MOCK, model="fallback1")
    fallback_config2 = LLMConfig(provider=LLMProvider.MOCK, model="fallback2")
    
    # Setup manager
    manager = LLMManager(primary_config)
    
    # We need to manually inject our MockClient because LLMClientFactory creates new instances
    # Or we can patch LLMClientFactory.
    
    # Let's add clients normally first
    await manager.initialize_default_client()
    await manager.add_client("fallback1", fallback_config1)
    await manager.add_client("fallback2", fallback_config2)
    
    # Now replace the clients with our controlled mocks
    primary_client = MockClient(primary_config, should_fail=True)
    await primary_client.initialize()
    manager.clients["default"] = primary_client
    
    fallback_client1 = MockClient(fallback_config1, should_fail=True)
    await fallback_client1.initialize()
    manager.clients["fallback1"] = fallback_client1
    
    fallback_client2 = MockClient(fallback_config2, should_fail=False)
    await fallback_client2.initialize()
    manager.clients["fallback2"] = fallback_client2
    
    # Set fallback chain
    manager.set_fallback_providers(["fallback1", "fallback2"])
    
    # Test generation
    messages = [LLMMessage(role="user", content="Hello")]
    response = await manager.generate_response(messages)
    
    assert response.content == "Mock response"
    
    # Verify primary failed
    # We can't easily verify fail count without spy, but success means fallback worked.
    
    # Test failure of all providers
    fallback_client2.should_fail = True
    
    with pytest.raises(Exception, match="Mock failure"):
        await manager.generate_response(messages)

