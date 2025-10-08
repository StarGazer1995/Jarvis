"""
Tests for LLM integration in ARK engine.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from src.core.ark_engine import ARKEngine
from src.core.llm_config import LLMConfig, LLMProvider
from src.core.llm_client import LLMManager, MockLLMClient
from src.core.prompt_manager import PromptManager


class TestLLMIntegration:
    """Test LLM integration functionality."""
    
    @pytest.fixture
    def llm_config(self):
        """Create test LLM configuration."""
        return LLMConfig(
            provider=LLMProvider.MOCK,
            model="test-model",
            max_tokens=500,
            temperature=0.5
        )
    
    @pytest.fixture
    def ark_config(self, llm_config):
        """Create ARK configuration with LLM enabled."""
        return {
            'enable_llm': True,
            'llm': llm_config.to_dict(),
            'max_conversation_history': 50,
            'confidence_threshold': 0.6
        }
    
    @pytest.fixture
    def ark_engine(self, ark_config):
        """Create ARK engine with LLM configuration."""
        return ARKEngine(ark_config)
    
    def test_ark_engine_llm_initialization(self, ark_engine):
        """Test ARK engine initializes LLM components correctly."""
        assert ark_engine.llm_enabled is True
        assert ark_engine.llm_config is not None
        assert ark_engine.llm_manager is not None
        assert ark_engine.prompt_manager is not None
        assert ark_engine.llm_config.provider == LLMProvider.MOCK
    
    @pytest.mark.asyncio
    async def test_llm_initialization_in_ark(self, ark_engine):
        """Test LLM initialization during ARK engine startup."""
        with patch.object(ark_engine.llm_manager, 'initialize_default_client', new_callable=AsyncMock, return_value=True) as mock_init:
            with patch.object(ark_engine.prompt_manager, 'load_default_templates') as mock_load:
                await ark_engine._initialize_llm()
                
                mock_init.assert_called_once()
                mock_load.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_llm_response_generation(self, ark_engine):
        """Test LLM response generation."""
        # Mock intent result
        from src.core.intent_engine import IntentResult, IntentType
        intent_result = IntentResult(
            intent=IntentType.QUESTION,
            confidence=0.8,
            entities=[],
            raw_text="What's the weather like?",
            processed_text="what's the weather like?"
        )
        
        # Mock ARK decision
        from src.core.ark_engine import ARKDecision
        decision = ARKDecision(
            action_type="answer_question",
            confidence=0.8,
            reasoning="User asked about weather",
            tools_to_use=[],
            parameters={},
            metadata={}
        )
        
        tool_results = {}
        
        # Mock LLM manager response
        with patch.object(ark_engine.llm_manager, 'generate_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = "The weather is sunny today!"
            
            response = await ark_engine._generate_llm_response(intent_result, decision, tool_results)
            
            assert response == "The weather is sunny today!"
            mock_generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_llm_fallback_on_error(self, ark_engine):
        """Test fallback to template response when LLM fails."""
        from src.core.intent_engine import IntentResult, IntentType
        intent_result = IntentResult(
            intent=IntentType.GREETING,
            confidence=0.9,
            entities=[],
            raw_text="Hello",
            processed_text="hello"
        )
        
        from src.core.ark_engine import ARKDecision
        decision = ARKDecision(
            action_type="respond_greeting",
            confidence=0.9,
            reasoning="User greeted",
            tools_to_use=[],
            parameters={},
            metadata={}
        )
        
        tool_results = {}
        
        # Mock LLM manager to raise exception
        with patch.object(ark_engine.llm_manager, 'generate_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.side_effect = Exception("LLM service unavailable")
            
            response = await ark_engine._generate_llm_response(intent_result, decision, tool_results)
            
            # Should fallback to template response
            assert "Hello" in response or "Hi" in response or "你好" in response
    
    @pytest.mark.asyncio
    async def test_template_response_fallback(self, ark_engine):
        """Test template response generation when LLM is disabled."""
        # Disable LLM
        ark_engine.llm_enabled = False
        
        from src.core.intent_engine import IntentResult, IntentType
        intent_result = IntentResult(
            intent=IntentType.GREETING,
            confidence=0.9,
            entities=[],
            raw_text="Hello",
            processed_text="hello"
        )
        
        from src.core.ark_engine import ARKDecision
        decision = ARKDecision(
            action_type="respond_greeting",
            confidence=0.9,
            reasoning="User greeted",
            tools_to_use=[],
            parameters={},
            metadata={}
        )
        
        tool_results = {}
        
        response = await ark_engine._generate_response(intent_result, decision, tool_results)
        
        # Should use template response
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_llm_config_loading(self):
        """Test LLM configuration loading."""
        config_dict = {
            'provider': 'mock',
            'model': 'test-model',
            'max_tokens': 800,
            'temperature': 0.3
        }
        
        ark_config = {
            'enable_llm': True,
            'llm': config_dict
        }
        
        engine = ARKEngine(ark_config)
        
        assert engine.llm_config.provider == LLMProvider.MOCK
        assert engine.llm_config.model == 'test-model'
        assert engine.llm_config.max_tokens == 800
        assert engine.llm_config.temperature == 0.3
    
    def test_llm_disabled_configuration(self):
        """Test ARK engine with LLM disabled."""
        ark_config = {
            'enable_llm': False
        }
        
        engine = ARKEngine(ark_config)
        
        assert engine.llm_enabled is False
        assert engine.llm_config is not None  # Still initialized for potential future use
    
    @pytest.mark.asyncio
    async def test_full_conversation_with_llm(self, ark_engine):
        """Test full conversation flow with LLM integration."""
        # Initialize the engine
        await ark_engine.initialize()
        
        # Mock LLM response
        with patch.object(ark_engine.llm_manager, 'generate_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = "Hello! How can I help you today?"
            
            response = await ark_engine.process_input("Hello")
            
            assert isinstance(response, str)
            assert len(response) > 0
    
    def test_prompt_manager_integration(self, ark_engine):
        """Test prompt manager integration with ARK engine."""
        assert ark_engine.prompt_manager is not None
        
        # Check if default templates are available by name
        templates = ark_engine.prompt_manager.templates
        assert 'jarvis_system' in templates
        assert 'conversation_response' in templates
        assert 'intent_recognition' in templates
        assert 'response_generation' in templates


class TestLLMConfig:
    """Test LLM configuration functionality."""
    
    def test_llm_config_creation(self):
        """Test LLM configuration creation."""
        config = LLMConfig(
            provider=LLMProvider.MOCK,
            model="test-model",
            max_tokens=1000,
            temperature=0.7
        )
        
        assert config.provider == LLMProvider.MOCK
        assert config.model == "test-model"
        assert config.max_tokens == 1000
        assert config.temperature == 0.7
    
    def test_llm_config_from_dict(self):
        """Test LLM configuration creation from dictionary."""
        config_dict = {
            'provider': 'openai',
            'model': 'gpt-4',
            'api_key': 'test-key',
            'max_tokens': 2000,
            'temperature': 0.5
        }
        
        config = LLMConfig.from_dict(config_dict)
        
        assert config.provider == LLMProvider.OPENAI
        assert config.model == 'gpt-4'
        assert config.api_key == 'test-key'
        assert config.max_tokens == 2000
        assert config.temperature == 0.5
    
    def test_llm_config_to_dict(self):
        """Test LLM configuration conversion to dictionary."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-3.5-turbo",
            api_key="test-key",
            max_tokens=1500
        )
        
        config_dict = config.to_dict()
        
        assert config_dict['provider'] == 'openai'
        assert config_dict['model'] == 'gpt-3.5-turbo'
        assert config_dict['api_key'] == 'test-key'
        assert config_dict['max_tokens'] == 1500


class TestPromptManager:
    """Test prompt manager functionality."""
    
    @pytest.fixture
    def prompt_manager(self):
        """Create prompt manager instance."""
        return PromptManager()
    
    def test_prompt_manager_initialization(self, prompt_manager):
        """Test prompt manager initialization."""
        # PromptManager now automatically loads default templates
        assert len(prompt_manager.templates) > 0
        assert 'intent_recognition' in prompt_manager.templates
        assert 'response_generation' in prompt_manager.templates
    
    def test_load_default_templates(self, prompt_manager):
        """Test loading default templates."""
        # Templates are already loaded in __init__, but we can call it again
        prompt_manager.load_default_templates()
        
        # Check that default templates are loaded by name
        assert 'jarvis_system' in prompt_manager.templates
        assert 'conversation_response' in prompt_manager.templates
        assert 'tool_usage_decision' in prompt_manager.templates
        assert 'intent_recognition' in prompt_manager.templates
        assert 'response_generation' in prompt_manager.templates
    
    def test_build_conversation_messages(self, prompt_manager):
        """Test building conversation messages."""
        prompt_manager.load_default_templates()
        
        messages = prompt_manager.build_conversation_messages(
            user_input="Hello",
            conversation_history=[],
            system_context={"agent_name": "Jarvis"},
            intent_info={"intent": "greeting", "confidence": 0.9},
            tool_results=[]
        )
        
        assert isinstance(messages, list)
        assert len(messages) > 0
        assert any(msg.role == "system" for msg in messages)
        assert any(msg.role == "user" for msg in messages)