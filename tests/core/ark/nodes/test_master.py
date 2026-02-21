import pytest
from unittest.mock import Mock, MagicMock
from src.core.ark.nodes.master import MasterNode
from src.core.llm.client import LLMManager
from src.core.ark.state import JarvisState

class TestMasterNode:
    def test_init(self):
        llm_manager = Mock(spec=LLMManager)
        node = MasterNode(llm_manager=llm_manager)
        assert node.prompt_manager is not None
        
    def test_get_system_prompt_empty_state(self):
        llm_manager = Mock(spec=LLMManager)
        node = MasterNode(llm_manager=llm_manager)
        
        state = {
            "todo_list": [],
            "available_tools": {},
            "messages": []
        }
        
        prompt = node._get_system_prompt(state)
        
        assert "No tasks in todo list." in prompt
        assert "No tools available." in prompt
        assert "You are Jarvis, an intelligent agent acting as an Orchestrator." in prompt
        
    def test_get_system_prompt_with_data(self):
        llm_manager = Mock(spec=LLMManager)
        node = MasterNode(llm_manager=llm_manager)
        
        state = {
            "todo_list": [{"id": "1", "status": "pending", "description": "Test Task"}],
            "available_tools": {
                "test_tool": {
                    "description": "A test tool",
                    "input_schema": {"type": "object"}
                }
            },
            "messages": []
        }
        
        prompt = node._get_system_prompt(state)
        
        assert "Current Todo List:" in prompt
        assert "- [1] pending: Test Task" in prompt
        assert "Available Tools:" in prompt
        assert "- test_tool: A test tool" in prompt
