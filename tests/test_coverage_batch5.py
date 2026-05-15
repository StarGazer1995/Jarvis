"""
最后一波推向90% - nodes, parsers, tasks, utils, litellm
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════
# src/core/ark/nodes/tools.py
# ═══════════════════════════════════════════════════════════════

class TestToolsNodeFinal:
    """ToolsNode 最后覆盖"""

    def test_tools_node_custom_concurrency(self):
        from unittest.mock import MagicMock
        from src.core.ark.nodes.tools import ToolsNode

        mcp = MagicMock()
        node = ToolsNode(mcp_client=mcp, max_concurrency=3)
        assert node.max_concurrency == 3


# ═══════════════════════════════════════════════════════════════
# src/core/ark/nodes/master.py
# ═══════════════════════════════════════════════════════════════

class TestMasterNodeFinal:
    """MasterNode 最后覆盖"""

    def test_master_node_empty_agents(self):
        from unittest.mock import MagicMock
        from src.core.ark.nodes.master import MasterNode

        llm = MagicMock()
        node = MasterNode(llm_manager=llm)
        assert node.agents == []
        assert node.agent_map == {}

    def test_master_node_with_agents(self):
        from unittest.mock import MagicMock
        from src.core.ark.nodes.master import MasterNode

        llm = MagicMock()
        node = MasterNode(llm_manager=llm, agents=[])
        assert node.agents is not None


# ═══════════════════════════════════════════════════════════════
# src/core/ark/utils.py
# ═══════════════════════════════════════════════════════════════

class TestArkUtilsFinal:
    """ark/utils.py 最后覆盖"""

    def test_create_agent_node_str_result(self):
        from src.core.ark.utils import create_agent_node
        from src.core.ark.state import MultiAgentState

        def simple_agent(state: MultiAgentState):
            return "Hello from agent"

        node_func = create_agent_node("test_agent", simple_agent)
        assert node_func is not None


# ═══════════════════════════════════════════════════════════════
# src/core/llm/parsers.py
# ═══════════════════════════════════════════════════════════════

class TestParsersFinal:
    """parsers.py 最后覆盖"""

    def test_base_output_parser(self):
        from src.core.llm.parsers import BaseOutputParser, JSONOutputParser

        parser = JSONOutputParser()
        assert isinstance(parser, BaseOutputParser)

    def test_json_parser_empty_string(self):
        from src.core.llm.parsers import JSONOutputParser

        parser = JSONOutputParser()
        with pytest.raises(ValueError):
            parser.parse("")

    def test_json_parser_invalid_json_without_repair(self):
        from src.core.llm.parsers import JSONOutputParser

        parser = JSONOutputParser(allow_repair=False)
        with pytest.raises(ValueError):
            parser.parse("{invalid}")


# ═══════════════════════════════════════════════════════════════
# src/core/ark/tasks.py
# ═══════════════════════════════════════════════════════════════

class TestTasksFinal:
    """tasks.py 最后覆盖"""

    def test_task_status_enum(self):
        from src.core.ark.tasks import TaskStatus

        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"

    def test_execute_manage_add_no_description(self):
        from src.core.ark.tasks import execute_manage_tasks

        result = execute_manage_tasks({"action": "add"}, [])
        assert "missing" in result.lower() or "description" in result.lower()


# ═══════════════════════════════════════════════════════════════
# src/core/llm/providers/litellm_client.py
# ═══════════════════════════════════════════════════════════════

class TestLiteLLMFinal:
    """LiteLLM 最后覆盖"""

    def test_litellm_config_creation(self):
        from src.core.llm.types import LLMConfig, LLMProvider

        cfg = LLMConfig(provider=LLMProvider.LITELLM, model="gpt-4")
        assert cfg.provider == LLMProvider.LITELLM
        assert cfg.model == "gpt-4"
        assert cfg.provider_name is None
