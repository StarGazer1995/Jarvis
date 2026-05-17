"""Additional MasterNode tests migrated from legacy `src/test` files."""

from unittest.mock import MagicMock


class TestMasterNodeCoverage:
    """MasterNode 边界测试"""

    def test_master_node_creation(self):
        from src.core.ark.nodes.master import MasterNode

        llm = MagicMock()
        node = MasterNode(llm_manager=llm)
        assert node is not None
        assert node.agents == []
        assert node.agent_map == {}


class TestMasterNodeMore:
    """MasterNode 更多测试"""

    def test_master_node_creation_with_agents(self):
        from unittest.mock import MagicMock

        from src.core.ark.nodes.master import MasterNode

        llm = MagicMock()
        node = MasterNode(llm_manager=llm, agents=[])
        assert node.agents == []
        assert node.agent_map == {}


class TestMasterNodeFinal:
    """MasterNode 最后覆盖"""

    def test_master_node_empty_agents(self):
        from src.core.ark.nodes.master import MasterNode

        llm = MagicMock()
        node = MasterNode(llm_manager=llm)
        assert node.agents == []
        assert node.agent_map == {}

    def test_master_node_with_agents(self):
        from src.core.ark.nodes.master import MasterNode

        llm = MagicMock()
        node = MasterNode(llm_manager=llm, agents=[])
        assert node.agents is not None
