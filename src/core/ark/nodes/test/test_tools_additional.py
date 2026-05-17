"""Additional ToolsNode tests migrated from legacy `src/test` files."""

from unittest.mock import MagicMock


class TestToolsNodeDetailedCoverage:
    """ToolsNode 边界测试"""

    def test_tools_node_create_with_mock(self):
        from src.core.ark.nodes.tools import ToolsNode

        mcp = MagicMock()
        node = ToolsNode(mcp_client=mcp)
        assert node is not None
        assert node.max_concurrency == 10


class TestToolsNodeMethods:
    """ToolsNode 方法测试"""

    def test_tools_node_creation(self):
        from src.core.ark.nodes.tools import ToolsNode

        mcp = MagicMock()
        node = ToolsNode(mcp_client=mcp)
        assert node.max_concurrency == 10


class TestToolsNodeLastBatch:
    """ToolsNode 剩余测试"""

    def test_tools_node_creation_with_concurrency(self):
        from unittest.mock import MagicMock

        from src.core.ark.nodes.tools import ToolsNode

        mcp = MagicMock()
        node = ToolsNode(mcp_client=mcp, max_concurrency=5)
        assert node.max_concurrency == 5


class TestToolsNodeTier1:
    """ToolsNode 全覆盖"""

    def test_tools_node_creation_defaults(self):
        from src.core.ark.nodes.tools import ToolsNode

        mcp = MagicMock()
        node = ToolsNode(mcp_client=mcp)
        assert node.max_concurrency == 10
        assert node.mcp_client is mcp


class TestToolsNodeFinal:
    """ToolsNode 最后覆盖"""

    def test_tools_node_custom_concurrency(self):
        from src.core.ark.nodes.tools import ToolsNode

        mcp = MagicMock()
        node = ToolsNode(mcp_client=mcp, max_concurrency=3)
        assert node.max_concurrency == 3
