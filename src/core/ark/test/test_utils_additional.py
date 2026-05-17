"""Additional ARK utils tests migrated from legacy `src/test` files."""


class TestArkUtilsTier1:
    """ark/utils.py 全覆盖"""

    def test_create_agent_node(self):
        from src.core.ark.state import MultiAgentState
        from src.core.ark.utils import create_agent_node

        async def simple_agent(state: MultiAgentState):
            return "Hello from agent"

        node_func = create_agent_node("test_agent", simple_agent)
        assert node_func is not None


class TestArkUtilsFinal:
    """ark/utils.py 最后覆盖"""

    def test_create_agent_node_str_result(self):
        from src.core.ark.state import MultiAgentState
        from src.core.ark.utils import create_agent_node

        def simple_agent(state: MultiAgentState):
            return "Hello from agent"

        node_func = create_agent_node("test_agent", simple_agent)
        assert node_func is not None
