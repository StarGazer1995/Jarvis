from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph

from ..llm.client import LLMManager
from ..mcp.client import ARKMCPClient
from .nodes.master import MasterNode
from .nodes.tools import ToolsNode
from .state import JarvisState, MultiAgentState
from .utils import AgentSpec


def _get_last_ai_tool_calls(messages: list) -> list | None:
    if not messages:
        return None
    last_message = messages[-1]
    if not isinstance(last_message, AIMessage):
        return None
    tool_calls = getattr(last_message, "tool_calls", None)
    return tool_calls or None


def _get_first_tool_name(messages: list) -> str | None:
    tool_calls = _get_last_ai_tool_calls(messages)
    if not tool_calls:
        return None
    first = tool_calls[0]
    return first.get("name")


def create_ark_graph(
    llm_manager: LLMManager,
    mcp_client: ARKMCPClient,
    tools_node_instance: ToolsNode = None,
):
    """
    Constructs the LangGraph for ARK engine.

    Args:
        llm_manager: The LLM Manager
        mcp_client: The MCP Client
        tools_node_instance: Optional pre-initialized ToolsNode.
            If None, one will be created.
    """
    workflow = StateGraph(JarvisState)

    # Initialize Nodes
    master_node = MasterNode(llm_manager)
    tools_node = tools_node_instance or ToolsNode(mcp_client)

    # Add Nodes
    workflow.add_node("master", master_node)
    workflow.add_node("tools", tools_node)

    # Define Edges
    workflow.set_entry_point("master")

    def should_continue(state: JarvisState):
        if _get_last_ai_tool_calls(state["messages"]):
            return "tools"

        return END

    workflow.add_conditional_edges(
        "master", should_continue, {"tools": "tools", END: END}
    )

    # After tools execution, go back to master to interpret results
    workflow.add_edge("tools", "master")

    # Compile the graph
    return workflow.compile()


def create_supervisor_graph(llm_manager: LLMManager, agents: list[AgentSpec]):
    """
    Builds the Supervisor StateGraph using MasterNode as the Orchestrator.
    This replaces the legacy SupervisorNode.
    """
    workflow = StateGraph(MultiAgentState)

    # Initialize MasterNode with Agents
    master_node = MasterNode(llm_manager, agents=agents)

    # Add Supervisor (Master) Node
    workflow.add_node("supervisor", master_node)

    # Add Worker Nodes
    for agent in agents:
        workflow.add_node(agent.name, agent.node)

    # Define Edges
    for agent in agents:
        # Workers always go back to supervisor
        workflow.add_edge(agent.name, "supervisor")

    # Routing Logic
    def route_supervisor(state: MultiAgentState):
        action_name = _get_first_tool_name(state["messages"])
        if not action_name:
            return END
        for agent in agents:
            if agent.name == action_name:
                return agent.name
        return END

    # Build conditional map
    conditional_map = {a.name: a.name for a in agents}
    conditional_map[END] = END

    workflow.add_conditional_edges("supervisor", route_supervisor, conditional_map)

    workflow.set_entry_point("supervisor")

    return workflow.compile()
