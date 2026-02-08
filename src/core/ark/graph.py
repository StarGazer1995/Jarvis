from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage

from .state import JarvisState
from .nodes.master import MasterNode
from .nodes.tools import ToolsNode
from ..llm.client import LLMManager
from ..mcp.client import ARKMCPClient

def create_ark_graph(llm_manager: LLMManager, mcp_client: ARKMCPClient, tools_node_instance: ToolsNode = None):
    """
    Constructs the LangGraph for ARK engine.
    
    Args:
        llm_manager: The LLM Manager
        mcp_client: The MCP Client
        tools_node_instance: Optional pre-initialized ToolsNode. If None, one will be created.
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
        messages = state["messages"]
        if not messages:
            return END
            
        last_message = messages[-1]
        
        # If the last message is an AIMessage and has tool_calls, go to tools
        if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        # Otherwise, end
        return END

    workflow.add_conditional_edges(
        "master",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    
    # After tools execution, go back to master to interpret results
    workflow.add_edge("tools", "master")
    
    # Compile the graph
    return workflow.compile()
