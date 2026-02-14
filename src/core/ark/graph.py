from typing import List
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage

from .state import JarvisState, MultiAgentState
from .nodes.master import MasterNode
from .nodes.tools import ToolsNode
from ..llm.client import LLMManager
from ..mcp.client import ARKMCPClient
from .utils import AgentSpec

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

def create_supervisor_graph(llm_manager: LLMManager, agents: List[AgentSpec]):
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
        messages = state["messages"]
        if not messages:
            return END
            
        last_message = messages[-1]
        
        # Check for tool calls (actions)
        if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_call = last_message.tool_calls[0]
            action_name = tool_call["name"]
            
            # If action matches an agent name, route to it
            for agent in agents:
                if agent.name == action_name:
                    return agent.name
            
            # If action is 'FINISH' (though MasterNode usually does Final Answer)
            # Or if it's unknown.
            return END
            
        return END

    # Build conditional map
    conditional_map = {a.name: a.name for a in agents}
    conditional_map[END] = END
    
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        conditional_map
    )
    
    workflow.set_entry_point("supervisor")
    
    return workflow.compile()
