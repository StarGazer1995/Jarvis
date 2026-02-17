# Building AI Agents: Architecture and Best Practices

*Discussion Document - Updated: February 17, 2026*

## Overview

This document captures key concepts, patterns, and architectural decisions for building effective AI agents, with specific reference to the Project Jarvis framework. The architecture centers around **Jarvis** (the AI agent interface) powered by **ARK** (the core MCP-enabled engine that provides all capabilities and tool integrations).

## Project Jarvis Architecture: Jarvis + ARK

**Jarvis** serves as the conversational AI interface that users interact with, while **ARK** (Agent Reactor Kernel) is the core engine that powers all of Jarvis's capabilities through **LangGraph orchestration**, **MCP integration**, and intelligent decision-making.

## Core Components of an AI Agent

### 1. Agent Orchestration: LangGraph (Current Implementation)
Instead of a simple while loop, Project Jarvis now utilizes **LangGraph** to manage complex agent workflows. This allows for:
- **State Management**: Maintaining conversation history, tool outputs, and internal reasoning steps.
- **Dynamic Routing**: Conditionally moving between reasoning (MasterNode) and execution (ToolsNode).
- **Multi-Agent Coordination**: Supporting Supervisor patterns where a Master agent delegates tasks to specialized workers.

**Current Implementation Pattern:**
```python
def create_ark_graph(llm_manager: LLMManager, mcp_client: ARKMCPClient):
    """Constructs the LangGraph for ARK engine."""
    workflow = StateGraph(JarvisState)
    
    # Initialize Nodes
    master_node = MasterNode(llm_manager)
    tools_node = ToolsNode(mcp_client)
    
    # Add Nodes
    workflow.add_node("master", master_node)
    workflow.add_node("tools", tools_node)
    
    # Define Edges
    workflow.set_entry_point("master")
    
    # Conditional edge based on tool calls
    workflow.add_conditional_edges(
        "master",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    
    # Loop back to master after tool execution
    workflow.add_edge("tools", "master")
    
    return workflow.compile()
```

### 2. Reasoning Engine: MasterNode
The **MasterNode** acts as the central brain. It:
- Processes user input and context.
- Decides whether to call a tool or respond directly.
- Handles multi-turn reasoning (ReAct pattern).
- Can delegate sub-tasks to specialized agents in a multi-agent setup.

```python
class MasterNode:
    """The main reasoning node (Agent) for Jarvis."""
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager

    async def __call__(self, state: JarvisState) -> Dict[str, Any]:
        # Invoke LLM with tools binding
        response = await self.llm_manager.ainvoke(state["messages"])
        return {"messages": [response]}
```

### 3. Execution Engine: ToolsNode
The **ToolsNode** is responsible for executing actions. It handles:
- **Local Tools**: Built-in functions like `manage_tasks` for todo list management.
- **MCP Tools**: External tools provided by Model Context Protocol servers (e.g., filesystem, web search).

```python
class ToolsNode:
    """Node responsible for executing tool calls."""
    def __init__(self, mcp_client: ARKMCPClient):
        self.mcp_client = mcp_client

    async def __call__(self, state: JarvisState) -> Dict[str, Any]:
        # Execute tool calls from the last message
        results = []
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls"):
            for tool_call in last_message.tool_calls:
                if tool_call["name"] in self.local_tools:
                    result = await self.local_tools[tool_call["name"]](**tool_call["args"])
                else:
                    result = await self.mcp_client.execute_tool(tool_call["name"], tool_call["args"])
                results.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"], name=tool_call["name"]))
        return {"messages": results}
```

## Agent Architecture Patterns

### 1. ReAct Pattern (Native Support)
The core loop `MasterNode -> ToolsNode -> MasterNode` naturally implements the **ReAct** (Reasoning + Acting) pattern. The model reasons about the task, emits a tool call (Act), the system executes it (Observe), and the model processes the result (Reason again).

### 2. Multi-Agent Systems (Supervisor Pattern)
Jarvis supports a **Supervisor** architecture where the MasterNode delegates work to specialized agents.
- **Supervisor**: Routes tasks to workers based on descriptions.
- **Workers**: Specialized agents (e.g., "Coder", "Researcher") with focused toolsets.
- **Graph**: The LangGraph definition manages the state transfer between Supervisor and Workers.

## MCP Integration Architecture

### 1. ARK MCP Client
The `ARKMCPClient` manages connections to multiple MCP servers, handling:
- **Server Configuration**: Loading from JSON/YAML configs.
- **Tool Discovery**: Listing available tools from all connected servers.
- **Execution**: Routing calls to the appropriate server.

```python
class ARKMCPClient:
    """ARK's MCP client for communicating with MCP servers."""
    async def execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
        # Find the server that provides this tool
        server = self.tool_registry.get_server(name)
        if not server:
            raise ValueError(f"Tool {name} not found")
        return await server.call_tool(name, args)
```

## Security and Reliability

### 1. ARK Security Manager
The `ARKSecurityManager` provides granular control over tool execution:
- **Policies**: Define security levels (Low, Medium, High, Critical).
- **Permissions**: Control access to specific capabilities (Filesystem, Network, etc.).
- **Rate Limiting**: Prevent abuse via `RateLimiter`.
- **Audit Logging**: Track all security-relevant events.

```python
class ARKSecurityManager:
    """Main security manager for ARK tool validation."""
    def __init__(self):
        self.policies = self._load_default_policies()
        self.rate_limiter = RateLimiter()
        self.audit_logger = AuditLogger()

    async def validate_tool_execution(self, request: ValidationRequest) -> ValidationResponse:
        # Check permissions, rate limits, and policy compliance
        if not self.rate_limiter.check(request.user_id):
            return ValidationResponse(allowed=False, reason="Rate limit exceeded")
        # ... additional checks ...
        return ValidationResponse(allowed=True)
```

### 2. Reliability Mechanisms
- **RetryHandler**: Automatically retries failed LLM or tool calls with exponential backoff.
- **RateLimitHandler**: Manages API quota usage to prevent 429 errors.
- **State Recovery**: LangGraph's checkpointing allows resuming interrupted workflows (planned feature).

## Next Steps for Project Jarvis

### Completed Milestones
- [x] **Basic MCP Integration**: Filesystem, basic tools.
- [x] **Workflow Engine**: Implemented via LangGraph.
- [x] **Security Framework**: ARKSecurityManager with policies and audit logging.
- [x] **Tool Ecosystem**: Support for standard MCP servers.

### Future Enhancements (Phase 3+)
1.  **Advanced Multi-Agent Orchestration**
    - [ ] Implement complex hierarchical agent teams.
    - [ ] Dynamic agent spawning based on task needs.

2.  **Memory Optimization**
    - [ ] Implement long-term memory (Vector Store) for user preferences and past interactions.
    - [ ] Context window management for long conversations.

3.  **User Interface**
    - [ ] Full integration with Chainlit for a rich web UI.
    - [ ] Visualization of the agent's reasoning graph.

4.  **Production Hardening**
    - [ ] Comprehensive integration testing suite.
    - [ ] Docker containerization for easy deployment.

---
*This document serves as a living guide for ARK-powered Jarvis development.*
