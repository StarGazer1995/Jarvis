import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, ANY
from langchain_core.messages import HumanMessage, AIMessage

from src.core.ark.engine import ARKEngine, ARKState, TaskStatus, Task
from src.core.config.server import SimpleMCPServerConfig

class TestARKEngine(unittest.IsolatedAsyncioTestCase):
    
    async def asyncSetUp(self):
        # Patch dependencies that are instantiated in __init__
        self.mcp_client_patcher = patch('src.core.ark.engine.ARKMCPClient')
        self.context_manager_patcher = patch('src.core.ark.engine.ConversationContext')
        self.prompt_manager_patcher = patch('src.core.ark.engine.PromptManager')
        
        self.mock_mcp_client_cls = self.mcp_client_patcher.start()
        self.mock_context_manager_cls = self.context_manager_patcher.start()
        self.mock_prompt_manager_cls = self.prompt_manager_patcher.start()
        
        # Setup mock instances
        self.mock_mcp_client = self.mock_mcp_client_cls.return_value
        self.mock_context_manager = self.mock_context_manager_cls.return_value
        self.mock_prompt_manager = self.mock_prompt_manager_cls.return_value
        
        # Patch create_ark_graph and ToolsNode for initialize
        self.create_graph_patcher = patch('src.core.ark.engine.create_ark_graph')
        self.tools_node_patcher = patch('src.core.ark.engine.ToolsNode')
        
        self.mock_create_graph = self.create_graph_patcher.start()
        self.mock_tools_node_cls = self.tools_node_patcher.start()
        self.mock_tools_node = self.mock_tools_node_cls.return_value
        
        # Initialize engine
        self.engine = ARKEngine(config={"enable_llm": True})
        
        # Setup mock mcp client behavior
        self.mock_mcp_client.sessions = {}
        self.mock_mcp_client.connect_to_server = AsyncMock(return_value=True)
        self.mock_mcp_client.discover_tools = AsyncMock(return_value=[])
        
    async def asyncTearDown(self):
        self.mcp_client_patcher.stop()
        self.context_manager_patcher.stop()
        self.prompt_manager_patcher.stop()
        self.create_graph_patcher.stop()
        self.tools_node_patcher.stop()
        
    async def test_initialize_success(self):
        """Test successful initialization of the engine."""
        # Mock mcp client discover tools to return some tools
        self.mock_mcp_client.sessions = {"server1": "session1"}
        self.mock_mcp_client.discover_tools.return_value = [{"name": "tool1", "description": "desc"}]
        
        # Mock graph creation
        mock_graph = AsyncMock()
        self.mock_create_graph.return_value = mock_graph
        
        server_config = SimpleMCPServerConfig(name="test_server", command="echo", args=[])
        
        # Run initialize
        success = await self.engine.initialize(mcp_servers=[server_config])
        
        # Verify
        self.assertTrue(success)
        self.assertEqual(self.engine.state, ARKState.READY)
        
        # Check MCP connection
        self.mock_mcp_client.connect_to_server.assert_called_with(server_config)
        
        # Check tool discovery
        self.mock_mcp_client.discover_tools.assert_called()
        self.assertIn("tool1", self.engine.available_tools)
        
        # Check LangGraph initialization
        self.mock_create_graph.assert_called_once()
        self.mock_tools_node_cls.assert_called_with(self.mock_mcp_client)
        self.assertIsNotNone(self.engine.graph)
        
    async def test_initialize_failure(self):
        """Test initialization failure handling."""
        # Make mcp_client.connect_to_server raise exception (or just fail later)
        # Let's make create_ark_graph fail
        self.mock_create_graph.side_effect = Exception("Graph creation failed")
        
        success = await self.engine.initialize()
        
        self.assertFalse(success)
        self.assertEqual(self.engine.state, ARKState.ERROR)
        
    async def test_process_input_success_add_task(self):
        """Test process_input where a task is added via LangGraph."""
        # Setup engine as ready
        self.engine.state = ARKState.READY
        mock_graph = AsyncMock()
        self.engine.graph = mock_graph
        
        # Mock graph response
        final_state = {
            "messages": [
                HumanMessage(content="Add task"),
                AIMessage(content="I added the task.")
            ],
            "todo_list": [
                {"id": "1", "description": "New Task", "status": "pending"}
            ],
            "sender": "agent"
        }
        mock_graph.ainvoke.return_value = final_state
        
        # Run process_input
        response = await self.engine.process_input("Add task")
        
        # Verify
        self.assertEqual(response, "I added the task.")
        
        # Check todo list update
        self.assertEqual(len(self.engine.todo_list), 1)
        self.assertEqual(self.engine.todo_list[0].description, "New Task")
        self.assertEqual(self.engine.todo_list[0].status, TaskStatus.PENDING)
        
        # Check graph invocation
        mock_graph.ainvoke.assert_called_once()
        call_args = mock_graph.ainvoke.call_args[0][0]
        self.assertEqual(call_args["user_input"], "Add task")
        self.assertIsInstance(call_args["messages"][0], HumanMessage)
        
        # Check context update
        self.mock_context_manager.add_exchange.assert_called()
        
    async def test_process_input_not_ready(self):
        """Test process_input when engine is not ready."""
        self.engine.state = ARKState.INITIALIZING # Not READY
        
        response = await self.engine.process_input("Hi")
        
        self.assertIn("not ready", response.lower())
        self.mock_context_manager.add_exchange.assert_not_called()
        
    async def test_process_input_graph_error(self):
        """Test process_input when graph execution fails."""
        self.engine.state = ARKState.READY
        mock_graph = AsyncMock()
        self.engine.graph = mock_graph
        
        mock_graph.ainvoke.side_effect = Exception("Graph execution failed")
        
        response = await self.engine.process_input("Hi")
        
        self.assertIn("Error executing request", response)
        
    async def test_shutdown(self):
        """Test shutdown sequence."""
        # Setup mock MCP client to have close method
        self.mock_mcp_client.close = AsyncMock()
        
        await self.engine.shutdown()
        
        # Verify MCP client closed
        self.mock_mcp_client.close.assert_called_once()
        
        # Verify conversation exported
        self.mock_context_manager.export_conversation.assert_called_once()
        
    def test_get_status(self):
        """Test get_status method."""
        self.engine.state = ARKState.READY
        self.engine.available_tools = {"t1": {}}
        self.engine.todo_list = [Task(id="1", description="t", status=TaskStatus.PENDING)]
        
        status = self.engine.get_status()
        
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["available_tools"], 1)
        self.assertEqual(status["todo_count"], 1)

    async def test_task_to_dict(self):
        """Test Task.to_dict serialization."""
        task = Task(id="123", description="Test Task", status=TaskStatus.IN_PROGRESS, result="Partial")
        data = task.to_dict()
        self.assertEqual(data["id"], "123")
        self.assertEqual(data["description"], "Test Task")
        self.assertEqual(data["status"], "in_progress")
        self.assertEqual(data["result"], "Partial")

    async def test_legacy_methods(self):
        """Test legacy methods for compatibility."""
        # _get_system_prompt
        with patch('src.core.agent.react.ReActAgent._get_system_prompt', return_value="sys prompt"):
            prompt = self.engine._get_system_prompt()
            self.assertEqual(prompt, "sys prompt")

        # execute_tool
        with patch('src.core.agent.react.ReActAgent.execute_tool', new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = "result"
            res = await self.engine.execute_tool("tool", "arg")
            self.assertEqual(res, "result")
            mock_exec.assert_called_with("tool", "arg")

        # _manage_tasks
        res = self.engine._manage_tasks("add")
        self.assertIn("Legacy", res)

    async def test_initialize_partial_failures(self):
        """Test initialization with partial failures."""
        # Mock mcp client connect failure
        self.mock_mcp_client.connect_to_server.return_value = False
        
        # Mock discover_tools partial failure
        self.mock_mcp_client.sessions = {"s1": "sess1", "s2": "sess2"}
        
        async def discover_side_effect(server):
            if server == "s1":
                return [{"name": "t1"}]
            raise Exception("Discovery failed")
            
        self.mock_mcp_client.discover_tools.side_effect = discover_side_effect
        
        server_config = SimpleMCPServerConfig(name="test_server", command="echo", args=[])
        
        # Run initialize
        success = await self.engine.initialize(mcp_servers=[server_config])
        
        self.assertTrue(success) # Should still succeed even if one server fails discovery
        self.assertIn("t1", self.engine.available_tools)
        # Usage stats should be initialized for t1
        self.assertIn("t1", self.engine.tool_usage_stats)

    async def test_initialize_super_fail(self):
        """Test initialization failure when super().initialize() fails."""
        with patch('src.core.agent.react.ReActAgent.initialize', new_callable=AsyncMock) as mock_super_init:
            mock_super_init.return_value = False
            success = await self.engine.initialize()
            self.assertFalse(success)

    async def test_initialize_discover_tools_crash(self):
        """Test initialization when _discover_tools crashes completely."""
        # Patch _discover_tools to raise exception
        with patch.object(self.engine, '_discover_tools', side_effect=Exception("Crash")):
            # Initialize catches exception and returns False
            success = await self.engine.initialize()
            self.assertFalse(success)
            self.assertEqual(self.engine.state, ARKState.ERROR)

    async def test_discover_tools_exception(self):
        """Test _discover_tools exception handling (internal loop)."""
        # We need to make sessions access raise exception
        # But sessions is a property or dict on mock.
        # If we can't easily mock attribute access raising, we can try mocking discover_tools to fail
        # But we want the OUTER try/catch in _discover_tools (lines 276-278)
        
        # The outer try catch wraps the whole function body basically.
        # So if any line inside raises, it catches.
        
        # Let's mock mcp_client.sessions.keys() to raise exception
        # We need to mock the sessions attribute itself to be something that raises on keys()
        
        mock_sessions = MagicMock()
        mock_sessions.keys.side_effect = Exception("Sessions Access Fail")
        self.mock_mcp_client.sessions = mock_sessions
        
        await self.engine._discover_tools()
        
        # Should catch exception and reset available_tools
        self.assertEqual(self.engine.available_tools, {})
        # Logs should contain error
        # We can't easily check logs unless we mock logger, but code coverage will show it.

    async def test_process_input_edge_cases(self):
        """Test edge cases in process_input."""
        self.engine.state = ARKState.READY
        mock_graph = AsyncMock()
        self.engine.graph = mock_graph

        # 1. No graph
        self.engine.graph = None
        res = await self.engine.process_input("hi")
        self.assertIn("Error: LangGraph not initialized", res)
        self.engine.graph = mock_graph # Restore

        # 2. Invalid task status
        mock_graph.ainvoke.return_value = {
            "messages": [AIMessage(content="done")],
            "todo_list": [{"id": "1", "description": "d", "status": "INVALID_STATUS"}]
        }
        await self.engine.process_input("hi")
        self.assertEqual(self.engine.todo_list[0].status, TaskStatus.PENDING)

        # 3. Non-AIMessage response
        mock_graph.ainvoke.return_value = {
            "messages": [HumanMessage(content="user"), "Simple string message"],
        }
        res = await self.engine.process_input("hi")
        self.assertEqual(res, "Simple string message")

        # 3b. Object with content attribute but not AIMessage
        mock_msg = MagicMock()
        mock_msg.content = "Content Msg"
        mock_graph.ainvoke.return_value = {
            "messages": [mock_msg],
        }
        res = await self.engine.process_input("hi")
        self.assertEqual(res, "Content Msg")

        # 4. Context update failure (should log warning but not fail)
        mock_graph.ainvoke.return_value = {"messages": [AIMessage(content="ok")]}
        self.mock_context_manager.add_exchange.side_effect = Exception("Context fail")
        
        res = await self.engine.process_input("hi")
        self.assertEqual(res, "ok") # Should still return response

    async def test_get_engine_status(self):
        """Test get_engine_status full report."""
        self.engine.state = ARKState.READY
        self.mock_context_manager.get_session_stats.return_value = {"stats": "ok"}
        
        status = self.engine.get_engine_status()
        
        self.assertEqual(status["state"], "ready")
        self.assertIn("performance_metrics", status)
        self.assertIn("configuration", status)
        self.assertEqual(status["conversation_stats"], {"stats": "ok"})

    async def test_shutdown_exceptions(self):
        """Test shutdown with exceptions."""
        # Mock mcp_client.close missing but disconnect_all present
        del self.mock_mcp_client.close
        self.mock_mcp_client.disconnect_all = AsyncMock()
        self.mock_mcp_client.disconnect_all.side_effect = Exception("MCP Disconnect Fail") # Trigger catch block
        
        # Mock context export failure
        self.mock_context_manager.export_conversation.side_effect = Exception("Export fail")
        
        await self.engine.shutdown()
        
        self.mock_mcp_client.disconnect_all.assert_called_once()
        # Should complete without raising exception

    async def test_close_alias(self):
        """Test close alias calls shutdown."""
        with patch.object(self.engine, 'shutdown', new_callable=AsyncMock) as mock_shutdown:
            await self.engine.close()
            mock_shutdown.assert_called_once()
