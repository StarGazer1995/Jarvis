"""
Tests for the main module.

This module contains comprehensive tests for the main entry point,
including command-line interface, interactive mode, demo mode,
and argument parsing functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock, mock_open
import argparse
import logging
import sys
import signal
import json
import asyncio
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.main import JarvisApp, main, cli_main


class TestJarvisApp:
    """Test cases for JarvisApp class."""
    
    def test_jarvis_app_init(self):
        """Test JarvisApp initialization."""
        app = JarvisApp()
        assert app.agent is None
        assert not app.running
        assert app.logger is not None
    
    def test_setup_signal_handlers(self):
        """Test signal handlers setup."""
        app = JarvisApp()
        
        with patch('signal.signal') as mock_signal:
            app.setup_signal_handlers()
            
            # Verify signal handlers were set
            assert mock_signal.call_count == 2
            mock_signal.assert_any_call(signal.SIGINT, mock_signal.call_args_list[0][0][1])
            mock_signal.assert_any_call(signal.SIGTERM, mock_signal.call_args_list[1][0][1])
    
    def test_signal_handler_functionality(self):
        """Test signal handler sets running to False."""
        app = JarvisApp()
        app.running = True
        
        with patch('signal.signal') as mock_signal:
            app.setup_signal_handlers()
            
            # Get the signal handler function
            signal_handler = mock_signal.call_args_list[0][0][1]
            
            # Call the signal handler
            signal_handler(signal.SIGINT, None)
            
            # Verify running is set to False
            assert not app.running
    
    def test_jarvis_app_parse_arguments_defaults(self):
        """Test JarvisApp parse_arguments with default values."""
        app = JarvisApp()
        
        with patch('sys.argv', ['main.py']):
            args = app.parse_arguments()
            assert args.log_level == 'INFO'
            assert not args.interactive
            assert not args.demo
            assert args.config_file is None
            assert args.max_history == 100
            assert args.confidence_threshold == 0.7
    
    def test_jarvis_app_parse_arguments_interactive(self):
        """Test JarvisApp parse_arguments with interactive mode."""
        app = JarvisApp()
        
        with patch('sys.argv', ['main.py', '--interactive']):
            args = app.parse_arguments()
            assert args.interactive
    
    def test_jarvis_app_parse_arguments_demo(self):
        """Test JarvisApp parse_arguments with demo mode."""
        app = JarvisApp()
        
        with patch('sys.argv', ['main.py', '--demo']):
            args = app.parse_arguments()
            assert args.demo
    
    def test_parse_arguments_all_options(self):
        """Test parsing all command-line options."""
        app = JarvisApp()
        
        with patch('sys.argv', [
            'main.py', 
            '--log-level', 'DEBUG',
            '--interactive',
            '--demo',
            '--config-file', 'config.json',
            '--max-history', '50',
            '--confidence-threshold', '0.8'
        ]):
            args = app.parse_arguments()
            assert args.log_level == 'DEBUG'
            assert args.interactive
            assert args.demo
            assert args.config_file == 'config.json'
            assert args.max_history == 50
            assert args.confidence_threshold == 0.8
    
    def test_parse_arguments_version(self):
        """Test version argument."""
        app = JarvisApp()
        
        with patch('sys.argv', ['main.py', '--version']):
            with pytest.raises(SystemExit):
                app.parse_arguments()
    
    def test_load_config_basic(self):
        """Test basic configuration loading."""
        app = JarvisApp()
        args = argparse.Namespace(
            log_level='DEBUG',
            max_history=50,
            confidence_threshold=0.8,
            config_file=None
        )
        
        config = app.load_config(args)
        assert config.log_level == 'DEBUG'
        assert config.max_conversation_history == 50
        assert config.confidence_threshold == 0.8
        assert len(config.mcp_servers) == 1
    
    def test_load_config_with_file(self):
        """Test configuration loading with config file."""
        app = JarvisApp()
        args = argparse.Namespace(
            log_level='INFO',
            max_history=100,
            confidence_threshold=0.7,
            config_file='test_config.json'
        )
        
        config_data = {
            'log_level': 'ERROR',
            'max_conversation_history': 200
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(config_data))):
            with patch('json.load', return_value=config_data):
                config = app.load_config(args)
                assert config.log_level == 'ERROR'
                assert config.max_conversation_history == 200
    
    def test_load_config_file_error(self):
        """Test configuration loading with file error."""
        app = JarvisApp()
        args = argparse.Namespace(
            log_level='INFO',
            max_history=100,
            confidence_threshold=0.7,
            config_file='nonexistent.json'
        )
        
        with patch('builtins.open', side_effect=FileNotFoundError()):
            config = app.load_config(args)
            # Should still work with default values
            assert config.log_level == 'INFO'
    
    @pytest.mark.asyncio
    async def test_run_interactive_mode(self):
        """Test interactive mode execution."""
        app = JarvisApp()
        app.agent = AsyncMock()
        app.agent.start_conversation.return_value = "Hello! I'm Jarvis."
        app.agent.process_message.return_value = "I understand."
        app.agent.end_conversation.return_value = "Goodbye!"
        
        # Mock input to simulate user typing 'quit'
        with patch('builtins.input', side_effect=['quit']):
            await app.run_interactive_mode()
            
        app.agent.start_conversation.assert_called_once()
        app.agent.end_conversation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_interactive_mode_help(self):
        """Test interactive mode help command."""
        app = JarvisApp()
        app.agent = AsyncMock()
        app.agent.start_conversation.return_value = "Hello!"
        
        with patch('builtins.input', side_effect=['help', 'quit']):
            with patch.object(app, '_show_help') as mock_help:
                await app.run_interactive_mode()
                mock_help.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_interactive_mode_status(self):
        """Test interactive mode status command."""
        app = JarvisApp()
        app.agent = AsyncMock()
        app.agent.start_conversation.return_value = "Hello!"
        
        with patch('builtins.input', side_effect=['status', 'quit']):
            with patch.object(app, '_show_status') as mock_status:
                await app.run_interactive_mode()
                mock_status.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_interactive_mode_tools(self):
        """Test interactive mode tools command."""
        app = JarvisApp()
        app.agent = AsyncMock()
        app.agent.start_conversation.return_value = "Hello!"
        
        with patch('builtins.input', side_effect=['tools', 'quit']):
            with patch.object(app, '_show_tools') as mock_tools:
                await app.run_interactive_mode()
                mock_tools.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_interactive_mode_export(self):
        """Test interactive mode export command."""
        app = JarvisApp()
        app.agent = AsyncMock()
        app.agent.start_conversation.return_value = "Hello!"
        
        with patch('builtins.input', side_effect=['export', 'quit']):
            with patch.object(app, '_export_conversation') as mock_export:
                await app.run_interactive_mode()
                mock_export.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_interactive_mode_keyboard_interrupt(self):
        """Test interactive mode with keyboard interrupt."""
        app = JarvisApp()
        app.agent = AsyncMock()
        app.agent.start_conversation.return_value = "Hello!"
        
        with patch('builtins.input', side_effect=KeyboardInterrupt()):
            await app.run_interactive_mode()
    
    @pytest.mark.asyncio
    async def test_run_interactive_mode_eof_error(self):
        """Test interactive mode with EOF error."""
        app = JarvisApp()
        app.agent = AsyncMock()
        app.agent.start_conversation.return_value = "Hello!"
        
        with patch('builtins.input', side_effect=EOFError()):
            await app.run_interactive_mode()
    
    @pytest.mark.asyncio
    async def test_run_interactive_mode_exception(self):
        """Test interactive mode with general exception."""
        app = JarvisApp()
        app.agent = AsyncMock()
        app.agent.start_conversation.return_value = "Hello!"
        app.agent.process_message.side_effect = Exception("Test error")
        
        with patch('builtins.input', side_effect=['test message', 'quit']):
            await app.run_interactive_mode()
    
    @pytest.mark.asyncio
    async def test_run_demo_mode(self):
        """Test demo mode execution."""
        app = JarvisApp()
        app.agent = AsyncMock()
        app.agent.start_conversation.return_value = "Hello! I'm Jarvis."
        app.agent.process_message.return_value = "Demo response"
        app.agent.end_conversation.return_value = "Goodbye!"
        
        with patch('asyncio.sleep'):  # Speed up the test
            await app.run_demo_mode()
            
        app.agent.start_conversation.assert_called_once()
        app.agent.end_conversation.assert_called_once()
        # Should process 5 demo messages
        assert app.agent.process_message.call_count == 5
    
    def test_show_help(self):
        """Test help display."""
        app = JarvisApp()
        
        with patch('builtins.print') as mock_print:
            app._show_help()
            mock_print.assert_called()
    
    @pytest.mark.asyncio
    async def test_show_status(self):
        """Test status display."""
        app = JarvisApp()
        app.agent = Mock()
        app.agent.get_status.return_value = {
            'agent': {
                'name': 'Jarvis',
                'version': '1.0.0',
                'is_running': True,
                'conversation_active': False
            },
            'configuration': {
                'mcp_servers_count': 1
            }
        }
        app.agent.health_check = AsyncMock(return_value={'overall': 'healthy'})
        app.agent.get_available_tools.return_value = ['tool1', 'tool2']
        
        with patch('builtins.print') as mock_print:
            await app._show_status()
            mock_print.assert_called()
    
    @pytest.mark.asyncio
    async def test_show_status_exception(self):
        """Test status display with exception."""
        app = JarvisApp()
        app.agent = Mock()
        app.agent.get_status.side_effect = Exception("Status error")
        
        with patch('builtins.print') as mock_print:
            await app._show_status()
            mock_print.assert_called()
    
    def test_show_tools(self):
        """Test tools display."""
        app = JarvisApp()
        app.agent = Mock()
        app.agent.get_available_tools.return_value = ['tool1', 'tool2']
        app.agent.get_tool_usage_stats.return_value = {'tool1': 5, 'tool2': 3}
        
        with patch('builtins.print') as mock_print:
            app._show_tools()
            mock_print.assert_called()
    
    def test_show_tools_empty(self):
        """Test tools display with no tools."""
        app = JarvisApp()
        app.agent = Mock()
        app.agent.get_available_tools.return_value = []
        app.agent.get_tool_usage_stats.return_value = {}
        
        with patch('builtins.print') as mock_print:
            app._show_tools()
            mock_print.assert_called()
    
    def test_show_tools_exception(self):
        """Test tools display with exception."""
        app = JarvisApp()
        app.agent = Mock()
        app.agent.get_available_tools.side_effect = Exception("Tools error")
        
        with patch('builtins.print') as mock_print:
            app._show_tools()
            mock_print.assert_called()
    
    def test_export_conversation(self):
        """Test conversation export."""
        app = JarvisApp()
        app.agent = Mock()
        app.agent.get_conversation_export.return_value = "Conversation history"
        
        with patch('builtins.print') as mock_print:
            app._export_conversation()
            mock_print.assert_called()
            app.agent.get_conversation_export.assert_called_with("text")
    
    def test_export_conversation_exception(self):
        """Test conversation export with exception."""
        app = JarvisApp()
        app.agent = Mock()
        app.agent.get_conversation_export.side_effect = Exception("Export error")
        
        with patch('builtins.print') as mock_print:
            app._export_conversation()
            mock_print.assert_called()
    
    @pytest.mark.asyncio
    async def test_run_success_default_mode(self):
        """Test successful run in default mode."""
        app = JarvisApp()
        
        with patch.object(app, 'parse_arguments') as mock_parse:
            with patch.object(app, 'setup_signal_handlers'):
                with patch.object(app, 'load_config') as mock_load_config:
                    with patch('src.main.JarvisAgent') as mock_agent_class:
                        mock_args = Mock()
                        mock_args.demo = False
                        mock_args.interactive = False
                        mock_parse.return_value = mock_args
                        
                        mock_config = Mock()
                        mock_config.version = '1.0.0'
                        mock_load_config.return_value = mock_config
                        
                        mock_agent = Mock()
                        mock_agent.initialize = AsyncMock(return_value=True)
                        mock_agent.start = AsyncMock(return_value=None)
                        mock_agent.stop = AsyncMock(return_value=None)
                        mock_agent.get_status.return_value = {
                            'agent': {'name': 'Jarvis', 'version': '1.0.0'}
                        }
                        mock_agent.get_available_tools.return_value = []
                        mock_agent.health_check = AsyncMock(return_value={'overall': 'healthy'})
                        mock_agent_class.return_value = mock_agent
                        
                        with patch('builtins.print'):  # Suppress print output during test
                            result = await app.run()
                        assert result == 0
    
    @pytest.mark.asyncio
    async def test_run_success_demo_mode(self):
        """Test successful run in demo mode."""
        app = JarvisApp()
        
        with patch.object(app, 'parse_arguments') as mock_parse:
            with patch.object(app, 'setup_signal_handlers'):
                with patch.object(app, 'load_config') as mock_load_config:
                    with patch.object(app, 'run_demo_mode') as mock_demo:
                        with patch('src.main.JarvisAgent') as mock_agent_class:
                            mock_args = Mock()
                            mock_args.demo = True
                            mock_args.interactive = False
                            mock_parse.return_value = mock_args
                            
                            mock_config = Mock()
                            mock_config.version = '1.0.0'
                            mock_load_config.return_value = mock_config
                            
                            mock_agent = Mock()
                            mock_agent.initialize = AsyncMock(return_value=True)
                            mock_agent.start = AsyncMock(return_value=None)
                            mock_agent.stop = AsyncMock(return_value=None)
                            mock_agent.health_check = AsyncMock(return_value={'overall': 'healthy'})
                            
                            mock_agent_class.return_value = mock_agent
                            
                            result = await app.run()
                            assert result == 0
                            mock_demo.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_success_interactive_mode(self):
        """Test successful run in interactive mode."""
        app = JarvisApp()
        
        with patch.object(app, 'parse_arguments') as mock_parse:
            with patch.object(app, 'setup_signal_handlers'):
                with patch.object(app, 'load_config') as mock_load_config:
                    with patch.object(app, 'run_interactive_mode') as mock_interactive:
                        with patch('src.main.JarvisAgent') as mock_agent_class:
                            mock_args = Mock()
                            mock_args.demo = False
                            mock_args.interactive = True
                            mock_parse.return_value = mock_args
                            
                            mock_config = Mock()
                            mock_config.version = '1.0.0'
                            mock_load_config.return_value = mock_config
                            
                            mock_agent = Mock()
                            mock_agent.initialize = AsyncMock(return_value=True)
                            mock_agent.start = AsyncMock(return_value=None)
                            mock_agent.stop = AsyncMock(return_value=None)
                            mock_agent.health_check = AsyncMock(return_value={'overall': 'healthy'})
                            
                            mock_agent_class.return_value = mock_agent
                            
                            result = await app.run()
                            assert result == 0
                            mock_interactive.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_initialization_failure(self):
        """Test run with agent initialization failure."""
        app = JarvisApp()
        
        with patch.object(app, 'parse_arguments') as mock_parse:
            with patch.object(app, 'setup_signal_handlers'):
                with patch.object(app, 'load_config') as mock_load_config:
                    with patch('src.main.JarvisAgent') as mock_agent_class:
                        mock_args = Mock()
                        mock_parse.return_value = mock_args
                        
                        mock_config = Mock()
                        mock_load_config.return_value = mock_config
                        
                        mock_agent = AsyncMock()
                        mock_agent.initialize.return_value = False
                        mock_agent_class.return_value = mock_agent
                        
                        result = await app.run()
                        assert result == 1
    
    @pytest.mark.asyncio
    async def test_run_keyboard_interrupt(self):
        """Test run with keyboard interrupt."""
        app = JarvisApp()
        
        with patch.object(app, 'parse_arguments', side_effect=KeyboardInterrupt()):
            result = await app.run()
            assert result == 0
    
    @pytest.mark.asyncio
    async def test_run_exception(self):
        """Test run with general exception."""
        app = JarvisApp()
        
        with patch.object(app, 'parse_arguments', side_effect=Exception("Test error")):
            result = await app.run()
            assert result == 1
    
    @pytest.mark.asyncio
    async def test_run_cleanup_on_exception(self):
        """Test cleanup is called even when exception occurs."""
        app = JarvisApp()
        
        with patch.object(app, 'parse_arguments') as mock_parse:
            with patch.object(app, 'setup_signal_handlers'):
                with patch.object(app, 'load_config') as mock_load_config:
                    with patch('src.main.JarvisAgent') as mock_agent_class:
                        mock_args = Mock()
                        mock_parse.return_value = mock_args
                        
                        mock_config = Mock()
                        mock_load_config.return_value = mock_config
                        
                        mock_agent = AsyncMock()
                        mock_agent.initialize.return_value = True
                        mock_agent.start.side_effect = Exception("Start error")
                        mock_agent_class.return_value = mock_agent
                        
                        app.agent = mock_agent
                        
                        result = await app.run()
                        assert result == 1
                        mock_agent.stop.assert_called_once()


class TestMain:
    """Test cases for main function."""
    
    @pytest.mark.asyncio
    async def test_main_function_success(self):
        """Test main function returns success code."""
        with patch('src.main.JarvisApp') as mock_app_class:
            mock_app = Mock()
            mock_app.run = AsyncMock(return_value=0)
            mock_app_class.return_value = mock_app
            
            result = await main()
            assert result == 0
    
    @pytest.mark.asyncio
    async def test_main_function_failure(self):
        """Test main function returns failure code."""
        with patch('src.main.JarvisApp') as mock_app_class:
            mock_app = Mock()
            mock_app.run = AsyncMock(return_value=1)
            mock_app_class.return_value = mock_app
            
            result = await main()
            assert result == 1


class TestCliMain:
    """Test cases for cli_main function."""
    
    def test_cli_main_success(self):
        """Test cli_main with successful execution."""
        with patch('asyncio.run', return_value=0):
            with patch('sys.exit') as mock_exit:
                cli_main()
                mock_exit.assert_called_with(0)
    
    def test_cli_main_failure(self):
        """Test cli_main with failure."""
        with patch('asyncio.run', return_value=1):
            with patch('sys.exit') as mock_exit:
                cli_main()
                mock_exit.assert_called_with(1)
    
    def test_cli_main_keyboard_interrupt(self):
        """Test cli_main with keyboard interrupt."""
        with patch('asyncio.run', side_effect=KeyboardInterrupt()):
            with patch('sys.exit') as mock_exit:
                cli_main()
                mock_exit.assert_called_with(0)
    
    def test_cli_main_exception(self):
        """Test cli_main with general exception."""
        with patch('asyncio.run', side_effect=Exception("Test error")):
            with patch('sys.exit') as mock_exit:
                cli_main()
                mock_exit.assert_called_with(1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])