"""
Tests for the main module.

This module contains comprehensive tests for the main entry point,
including command-line interface, interactive mode, demo mode,
and argument parsing functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import argparse
import logging
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.main import JarvisApp, main


class TestJarvisApp:
    """Test cases for JarvisApp class."""
    
    def test_jarvis_app_init(self):
        """Test JarvisApp initialization."""
        app = JarvisApp()
        assert app.agent is None
        assert not app.running
        assert app.logger is not None
    
    def test_jarvis_app_parse_arguments_defaults(self):
        """Test JarvisApp parse_arguments with default values."""
        app = JarvisApp()
        
        with patch('sys.argv', ['main.py']):
            args = app.parse_arguments()
            assert args.log_level == 'INFO'
            assert not args.interactive
            assert not args.demo
    
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







class TestMain:
    """Test cases for main function."""
    
    def test_main_function_exists(self):
        """Test that main function exists and is callable."""
        # Just test that main function exists and is callable
        assert callable(main)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])