"""
Base Agent Class
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from ..llm.client import LLMManager
from ..llm.config import load_llm_config
from .types import AgentState


class BaseAgent(ABC):
    """
    Abstract base class for AI agents.

    Provides common functionality for initialization, state management,
    and LLM integration.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the base agent.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.state = AgentState.INITIALIZING
        self.logger = logging.getLogger("agent.core")

        # LLM Setup
        llm_config_dict = self.config.get("llm", {})
        self.llm_config = (
            load_llm_config(llm_config_dict) if llm_config_dict else load_llm_config()
        )
        self.llm_manager = LLMManager(self.llm_config)
        self.llm_enabled = self.config.get("enable_llm", True)

        # Tools
        self.available_tools: dict[str, Any] = {}

    @abstractmethod
    async def process_input(
        self, user_input: str, callbacks: dict[str, Callable] | None = None, **kwargs
    ) -> str:
        """
        Process user input and return a response.

        Args:
            user_input: The user's message or query
            callbacks: Optional dictionary of callback functions
            **kwargs: Additional keyword arguments

        Returns:
            The agent's response
        """
        pass

    async def initialize(self) -> bool:
        """
        Initialize the agent.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.state = AgentState.INITIALIZING

            # 如果启用了LLM且配置不是Mock，则初始化LLM
            if self.llm_enabled:
                await self._initialize_llm()

            self.state = AgentState.READY
            return True
        except Exception as e:
            self.state = AgentState.ERROR
            self.logger.error(f"Agent initialization failed: {e}")
            return False

    async def _initialize_llm(self) -> None:
        """Initialize the LLM client."""
        success = await self.llm_manager.initialize_default_client()
        if not success:
            self.logger.error("LLM client initialization failed")
            self.llm_enabled = False

    async def shutdown(self) -> None:
        """Shutdown the agent and clean up resources."""
        self.state = AgentState.SHUTDOWN
        self.logger.info("Agent shutting down")
