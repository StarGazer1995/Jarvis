"""
Deep Research Agent Implementation
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from src.core.agent.react import ReActAgent
from src.capabilities.deep_research_tools import DeepResearchTools

class DeepResearchAgent(ReActAgent):
    """
    Agent implementing the Deep Research paradigm with ReAct loop and JSON-based tool calling.
    Inherits from ReActAgent to reuse JSON parsing and execution loop logic.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.logger = logging.getLogger('agent.deep_research')
        # Deep research typically requires more steps
        self.max_steps = self.config.get('max_steps', 30)
        self.tools = DeepResearchTools(self.llm_manager)
        
    def _get_system_prompt(self) -> str:
        """Override system prompt with Deep Research specific prompt."""
        current_date = datetime.now().strftime("%Y-%m-%d")
        return self.prompt_manager.render_template("deep_research_system", current_date=current_date)

    async def execute_tool(self, name: str, params: Any) -> Any:
        """
        Execute Deep Research tools.
        
        Args:
            name: Tool name
            params: Tool parameters (dict)
        """
        try:
            if name == "search":
                return await self.tools.search(params.get("query", []))
            elif name == "visit":
                return await self.tools.visit(params.get("url", []), params.get("goal", ""))
            elif name == "google_scholar":
                return await self.tools.google_scholar(params.get("query", []))
            elif name == "parse_file":
                return await self.tools.parse_file(params.get("files", []))
            elif name == "PythonInterpreter":
                 # Code is provided in arguments
                 if 'code' in params:
                     return await self.tools.python_interpreter(params['code'])
                 return "Error: PythonInterpreter must provide 'code' argument."
            else:
                return f"Error: Unknown tool '{name}'"
                
        except Exception as e:
            return f"Error executing tool '{name}': {str(e)}"

