"""
Iterative Refinement Loop

This module implements a generic loop for refining agent outputs through
a generator-reviewer cycle.
"""

import os
import logging
import re
from datetime import datetime
from typing import Callable, Any, Optional, Union
import inspect

logger = logging.getLogger(__name__)

class RefinementLoop:
    """
    A generic loop that orchestrates a Generator -> Reviewer -> Refinement cycle.
    """
    
    def __init__(
        self, 
        generator: Callable[[str], Any], 
        reviewer: Callable[[str], str],
        max_retries: int = 2
    ):
        """
        Initialize the refinement loop.
        
        Args:
            generator: Function that takes a prompt and returns generated content.
                       Can be sync or async.
            reviewer: Function that takes content and returns feedback.
                      Must return "PASS" for success, or "RETRY: <feedback>" for failure.
                      Can be sync or async.
            max_retries: Maximum number of refinement attempts.
        """
        self.generator = generator
        self.reviewer = reviewer
        self.max_retries = max_retries

    async def _call_func(self, func: Callable, *args) -> Any:
        """Helper to call both sync and async functions."""
        if inspect.iscoroutinefunction(func):
            return await func(*args)
        return func(*args)

    def _clean_content(self, content: str) -> str:
        """
        Clean Chain-of-Thought (CoT) artifacts from the content.
        
        Removes:
        1. Content inside <think>...</think> tags.
        2. Content before </think> tag if opening tag is missing.
        3. Content before 'Final Answer:' marker.
        """
        if not content:
            return ""
            
        cleaned = content
        
        # 1. Remove <think> blocks (DeepSeek/Qwen style)
        # Handle complete blocks
        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
        # Handle isolated closing tags (often happens if opening tag is at start)
        if '</think>' in cleaned:
            parts = cleaned.split('</think>', 1)
            if len(parts) > 1:
                cleaned = parts[1].strip()
                
        # 2. Remove "Thought: ... Final Answer:" (ReAct style)
        # We look for the LAST occurrence of "Final Answer:" to be safe
        final_answer_match = re.search(r"(?:.*?Final Answer:\s*)(.*)", cleaned, re.DOTALL)
        if final_answer_match:
            cleaned = final_answer_match.group(1).strip()
            
        return cleaned.strip()

    async def run(self, initial_prompt: str) -> str:
        """
        Execute the refinement loop.
        
        Args:
            initial_prompt: The initial task description.
            
        Returns:
            The final refined content.
        """
        logger.info(f"Starting refinement loop with {self.max_retries} max retries")
        
        # 1. Initial Generation
        logger.info("[Refinement] Phase 1: Initial Generation")
        current_content = await self._call_func(self.generator, initial_prompt)
        
        # 2. Review Loop
        for attempt in range(self.max_retries):
            logger.info(f"[Refinement] Phase 2: Reviewing (Attempt {attempt + 1}/{self.max_retries})")
            
            feedback = await self._call_func(self.reviewer, current_content)
            logger.info(f"[Refinement] Reviewer Verdict: {feedback[:100]}...")
            
            if "PASS" in feedback:
                logger.info("✅ Content approved by reviewer")
                return current_content
                
            if "RETRY:" in feedback:
                logger.info("❌ Content rejected. Triggering refinement...")
                instructions = feedback.split("RETRY:", 1)[1].strip()
                
                # Construct refinement prompt
                refinement_prompt = (
                    f"<refinement_task>\n"
                    f"The previous output was incomplete or incorrect. Update the content based on the reviewer's feedback.\n"
                    f"</refinement_task>\n\n"
                    f"<feedback>\n{instructions}\n</feedback>\n\n"
                    f"<instructions>\n"
                    f"1. Address the issues raised in the feedback specifically.\n"
                    f"2. Combine new findings with valid previous information.\n"
                    f"3. Ensure the final output is complete and accurate.\n"
                    f"</instructions>"
                )
                
                logger.info(f"[Refinement] Phase 3: Refining...")
                current_content = await self._call_func(self.generator, refinement_prompt)
            else:
                logger.warning("⚠️ Reviewer gave ambiguous response. Stopping loop.")
                break
                
        logger.warning("Max retries reached. Returning current content.")
        return current_content

    def save_result(self, content: str, directory: str = "reports", prefix: str = "report", clean_cot: bool = True, task: Optional[str] = None) -> str:
        """
        Save the content to a file with a timestamp.
        
        Args:
            content: The string content to save.
            directory: Target directory (relative to CWD).
            prefix: Filename prefix.
            clean_cot: Whether to remove Chain-of-Thought artifacts before saving.
            task: Optional task description to include at the top of the file.
            
        Returns:
            Absolute path to the saved file.
        """
        # Create directory
        abs_dir = os.path.join(os.getcwd(), directory)
        os.makedirs(abs_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.md"
        file_path = os.path.join(abs_dir, filename)
        
        # Clean content if requested
        content_to_save = self._clean_content(content) if clean_cot else content
        
        # Add task description if provided
        if task:
            content_to_save = f"# Research Report\n\n## Task Description\n{task}\n\n## Findings\n{content_to_save}"
        
        # Write file
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content_to_save)
            logger.info(f"Result saved to: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to save result: {e}")
            return ""
