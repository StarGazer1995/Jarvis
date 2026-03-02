"""
Stream Token Handler

Handles streaming tokens from LLM, parsing JSON structure to extract
thought and content fields for real-time display.
"""

import logging
import re
from typing import Dict, Callable, AsyncGenerator, Optional


class StreamTokenHandler:
    """
    Handles streaming tokens from LLM, parsing specific tags and invoking callbacks.
    """

    def __init__(self, callbacks: Optional[Dict[str, Callable]] = None):
        """
        Initialize the handler.

        Args:
            callbacks: Dictionary of callback functions.
                      Supported: 'on_token', 'on_thought_start', 'on_thought_token', 'on_thought_end'
        """
        self.callbacks = callbacks or {}
        self.logger = logging.getLogger("llm.stream_handler")

    async def process_stream(self, stream_generator: AsyncGenerator[str, None]) -> str:
        """
        Process the stream generator, accumulating the full response.

        Since we enforce JSON mode via the API, we parse the stream incrementally
        to emit 'thought' and 'content' tokens for real-time UI updates.

        Args:
            stream_generator: Async generator yielding response chunks.

        Returns:
            The full accumulated JSON response string.
        """
        full_response = ""
        buffer = ""

        # States for simple JSON stream parsing
        # 0: Init / Looking for "thought": "
        # 1: In Thought string
        # 2: Post Thought / Looking for "content": "
        # 3: In Content string (Answer only)
        # 4: Done / Object mode
        state = 0

        def simple_unescape(text: str) -> str:
            """Simple unescape for display purposes."""
            return text.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")

        async for chunk in stream_generator:
            full_response += chunk
            buffer += chunk

            # State 0: Look for "thought": "
            if state == 0:
                match = re.search(r'"thought"\s*:\s*"', buffer)
                if match:
                    if "on_thought_start" in self.callbacks:
                        try:
                            self.callbacks["on_thought_start"]()
                        except Exception as e:
                            self.logger.error(f"Error in on_thought_start: {e}")

                    buffer = buffer[match.end() :]
                    state = 1

            # State 1: In Thought string
            if state == 1:
                # Find unescaped quote
                idx = -1
                search_start = 0
                while True:
                    quote_idx = buffer.find('"', search_start)
                    if quote_idx == -1:
                        break

                    # Check backslashes
                    backslashes = 0
                    i = quote_idx - 1
                    while i >= 0 and buffer[i] == "\\":
                        backslashes += 1
                        i -= 1

                    if backslashes % 2 == 0:
                        idx = quote_idx
                        break
                    else:
                        search_start = quote_idx + 1

                if idx != -1:
                    # Found end of thought
                    content = buffer[:idx]
                    if content and "on_thought_token" in self.callbacks:
                        try:
                            self.callbacks["on_thought_token"](simple_unescape(content))
                        except Exception as e:
                            self.logger.error(f"Error in on_thought_token: {e}")

                    if "on_thought_end" in self.callbacks:
                        try:
                            self.callbacks["on_thought_end"]()
                        except Exception as e:
                            self.logger.error(f"Error in on_thought_end: {e}")

                    buffer = buffer[idx + 1 :]
                    state = 2
                else:
                    # Emit safe part
                    safe_len = len(buffer) - 10  # Keep a buffer for escape sequences

                    # Ensure we don't split an escape sequence
                    while safe_len > 0 and buffer[safe_len - 1] == "\\":
                        safe_len -= 1

                    if safe_len > 0:
                        content = buffer[:safe_len]
                        if "on_thought_token" in self.callbacks:
                            try:
                                self.callbacks["on_thought_token"](
                                    simple_unescape(content)
                                )
                            except Exception as e:
                                self.logger.error(f"Error in on_thought_token: {e}")
                        buffer = buffer[safe_len:]

            # State 2: Look for "content": "
            if state == 2:
                # We only stream content if it's a string (answer).
                # If it's a tool_call (object), we don't stream it to on_token.
                # Heuristic: check if "content": " matches.
                match = re.search(r'"content"\s*:\s*"', buffer)
                if match:
                    buffer = buffer[match.end() :]
                    state = 3
                elif re.search(r'"content"\s*:\s*\{', buffer):
                    # It's an object (tool call), skip streaming
                    state = 4

            # State 3: In Content string (Answer)
            if state == 3:
                # Find unescaped quote
                idx = -1
                search_start = 0
                while True:
                    quote_idx = buffer.find('"', search_start)
                    if quote_idx == -1:
                        break

                    backslashes = 0
                    i = quote_idx - 1
                    while i >= 0 and buffer[i] == "\\":
                        backslashes += 1
                        i -= 1

                    if backslashes % 2 == 0:
                        idx = quote_idx
                        break
                    else:
                        search_start = quote_idx + 1

                if idx != -1:
                    content = buffer[:idx]
                    if content and "on_token" in self.callbacks:
                        try:
                            self.callbacks["on_token"](simple_unescape(content))
                        except Exception as e:
                            self.logger.error(f"Error in on_token: {e}")

                    buffer = buffer[idx + 1 :]
                    state = 4
                else:
                    safe_len = len(buffer) - 10
                    while safe_len > 0 and buffer[safe_len - 1] == "\\":
                        safe_len -= 1

                    if safe_len > 0:
                        content = buffer[:safe_len]
                        if "on_token" in self.callbacks:
                            try:
                                self.callbacks["on_token"](simple_unescape(content))
                            except Exception as e:
                                self.logger.error(f"Error in on_token: {e}")
                        buffer = buffer[safe_len:]

        return full_response
