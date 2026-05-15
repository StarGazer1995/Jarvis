"""
Stream Token Handler

Handles streaming tokens from LLM, parsing JSON structure to extract
thought and content fields for real-time display.
"""

import logging
import re
from collections.abc import AsyncGenerator, Callable


class StreamTokenHandler:
    """
    Handles streaming tokens from LLM, parsing specific tags and invoking callbacks.
    """

    def __init__(self, callbacks: dict[str, Callable] | None = None):
        """
        Initialize the handler.

        Args:
            callbacks: Dictionary of callback functions.
                      Supported: 'on_token', 'on_thought_start', 'on_thought_token', 'on_thought_end'
        """
        self.callbacks = callbacks or {}
        self.logger = logging.getLogger("llm.stream_handler")

    @staticmethod
    def _simple_unescape(text: str) -> str:
        return text.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")

    @staticmethod
    def _find_unescaped_quote(buffer: str) -> int:
        search_start = 0
        while True:
            quote_idx = buffer.find('"', search_start)
            if quote_idx == -1:
                return -1
            backslashes = 0
            i = quote_idx - 1
            while i >= 0 and buffer[i] == "\\":
                backslashes += 1
                i -= 1
            if backslashes % 2 == 0:
                return quote_idx
            search_start = quote_idx + 1

    @staticmethod
    def _safe_chunk_len(buffer: str, reserve: int = 10) -> int:
        safe_len = len(buffer) - reserve
        while safe_len > 0 and buffer[safe_len - 1] == "\\":
            safe_len -= 1
        return safe_len

    def _emit_callback(self, callback_name: str, content: str | None = None) -> None:
        if callback_name not in self.callbacks:
            return
        try:
            if content is None:
                self.callbacks[callback_name]()
            else:
                self.callbacks[callback_name](self._simple_unescape(content))
        except Exception as e:
            self.logger.error(f"Error in {callback_name}: {e}")

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

        async for chunk in stream_generator:
            full_response += chunk
            buffer += chunk

            # State 0: Look for "thought": "
            if state == 0:
                match = re.search(r'"thought"\s*:\s*"', buffer)
                if match:
                    self._emit_callback("on_thought_start")
                    buffer = buffer[match.end() :]
                    state = 1

            # State 1: In Thought string
            if state == 1:
                idx = self._find_unescaped_quote(buffer)
                if idx != -1:
                    content = buffer[:idx]
                    if content:
                        self._emit_callback("on_thought_token", content)
                    self._emit_callback("on_thought_end")
                    buffer = buffer[idx + 1 :]
                    state = 2
                else:
                    safe_len = self._safe_chunk_len(buffer)
                    if safe_len > 0:
                        content = buffer[:safe_len]
                        self._emit_callback("on_thought_token", content)
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
                idx = self._find_unescaped_quote(buffer)
                if idx != -1:
                    content = buffer[:idx]
                    if content:
                        self._emit_callback("on_token", content)
                    buffer = buffer[idx + 1 :]
                    state = 4
                else:
                    safe_len = self._safe_chunk_len(buffer)
                    if safe_len > 0:
                        content = buffer[:safe_len]
                        self._emit_callback("on_token", content)
                        buffer = buffer[safe_len:]

        return full_response
