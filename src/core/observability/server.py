"""
Observability HTTP Server

Exposes Prometheus metrics and health check endpoints via HTTP.
Can run as a standalone server or be embedded in the web app.
"""

import asyncio
import logging
import time

from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest

logger = logging.getLogger(__name__)

# Default port for the metrics server
METRICS_PORT = 9100
HEALTH_PATH = "/health"
METRICS_PATH = "/metrics"


class ObservabilityServer:
    """
    Lightweight HTTP server exposing ``/metrics`` and ``/health`` endpoints.

    Uses asyncio's start_server for zero-dependency HTTP serving.
    Designed to be started alongside the Chainlit web app.

    Usage::

        server = ObservabilityServer(port=9100)
        await server.start()
        # ... app runs ...
        await server.stop()
    """

    def __init__(self, port: int = METRICS_PORT):
        self.port = port
        self._server: asyncio.AbstractServer | None = None
        self._start_time: float = 0.0
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start the metrics HTTP server."""
        if self._running:
            logger.warning("Observability server already running")
            return

        self._start_time = time.time()
        try:
            self._server = await asyncio.start_server(
                self._handle_request,
                host="0.0.0.0",
                port=self.port,
            )
            self._running = True
            logger.info(f"Observability server started on port {self.port}")
        except OSError as e:
            logger.warning(
                f"Could not start observability server on port {self.port}: {e}. "
                "Metrics will be collected but not exposed via HTTP."
            )

    async def stop(self) -> None:
        """Stop the metrics HTTP server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self._running = False
        logger.info("Observability server stopped")

    async def _handle_request(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single HTTP request."""
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if not request_line:
                writer.close()
                return

            path = request_line.decode("utf-8", errors="replace").split(" ")[1]

            if path == METRICS_PATH:
                await self._send_metrics(writer)
            elif path == HEALTH_PATH:
                await self._send_health(writer)
            else:
                await self._send_response(writer, 404, b"Not Found\n")
        except Exception as e:
            logger.debug(f"Metrics request error: {e}")
            try:
                await self._send_response(writer, 500, b"Internal Server Error\n")
            except Exception:
                pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _send_metrics(self, writer: asyncio.StreamWriter) -> None:
        """Send Prometheus metrics."""
        data = generate_latest(REGISTRY)
        await self._send_response(
            writer,
            200,
            data,
            content_type=CONTENT_TYPE_LATEST,
        )

    async def _send_health(self, writer: asyncio.StreamWriter) -> None:
        """Send health check response."""
        uptime = time.time() - self._start_time
        health_data = (
            f'{{"status": "ok", "uptime_seconds": {uptime:.1f}, "port": {self.port}}}\n'
        ).encode()
        await self._send_response(
            writer, 200, health_data, content_type="application/json"
        )

    async def _send_response(
        self,
        writer: asyncio.StreamWriter,
        status_code: int,
        body: bytes,
        content_type: str = "text/plain",
    ) -> None:
        """Send an HTTP response."""
        status_text = {200: "OK", 404: "Not Found", 500: "Internal Server Error"}
        status_line = (
            f"HTTP/1.1 {status_code} {status_text.get(status_code, 'Unknown')}\r\n"
        )
        headers = (
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Access-Control-Allow-Origin: *\r\n"
            f"\r\n"
        )
        writer.write(status_line.encode("utf-8"))
        writer.write(headers.encode("utf-8"))
        writer.write(body)
        await writer.drain()


# Global singleton for the server
_observability_server: ObservabilityServer | None = None


def get_observability_server(port: int = METRICS_PORT) -> ObservabilityServer:
    """Get or create the global observability server singleton."""
    global _observability_server
    if _observability_server is None:
        _observability_server = ObservabilityServer(port=port)
    return _observability_server
