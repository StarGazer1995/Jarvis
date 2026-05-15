"""
Python Interpreter Capability

This module provides a Python execution environment that runs remotely via Docker
or cloud sandbox services. Local execution is NOT supported for security reasons.
"""

import logging
from typing import Any

try:
    import docker
except ImportError:
    docker = None

logger = logging.getLogger(__name__)


class PythonInterpreter:
    """
    Executes Python code.
    Supports:
    1. 'docker': Runs in a Docker container (local or remote).
    2. 'e2b': Placeholder for E2B cloud sandbox integration.

    Local execution is explicitly disabled.
    """

    def __init__(
        self,
        safe_globals: dict[str, Any] | None = None,
        execution_mode: str = "docker",
        docker_config: dict[str, str] | None = None,
        sandbox_config: dict[str, str] | None = None,
    ):
        """
        Initialize the interpreter.

        Args:
            safe_globals: Ignored.
            execution_mode: 'docker' or 'e2b'. Defaults to 'docker'.
            docker_config: Dictionary containing Docker configuration.
                - base_url: Docker daemon URL
                - image: Docker image to use (default: 'python:3.10-slim')
            sandbox_config: Dictionary containing Cloud Sandbox configuration (e.g. API keys).
        """
        self.execution_mode = execution_mode
        self.docker_config = docker_config or {}
        self.sandbox_config = sandbox_config or {}
        self.docker_client = None

        if self.execution_mode == "docker":
            if not docker:
                logger.error(
                    "Docker SDK not found but execution_mode is 'docker'. Execution will fail."
                )
            else:
                try:
                    base_url = self.docker_config.get("base_url")
                    if base_url:
                        self.docker_client = docker.DockerClient(base_url=base_url)
                    else:
                        self.docker_client = docker.from_env()

                    # Verify connection
                    self.docker_client.ping()
                except Exception as e:
                    logger.error(
                        f"Failed to connect to Docker: {e}. Execution will fail."
                    )
                    self.docker_client = None

        elif self.execution_mode == "e2b":
            # Placeholder for E2B initialization
            if not self.sandbox_config.get("api_key"):
                logger.warning("E2B API key not provided.")
        else:
            raise ValueError(
                f"Unsupported execution mode '{self.execution_mode}'. Only 'docker' and 'e2b' are supported."
            )

    def execute(self, code: str) -> str:
        """
        Executes the provided Python code and returns the captured stdout/stderr.
        """
        if self.execution_mode == "docker":
            if not self.docker_client:
                return "Execution Error: Docker client is not initialized or failed to connect."
            return self._execute_docker(code)

        elif self.execution_mode == "e2b":
            return self._execute_e2b(code)

        else:
            return f"Execution Error: Unsupported execution mode '{self.execution_mode}' or local execution is disabled."

    def _execute_docker(self, code: str) -> str:
        """
        Executes code in a Docker container.
        """
        image = self.docker_config.get("image", "python:3.10-slim")
        try:
            container = self.docker_client.containers.run(
                image,
                command=["python", "-c", code],
                remove=True,
                stdout=True,
                stderr=True,
                mem_limit="512m",
                nano_cpus=500000000,  # 0.5 CPU
                network_disabled=False,
            )

            output = container.decode("utf-8")
            return output.strip() or "(No output)"

        except docker.errors.ContainerError as e:
            return f"Execution Error (Container):\n{e.stderr.decode('utf-8')}"
        except docker.errors.ImageNotFound:
            return f"Execution Error: Docker image '{image}' not found. Please pull it first."
        except Exception as e:
            return f"Docker Execution Error:\n{str(e)}"

    def _execute_e2b(self, code: str) -> str:
        """
        Placeholder for E2B execution.
        """
        # In a real implementation, this would use e2b-code-interpreter SDK
        return "E2B Execution Placeholder: Sandbox execution not yet implemented."
