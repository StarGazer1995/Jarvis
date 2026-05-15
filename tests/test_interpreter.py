"""
Tests for Python Interpreter Capability
"""

from unittest.mock import MagicMock, patch

import pytest

from src.capabilities.interpreter import PythonInterpreter

# We need to mock docker since we removed local execution support
# and we don't have a real docker daemon in the test environment.


@pytest.fixture
def mock_docker_client():
    with patch("src.capabilities.interpreter.docker") as mock_docker:
        # Mock DockerClient
        mock_client = MagicMock()
        mock_docker.DockerClient.return_value = mock_client
        mock_docker.from_env.return_value = mock_client

        # Mock container run
        mock_container = MagicMock()
        mock_container.decode.return_value = "Hello, World!"
        mock_client.containers.run.return_value = mock_container

        yield mock_client


def test_docker_execution_success(mock_docker_client):
    interpreter = PythonInterpreter(execution_mode="docker")

    # Configure mock return value specifically for this test
    mock_docker_client.containers.run.return_value.decode.return_value = "Hello, World!"

    result = interpreter.execute("print('Hello, World!')")
    assert result == "Hello, World!"

    # Verify docker run was called correctly
    mock_docker_client.containers.run.assert_called_once()
    args, kwargs = mock_docker_client.containers.run.call_args
    assert kwargs["command"] == ["python", "-c", "print('Hello, World!')"]


def test_docker_execution_error(mock_docker_client):
    interpreter = PythonInterpreter(execution_mode="docker")

    # Mock container error
    # We need to mock docker.errors.ContainerError
    # Since we mocked the whole docker module, we need to setup the exception
    Exception("Container Error")
    # Actually, in the code we catch docker.errors.ContainerError
    # We need to make sure the mocked docker module has this exception class

    # Re-patch docker to ensure we can simulate errors correctly
    with patch("src.capabilities.interpreter.docker") as mock_docker_module:
        mock_client = MagicMock()
        mock_docker_module.from_env.return_value = mock_client

        # Create a mock exception class structure
        class MockContainerError(Exception):
            def __init__(self, stderr):
                self.stderr = stderr

        mock_docker_module.errors.ContainerError = MockContainerError

        # Raise error when run is called
        mock_client.containers.run.side_effect = MockContainerError(
            stderr=b"SyntaxError: invalid syntax"
        )

        interpreter = PythonInterpreter(execution_mode="docker")
        result = interpreter.execute("syntax error code")

        assert "Execution Error (Container)" in result
        assert "SyntaxError" in result


def test_local_execution_disabled():
    # Test that any unknown mode (or explicit 'local' if we tried) fails
    with pytest.raises(ValueError, match="Unsupported execution mode"):
        PythonInterpreter(execution_mode="local")


def test_e2b_placeholder():
    interpreter = PythonInterpreter(execution_mode="e2b")
    result = interpreter.execute("print('test')")
    assert "E2B Execution Placeholder" in result
