import pytest
import subprocess
import time
import requests
import sys
import os
import importlib.util
from playwright.sync_api import Page, expect

# Skip if chainlit is not installed or if running in environment where UI testing is hard
if importlib.util.find_spec("chainlit") is None:
    pytest.skip("Chainlit not installed", allow_module_level=True)


@pytest.fixture(scope="module")
def chainlit_server():
    """Start Chainlit server for testing."""
    # Define port
    port = 8001
    url = f"http://localhost:{port}"

    # Command to run chainlit
    # We use -h to run in headless mode (no browser auto-open)
    cmd = [
        sys.executable,
        "-m",
        "chainlit",
        "run",
        "src/web/app.py",
        "--port",
        str(port),
        "--headless",
    ]

    # Start process
    print(f"Starting Chainlit server on {url}...")
    process = subprocess.Popen(
        cmd,
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "CHAINLIT_PORT": str(port)},
    )

    # Wait for server to start
    max_retries = 20
    server_started = False

    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                server_started = True
                break
        except requests.exceptions.RequestException:
            pass

        # Check if process is still running
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            print(f"Chainlit server exited prematurely.")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            break

        time.sleep(1)

    if not server_started:
        if process.poll() is None:
            process.terminate()
            process.wait()
        pytest.fail("Chainlit server failed to start within timeout")

    yield url

    # Cleanup
    print("Stopping Chainlit server...")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


@pytest.mark.e2e
def test_app_title(page: Page, chainlit_server):
    """Test that the application loads and has correct title."""
    page.goto(chainlit_server)

    # Check title - Chainlit usually sets title based on config or default
    # Adjust expectation based on actual title
    import re

    expect(page).to_have_title(re.compile(r"Jarvis|Chainlit|Assistant"))


@pytest.mark.e2e
def test_chat_interface_visible(page: Page, chainlit_server):
    """Test that the chat interface elements are visible."""
    page.goto(chainlit_server)

    # Wait for chat input to be visible
    # Chainlit input usually has id="chat-input" or similar, or we can look for textarea
    chat_input = page.locator("textarea")
    expect(chat_input).to_be_visible(timeout=10000)

    # Check for welcome message if configured
    # This depends on app.py logic
