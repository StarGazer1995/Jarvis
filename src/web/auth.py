import os
from typing import Optional
import chainlit as cl

# Cache admin password at module level
_ADMIN_PASSWORD = os.environ.get("CHAINLIT_ADMIN_PASSWORD")


def auth_callback(username: str, password: str) -> Optional[cl.User]:
    """
    Chainlit authentication callback.
    Validates the username and password against environment variables.

    Args:
        username (str): The username provided by the user.
        password (str): The password provided by the user.

    Returns:
        Optional[cl.User]: Returns a User object if authentication is successful, None otherwise.
    """
    if username == "admin" and password == _ADMIN_PASSWORD:
        return cl.User(identifier="admin")

    return None
