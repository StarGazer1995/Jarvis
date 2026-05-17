from unittest.mock import patch

from src.web.auth import auth_callback


def test_auth_success():
    with patch("src.web.auth._ADMIN_PASSWORD", "secret"):
        user = auth_callback("admin", "secret")
        assert user is not None
        assert user.identifier == "admin"


def test_auth_failure_wrong_password():
    with patch("src.web.auth._ADMIN_PASSWORD", "secret"):
        user = auth_callback("admin", "wrong")
        assert user is None


def test_auth_failure_wrong_user():
    with patch("src.web.auth._ADMIN_PASSWORD", "secret"):
        user = auth_callback("user", "secret")
        assert user is None


def test_auth_failure_no_password_set():
    with patch("src.web.auth._ADMIN_PASSWORD", None):
        user = auth_callback("admin", "password")
        assert user is None
