import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Mock chainlit before importing src.web.auth
mock_cl = MagicMock()
sys.modules['chainlit'] = mock_cl

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from src.web.auth import auth_callback
except ImportError as e:
    print(f"Error importing auth_callback: {e}")
    sys.exit(1)

class TestAuthCallback(unittest.TestCase):
    def test_auth_success(self):
        """Test successful authentication"""
        with patch.dict(os.environ, {'CHAINLIT_ADMIN_PASSWORD': 'secret_password'}):
            mock_user = MagicMock()
            mock_cl.User.return_value = mock_user
            
            result = auth_callback("admin", "secret_password")
            
            self.assertEqual(result, mock_user)
            mock_cl.User.assert_called_with(identifier="admin")

    def test_auth_wrong_password(self):
        """Test authentication with wrong password"""
        with patch.dict(os.environ, {'CHAINLIT_ADMIN_PASSWORD': 'secret_password'}):
            result = auth_callback("admin", "wrong_password")
            self.assertIsNone(result)

    def test_auth_wrong_username(self):
        """Test authentication with wrong username"""
        with patch.dict(os.environ, {'CHAINLIT_ADMIN_PASSWORD': 'secret_password'}):
            result = auth_callback("user", "secret_password")
            self.assertIsNone(result)

    def test_auth_no_env_var(self):
        """Test authentication when environment variable is not set"""
        # Ensure env var is not set
        with patch.dict(os.environ, {}, clear=True):
            result = auth_callback("admin", "password")
            self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
