import os
import unittest
from unittest.mock import patch, MagicMock
from src.web.data_layer import get_data_layer


class TestDataLayer(unittest.TestCase):
    @patch("os.environ.get")
    def test_get_data_layer_success(self, mock_env):
        # Mock successful environment variable retrieval
        mock_env.return_value = "sqlite:///:memory:"

        # Mock SQLAlchemyDataLayer to avoid actual DB connection
        with patch("src.web.data_layer.SQLAlchemyDataLayer") as mock_dl:
            mock_instance = MagicMock()
            mock_dl.return_value = mock_instance

            dl = get_data_layer()

            # Verify initialization with correct parameters
            mock_dl.assert_called_once_with(
                conninfo="sqlite:///:memory:", show_logger=True
            )
            self.assertEqual(dl, mock_instance)

    @patch("os.environ.get")
    def test_get_data_layer_no_url(self, mock_env):
        # Mock missing environment variable
        mock_env.return_value = None

        dl = get_data_layer()

        self.assertIsNone(dl)

    @patch("os.environ.get")
    def test_get_data_layer_exception(self, mock_env):
        # Mock environment variable retrieval
        mock_env.return_value = "sqlite:///:memory:"

        # Mock SQLAlchemyDataLayer to raise an exception
        with patch("src.web.data_layer.SQLAlchemyDataLayer") as mock_dl:
            mock_dl.side_effect = Exception("Connection failed")

            dl = get_data_layer()

            self.assertIsNone(dl)


if __name__ == "__main__":
    unittest.main()
