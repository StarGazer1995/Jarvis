import pytest
from unittest.mock import patch
from src.web.data_layer import get_data_layer
import src.web.data_layer


@pytest.fixture
def mock_sqlalchemy_layer():
    with patch("src.web.data_layer.SQLAlchemyDataLayer") as mock:
        yield mock


@pytest.fixture
def reset_singleton():
    # Reset the singleton before and after test
    src.web.data_layer._data_layer_instance = None
    yield
    src.web.data_layer._data_layer_instance = None


def test_get_data_layer_singleton(mock_sqlalchemy_layer, reset_singleton):
    with patch.dict("os.environ", {"LITE_DB_URL": "sqlite:///test.db"}):
        dl1 = get_data_layer()
        dl2 = get_data_layer()

        assert dl1 is not None
        assert dl1 is dl2
        mock_sqlalchemy_layer.assert_called_once()


def test_get_data_layer_no_env(reset_singleton):
    with patch.dict("os.environ", {}, clear=True):
        dl = get_data_layer()
        assert dl is None


def test_get_data_layer_with_database_url(mock_sqlalchemy_layer, reset_singleton):
    with patch.dict("os.environ", {"DATABASE_URL": "sqlite:///test.db"}, clear=True):
        dl = get_data_layer()
        assert dl is not None
        mock_sqlalchemy_layer.assert_called_once()


def test_get_data_layer_exception(mock_sqlalchemy_layer, reset_singleton):
    mock_sqlalchemy_layer.side_effect = Exception("DB Error")
    with patch.dict("os.environ", {"LITE_DB_URL": "sqlite:///test.db"}):
        dl = get_data_layer()
        assert dl is None
