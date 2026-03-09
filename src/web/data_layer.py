import os
import logging
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

# Configure logger
logger = logging.getLogger(__name__)


_data_layer_instance = None


def get_database_url() -> str | None:
    return os.environ.get("LITE_DB_URL") or os.environ.get("DATABASE_URL")


def get_data_layer():
    """
    Initializes and returns the SQLAlchemyDataLayer (Singleton).

    This function:
    1. Loads LITE_DB_URL from environment.
    2. Initializes SQLAlchemyDataLayer with LITE_DB_URL if not already initialized.
    3. Handles database creation if needed (implicitly via SQLAlchemyDataLayer).
    """
    global _data_layer_instance

    if _data_layer_instance is not None:
        return _data_layer_instance

    database_url = get_database_url()

    if not database_url:
        logger.warning(
            "Database URL not found in environment variables. Data persistence will be disabled."
        )
        return None

    logger.info(f"Initializing SQLAlchemyDataLayer with URL: {database_url}")

    try:
        # Initialize SQLAlchemyDataLayer
        # We enable show_logger for better visibility during development
        _data_layer_instance = SQLAlchemyDataLayer(
            conninfo=database_url, show_logger=True
        )

        logger.info("SQLAlchemyDataLayer initialized successfully.")
        return _data_layer_instance
    except Exception as e:
        logger.error(f"Failed to initialize SQLAlchemyDataLayer: {e}")
        return None
