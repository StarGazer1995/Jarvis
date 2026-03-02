import os
import logging
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

# Configure logger
logger = logging.getLogger(__name__)

def get_data_layer():
    """
    Initializes and returns the SQLAlchemyDataLayer.
    
    This function:
    1. Loads DATABASE_URL from environment.
    2. Initializes SQLAlchemyDataLayer with DATABASE_URL.
    3. Handles database creation if needed (implicitly via SQLAlchemyDataLayer).
    """
    database_url = os.environ.get("LITE_DB_URL")
    
    if not database_url:
        logger.warning("LITE_DB_URL not found in environment variables. Data persistence will be disabled.")
        return None
    
    logger.info(f"Initializing SQLAlchemyDataLayer with URL: {database_url}")

    try:
        # Initialize SQLAlchemyDataLayer
        # We enable show_logger for better visibility during development
        data_layer = SQLAlchemyDataLayer(conninfo=database_url, show_logger=True)
        
        logger.info("SQLAlchemyDataLayer initialized successfully.")
        return data_layer
    except Exception as e:
        logger.error(f"Failed to initialize SQLAlchemyDataLayer: {e}")
        return None
