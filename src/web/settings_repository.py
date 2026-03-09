import sqlite3
from datetime import datetime, UTC
from pathlib import Path
from typing import Any


TOKEN_FIELDS = (
    "openai_api_key",
    "tavily_api_key",
    "confluence_page_token",
    "beacon_model_token",
)


def _resolve_sqlite_path(database_url: str) -> Path:
    prefixes = ("sqlite+aiosqlite:///", "sqlite:///")
    for prefix in prefixes:
        if database_url.startswith(prefix):
            raw_path = database_url[len(prefix) :]
            if raw_path.startswith("/"):
                return Path(raw_path)
            return Path.cwd() / raw_path
    raise ValueError("Only sqlite database URLs are supported in local mode")


class UserSettingsRepository:
    def __init__(self, database_url: str):
        self.database_path = _resolve_sqlite_path(database_url)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def init_table(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id TEXT PRIMARY KEY,
                    openai_api_key TEXT NULL,
                    tavily_api_key TEXT NULL,
                    confluence_page_token TEXT NULL,
                    beacon_model_token TEXT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def load_settings(self, user_id: str) -> dict[str, Any]:
        if not user_id:
            return {field: "" for field in TOKEN_FIELDS}

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT openai_api_key, tavily_api_key, confluence_page_token, beacon_model_token
                FROM user_settings
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()

        if row is None:
            return {field: "" for field in TOKEN_FIELDS}

        return {
            "openai_api_key": row["openai_api_key"] or "",
            "tavily_api_key": row["tavily_api_key"] or "",
            "confluence_page_token": row["confluence_page_token"] or "",
            "beacon_model_token": row["beacon_model_token"] or "",
        }

    def save_settings(self, user_id: str, settings: dict[str, Any]) -> None:
        if not user_id:
            return

        payload = {field: settings.get(field, "") for field in TOKEN_FIELDS}
        updated_at = datetime.now(UTC).isoformat()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO user_settings (
                    user_id, openai_api_key, tavily_api_key, confluence_page_token,
                    beacon_model_token, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    openai_api_key = excluded.openai_api_key,
                    tavily_api_key = excluded.tavily_api_key,
                    confluence_page_token = excluded.confluence_page_token,
                    beacon_model_token = excluded.beacon_model_token,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    payload["openai_api_key"],
                    payload["tavily_api_key"],
                    payload["confluence_page_token"],
                    payload["beacon_model_token"],
                    updated_at,
                ),
            )
            connection.commit()
