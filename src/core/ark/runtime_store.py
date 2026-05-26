"""
Persistent runtime state store for approval-based graph resumption.
"""

import json
import logging
from pathlib import Path
from typing import Any


class ApprovalRuntimeStore:
    """
    JSON-backed store for pending approvals and resumable graph state.

    The store keeps only the durable subset of runtime state needed to recover
    approval-gated executions after a process restart.
    """

    STORE_VERSION = 1

    def __init__(self, file_path: str | Path | None = None):
        """
        Initialize the runtime store.

        Args:
            file_path: Optional custom path for the JSON state file.
        """
        self.file_path = Path(file_path) if file_path else self._default_file_path()
        self.logger = logging.getLogger("ark.runtime_store")

    @classmethod
    def _default_file_path(cls) -> Path:
        """
        Resolve the default runtime store path under the project cache directory.

        Returns:
            Absolute path to the default store file.
        """
        project_root = Path(__file__).resolve().parents[3]
        return project_root / "cache" / "approval_runtime_state.json"

    def load(self) -> dict[str, Any]:
        """
        Load persisted runtime state from disk.

        Returns:
            Runtime state payload. Invalid or missing files return an empty state.
        """
        if not self.file_path.exists():
            return self._empty_state()

        try:
            with self.file_path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.warning(
                "Failed to load approval runtime state from %s: %s",
                self.file_path,
                exc,
            )
            return self._empty_state()

        if not isinstance(payload, dict):
            return self._empty_state()

        return {
            "version": int(payload.get("version", self.STORE_VERSION)),
            "pending_approvals": list(payload.get("pending_approvals", [])),
            "pending_continuations": list(payload.get("pending_continuations", [])),
        }

    def save(self, payload: dict[str, Any]) -> None:
        """
        Persist runtime state to disk using an atomic replace.

        Args:
            payload: Runtime state payload to persist.
        """
        approvals = list(payload.get("pending_approvals", []))
        continuations = list(payload.get("pending_continuations", []))
        if not approvals and not continuations:
            self.clear()
            return

        serializable_payload = {
            "version": self.STORE_VERSION,
            "pending_approvals": approvals,
            "pending_continuations": continuations,
        }
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self.file_path.with_suffix(f"{self.file_path.suffix}.tmp")
        with temp_file.open("w", encoding="utf-8") as handle:
            json.dump(serializable_payload, handle, indent=2, sort_keys=True)
        temp_file.replace(self.file_path)

    def clear(self) -> None:
        """
        Remove the persisted runtime state file if it exists.
        """
        self.file_path.unlink(missing_ok=True)

    @classmethod
    def _empty_state(cls) -> dict[str, Any]:
        """
        Build the canonical empty runtime state payload.

        Returns:
            Empty runtime state dictionary.
        """
        return {
            "version": cls.STORE_VERSION,
            "pending_approvals": [],
            "pending_continuations": [],
        }
