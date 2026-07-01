"""SQLite traceability store: request-id + history, NO raw PII.

Columns are exactly ``DOSSIER_FIELDS`` (from the shared contract), so there is no
``raw_text`` column by construction — anything persisted is post-anonymisation.
Soft-delete supports RGPD erasure while keeping an audit row.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .state import DOSSIER_FIELDS

_COLS = ", ".join(DOSSIER_FIELDS)
_PLACEHOLDERS = ", ".join("?" for _ in DOSSIER_FIELDS)
_INSERT_SQL = f"INSERT INTO dossier ({_COLS}) VALUES ({_PLACEHOLDERS})"


class Store:
    def __init__(self, db_path: str | Path) -> None:
        # check_same_thread=False lets the shared connection be used from graph worker threads;
        # a module-level lock then serialises every access so concurrent calls can't corrupt the
        # single sqlite3 connection ("bad parameter or other API misuse").
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.row_factory = sqlite3.Row
        # deleted is a flag → INTEGER so the value round-trips as int (not "1"/"0").
        cols = ", ".join(
            f"{c} INTEGER" if c == "deleted" else f"{c} TEXT" for c in DOSSIER_FIELDS
        )
        self._conn.execute(f"CREATE TABLE IF NOT EXISTS dossier ({cols})")
        self._conn.commit()

    def _columns(self) -> list[str]:
        cur = self._conn.execute("PRAGMA table_info(dossier)")
        return [r["name"] for r in cur.fetchall()]

    def record(self, case: dict) -> str:
        interaction_id = case.get("interaction_id") or str(uuid.uuid4())
        row = {k: case.get(k) for k in DOSSIER_FIELDS}
        row["interaction_id"] = interaction_id
        if not row.get("timestamp_utc"):
            row["timestamp_utc"] = datetime.now(UTC).isoformat()
        if row.get("deleted") is None:
            row["deleted"] = 0

        with self._lock:
            self._conn.execute(_INSERT_SQL, [row[c] for c in DOSSIER_FIELDS])
            self._conn.commit()
        return interaction_id

    def get(self, interaction_id: str) -> dict | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM dossier WHERE interaction_id = ?", (interaction_id,)
            )
            row = cur.fetchone()
        return dict(row) if row is not None else None

    def history(self, session_id: str) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM dossier WHERE session_id = ? AND deleted = 0 "
                "ORDER BY timestamp_utc",
                (session_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def soft_delete(self, interaction_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE dossier SET deleted = 1 WHERE interaction_id = ?", (interaction_id,)
            )
            self._conn.commit()

    def all_sessions(self) -> list[str]:
        with self._lock:
            cur = self._conn.execute("SELECT DISTINCT session_id FROM dossier")
            return [r["session_id"] for r in cur.fetchall()]
