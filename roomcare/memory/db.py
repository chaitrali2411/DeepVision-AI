"""SQLite memory: care_events and alerts. Only structured metadata persisted."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from roomcare.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS care_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    event_type TEXT NOT NULL,
    source TEXT NOT NULL,
    verification_level REAL,
    raw_metadata TEXT,
    created_at REAL DEFAULT (strftime('%s','now'))
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    alert_type TEXT NOT NULL,
    summary TEXT,
    severity REAL,
    related_event_ids TEXT,
    created_at REAL DEFAULT (strftime('%s','now'))
);

CREATE INDEX IF NOT EXISTS idx_care_events_ts ON care_events(ts);
CREATE INDEX IF NOT EXISTS idx_care_events_type ON care_events(event_type);
CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(ts);
"""


def init_db(db_path: Path | str | None = None) -> None:
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


class MemoryStore:
    """On-device memory: insert/query care_events and alerts. No raw media."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path or DB_PATH)
        init_db(self.db_path)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def insert_event(
        self,
        event_type: str,
        source: str,
        verification_level: float | None = None,
        raw_metadata: str | None = None,
        ts: float | None = None,
    ) -> int:
        ts = ts or time.time()
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO care_events (ts, event_type, source, verification_level, raw_metadata) VALUES (?,?,?,?,?)",
                (ts, event_type, source, verification_level, raw_metadata),
            )
            c.commit()
            return cur.lastrowid or 0

    def insert_alert(
        self,
        alert_type: str,
        summary: str,
        severity: float = 1.0,
        related_event_ids: list[int] | None = None,
        ts: float | None = None,
    ) -> int:
        ts = ts or time.time()
        ids_str = ",".join(map(str, related_event_ids or []))
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO alerts (ts, alert_type, summary, severity, related_event_ids) VALUES (?,?,?,?,?)",
                (ts, alert_type, summary, severity, ids_str),
            )
            c.commit()
            return cur.lastrowid or 0

    def get_events_since(self, since_ts: float) -> list[dict[str, Any]]:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            rows = c.execute(
                "SELECT id, ts, event_type, source, verification_level, raw_metadata FROM care_events WHERE ts >= ? ORDER BY ts",
                (since_ts,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_events_by_type(self, event_type: str, since_ts: float | None = None) -> list[dict[str, Any]]:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            if since_ts is not None:
                rows = c.execute(
                    "SELECT id, ts, event_type, source, verification_level FROM care_events WHERE event_type = ? AND ts >= ? ORDER BY ts",
                    (event_type, since_ts),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT id, ts, event_type, source, verification_level FROM care_events WHERE event_type = ? ORDER BY ts",
                    (event_type,),
                ).fetchall()
            return [dict(r) for r in rows]

    def get_alerts_since(self, since_ts: float) -> list[dict[str, Any]]:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            rows = c.execute(
                "SELECT id, ts, alert_type, summary, severity, related_event_ids FROM alerts WHERE ts >= ? ORDER BY ts",
                (since_ts,),
            ).fetchall()
            return [dict(r) for r in rows]
