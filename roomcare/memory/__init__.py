"""Memory layer: SQLite-backed care_events and alerts. No raw video/audio stored."""
from .db import MemoryStore, init_db

__all__ = ["MemoryStore", "init_db"]
