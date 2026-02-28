"""Rolling statistics: frequency and time intervals per event type. No raw data stored."""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Optional

from roomcare.config import ROLLING_WINDOW_HOURS


class RollingStats:
    """Maintain rolling counts and inter-event intervals per event_type."""

    def __init__(self, window_hours: float = ROLLING_WINDOW_HOURS):
        self.window_sec = window_hours * 3600.0
        self._timestamps: dict[str, list[float]] = defaultdict(list)

    def add(self, event_type: str, ts: float | None = None) -> None:
        ts = ts or time.time()
        if event_type == "no_event":
            return
        self._timestamps[event_type].append(ts)
        self._trim(event_type)

    def _trim(self, event_type: str) -> None:
        now = time.time()
        self._timestamps[event_type] = [t for t in self._timestamps[event_type] if now - t <= self.window_sec]

    def trim_all(self) -> None:
        now = time.time()
        for k in list(self._timestamps):
            self._timestamps[k] = [t for t in self._timestamps[k] if now - t <= self.window_sec]

    def frequency(self, event_type: str) -> float:
        """Events per hour in the rolling window."""
        self._trim(event_type)
        n = len(self._timestamps[event_type])
        return n / (self.window_sec / 3600.0) if self.window_sec else 0.0

    def count(self, event_type: str) -> int:
        self._trim(event_type)
        return len(self._timestamps[event_type])

    def intervals(self, event_type: str) -> list[float]:
        """Sorted list of inter-event intervals in seconds (within window)."""
        self._trim(event_type)
        t = sorted(self._timestamps[event_type])
        return [t[i + 1] - t[i] for i in range(len(t) - 1)]

    def mean_interval(self, event_type: str) -> Optional[float]:
        inter = self.intervals(event_type)
        if not inter:
            return None
        return sum(inter) / len(inter)

    def std_interval(self, event_type: str) -> Optional[float]:
        inter = self.intervals(event_type)
        if not inter or len(inter) < 2:
            return None
        mean = sum(inter) / len(inter)
        var = sum((x - mean) ** 2 for x in inter) / (len(inter) - 1)
        return var ** 0.5
