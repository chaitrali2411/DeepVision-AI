"""Adaptive anomaly detection using statistical deviation. No fixed hardcoded thresholds."""
from __future__ import annotations

import time
from typing import Optional

from roomcare.config import ANOMALY_Z_SCORE_CAP, MIN_SAMPLES_FOR_STATS

from .stats import RollingStats


class AnomalyDetector:
    """Detect anomalies from rolling stats: frequency and interval deviations (z-score based)."""

    def __init__(
        self,
        rolling_stats: Optional[RollingStats] = None,
        min_samples: int = MIN_SAMPLES_FOR_STATS,
        z_cap: float = ANOMALY_Z_SCORE_CAP,
    ):
        self.stats = rolling_stats or RollingStats()
        self.min_samples = min_samples
        self.z_cap = z_cap

    def evaluate_frequency_anomaly(self, event_type: str, current_count_in_window: int) -> tuple[bool, float]:
        """True if frequency is anomalous; second value is severity (0..1)."""
        self.stats.trim_all()
        count = self.stats.count(event_type)
        if count < self.min_samples:
            return False, 0.0
        # Simple heuristic: expect roughly uniform rate; anomaly if count is very high
        expected_per_hour = count / (self.stats.window_sec / 3600.0) if self.stats.window_sec else 0
        freq = self.stats.frequency(event_type)
        if expected_per_hour <= 0:
            return False, 0.0
        # Z-like: (freq - expected) / scale; use sqrt(expected) as scale
        scale = max((expected_per_hour ** 0.5), 1e-6)
        z = (freq - expected_per_hour) / scale
        z = max(-self.z_cap, min(self.z_cap, z))
        severity = min(1.0, abs(z) / self.z_cap)
        return severity > 0.5, severity

    def evaluate_interval_anomaly(self, event_type: str, last_interval_sec: float | None) -> tuple[bool, float]:
        """True if last interval is anomalous vs rolling distribution; severity 0..1."""
        self.stats.trim_all()
        intervals = self.stats.intervals(event_type)
        if last_interval_sec is None or len(intervals) < self.min_samples:
            return False, 0.0
        mean = sum(intervals) / len(intervals)
        var = sum((x - mean) ** 2 for x in intervals) / max(len(intervals) - 1, 1)
        std = (var ** 0.5) or 1e-6
        z = (last_interval_sec - mean) / std
        z = max(-self.z_cap, min(self.z_cap, z))
        severity = min(1.0, abs(z) / self.z_cap)
        return severity > 0.5, severity

    def check_and_report(
        self,
        event_type: str,
        ts: float | None = None,
    ) -> list[tuple[str, str, float]]:
        """Returns list of (alert_type, summary, severity) for any detected anomalies."""
        ts = ts or time.time()
        self.stats.add(event_type, ts)
        self.stats.trim_all()
        reports = []
        count = self.stats.count(event_type)
        intervals = self.stats.intervals(event_type)
        last_interval = intervals[-1] if intervals else None
        is_freq_anomaly, freq_severity = self.evaluate_frequency_anomaly(event_type, count)
        if is_freq_anomaly and freq_severity > 0.5:
            reports.append(
                ("frequency_anomaly", f"Unusual frequency for {event_type}: {count} in window", freq_severity)
            )
        is_interval_anomaly, interval_severity = self.evaluate_interval_anomaly(event_type, last_interval)
        if is_interval_anomaly and interval_severity > 0.5:
            reports.append(
                (
                    "interval_anomaly",
                    f"Unusual interval for {event_type}: last_interval_sec={last_interval:.0f}",
                    interval_severity,
                )
            )
        return reports
