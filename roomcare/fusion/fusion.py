"""Fusion: align voice + vision events and assign verification level (no raw media)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class FusionResult:
    event_type: str
    source: str  # "vision" | "audio" | "fusion"
    verification_level: float  # 0..1
    ts: float
    raw_metadata: Optional[str] = None


# Time window (seconds) to consider vision and audio as aligned
ALIGN_WINDOW_SEC = 30.0


class FusionLayer:
    """Align voice + vision; assign verification level. No raw audio/video stored."""

    def __init__(self, align_window_sec: float = ALIGN_WINDOW_SEC):
        self.align_window_sec = align_window_sec
        self._recent_vision: list[tuple[float, str]] = []
        self._recent_audio_intent: list[tuple[float, str]] = []

    def add_vision(self, event_type: str, ts: float | None = None) -> None:
        ts = ts or time.time()
        if event_type == "no_event":
            return
        self._recent_vision.append((ts, event_type))
        self._trim_recent()

    def add_audio_intent(self, event_type: str, ts: float | None = None) -> None:
        ts = ts or time.time()
        if not event_type:
            return
        self._recent_audio_intent.append((ts, event_type))
        self._trim_recent()

    def _trim_recent(self) -> None:
        now = time.time()
        self._recent_vision = [(t, e) for t, e in self._recent_vision if now - t < self.align_window_sec]
        self._recent_audio_intent = [(t, e) for t, e in self._recent_audio_intent if now - t < self.align_window_sec]

    def fuse_vision(self, event_type: str, ts: float | None = None) -> FusionResult:
        """Emit a fusion result for a vision event; check alignment with recent audio."""
        ts = ts or time.time()
        self.add_vision(event_type, ts)
        level = self._verification_level(event_type, ts, from_vision=True)
        return FusionResult(
            event_type=event_type,
            source="fusion" if level > 0.5 else "vision",
            verification_level=level,
            ts=ts,
            raw_metadata=None,
        )

    def fuse_audio_intent(self, event_type: str, ts: float | None = None) -> FusionResult | None:
        """Emit a fusion result for an audio-derived intent if not already explained by vision."""
        ts = ts or time.time()
        self.add_audio_intent(event_type, ts)
        level = self._verification_level(event_type, ts, from_vision=False)
        return FusionResult(
            event_type=event_type,
            source="fusion" if level > 0.5 else "audio",
            verification_level=level,
            ts=ts,
            raw_metadata=None,
        )

    def _verification_level(self, event_type: str, ts: float, from_vision: bool) -> float:
        """Higher when vision and audio agree within the time window."""
        now = time.time()
        other = self._recent_vision if not from_vision else self._recent_audio_intent
        for t, e in other:
            if abs(t - ts) <= self.align_window_sec and e == event_type:
                return 0.9
        return 0.4 if from_vision else 0.3
