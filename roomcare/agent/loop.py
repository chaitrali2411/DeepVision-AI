"""Agent loop: Observe → Interpret → Update Memory → Evaluate Pattern → Act. No raw media stored."""
from __future__ import annotations

import time
from typing import Optional

from roomcare.fusion import FusionLayer, FusionResult
from roomcare.memory import MemoryStore
from roomcare.perception import AudioTranscriber, IntentExtractor, VisionClassifier
from roomcare.reasoning import AnomalyDetector, RollingStats


class RoomCareAgent:
    """Observe → Interpret → Update Memory → Evaluate Pattern → Act."""

    def __init__(
        self,
        memory: Optional[MemoryStore] = None,
        vision: Optional[VisionClassifier] = None,
        audio: Optional[AudioTranscriber] = None,
        intent: Optional[IntentExtractor] = None,
        fusion: Optional[FusionLayer] = None,
        rolling_stats: Optional[RollingStats] = None,
        anomaly: Optional[AnomalyDetector] = None,
    ):
        self.memory = memory or MemoryStore()
        self.vision = vision or VisionClassifier()
        self.audio = audio or AudioTranscriber()
        self.intent = intent or IntentExtractor()
        self.fusion = fusion or FusionLayer()
        self.rolling_stats = rolling_stats or RollingStats()
        self.anomaly = anomaly or AnomalyDetector(rolling_stats=self.rolling_stats)

    def observe_frame(self, image) -> str:
        """Interpret one frame; return event_type. Does not persist raw image."""
        event_type = self.vision.classify(image)
        return event_type

    def observe_audio_chunk(self, audio_chunk) -> Optional[str]:
        """Transcribe and extract intent; return event_type or None. No raw audio stored."""
        text = self.audio.transcribe(audio_chunk)
        return self.intent.extract(text)

    def interpret_and_update(self, event_type: str, source: str = "vision", ts: float | None = None) -> None:
        """Fuse, update memory and rolling stats, evaluate anomalies, act (store alerts)."""
        ts = ts or time.time()
        if event_type == "no_event" and source == "vision":
            return
        if source == "vision":
            fused = self.fusion.fuse_vision(event_type, ts)
        else:
            fused = self.fusion.fuse_audio_intent(event_type, ts)
            if fused is None:
                return
        self.memory.insert_event(
            event_type=fused.event_type,
            source=fused.source,
            verification_level=fused.verification_level,
            raw_metadata=fused.raw_metadata,
            ts=fused.ts,
        )
        self.rolling_stats.add(fused.event_type, fused.ts)
        reports = self.anomaly.check_and_report(fused.event_type, fused.ts)
        for alert_type, summary, severity in reports:
            self.memory.insert_alert(
                alert_type=alert_type,
                summary=summary,
                severity=severity,
                ts=fused.ts,
            )

    def step_frame(self, image):
        """Full step for one frame: observe → interpret → update → evaluate → act."""
        event_type = self.observe_frame(image)
        self.interpret_and_update(event_type, source="vision")
        return event_type
