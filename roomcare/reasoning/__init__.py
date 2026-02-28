"""Reasoning layer: rolling statistics, adaptive anomaly detection (no fixed thresholds)."""
from .stats import RollingStats
from .anomaly import AnomalyDetector

__all__ = ["RollingStats", "AnomalyDetector"]
