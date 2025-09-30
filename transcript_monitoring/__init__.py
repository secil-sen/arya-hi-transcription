"""
Transcript Monitoring Package

A seamless integration package for monitoring transcript services with Google Cloud Vertex AI.
Provides decorator-based performance tracking, metrics collection, and real-time monitoring.
"""

from .decorators import (
    transcript_monitor,
    TranscriptMonitor,
    get_monitoring_stats,
    health_check,
    start_monitoring,
    stop_monitoring
)
from .config import MonitoringConfig
from .exceptions import MonitoringError, VertexAIError

__version__ = "1.0.0"
__all__ = [
    "transcript_monitor",
    "TranscriptMonitor",
    "get_monitoring_stats",
    "health_check",
    "start_monitoring",
    "stop_monitoring",
    "MonitoringConfig",
    "MonitoringError",
    "VertexAIError"
]