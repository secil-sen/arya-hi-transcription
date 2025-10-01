"""
Configuration management for transcript monitoring.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MonitoringConfig:
    """Configuration for transcript monitoring."""

    project_id: Optional[str] = None
    location: str = "us-central1"
    experiment_name: str = "transcript_monitoring"

    monitoring_enabled: bool = True
    send_to_vertex: bool = True
    offline_mode: bool = False

    credentials_path: Optional[str] = None

    batch_size: int = 10
    batch_timeout_seconds: int = 30
    max_retries: int = 3

    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60

    track_audio_metadata: bool = True
    track_performance_metrics: bool = True
    track_quality_metrics: bool = True
    track_business_metrics: bool = True

    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "MonitoringConfig":
        """Create configuration from environment variables."""
        return cls(
            project_id=os.getenv("VERTEX_AI_PROJECT_ID"),
            location=os.getenv("VERTEX_AI_LOCATION", "us-central1"),
            experiment_name=os.getenv("VERTEX_AI_EXPERIMENT_NAME", "transcript_monitoring"),

            monitoring_enabled=os.getenv("MONITORING_ENABLED", "true").lower() == "true",
            send_to_vertex=os.getenv("SEND_TO_VERTEX", "true").lower() == "true",
            offline_mode=os.getenv("MONITORING_OFFLINE_MODE", "false").lower() == "true",

            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),

            batch_size=int(os.getenv("MONITORING_BATCH_SIZE", "10")),
            batch_timeout_seconds=int(os.getenv("MONITORING_BATCH_TIMEOUT", "30")),
            max_retries=int(os.getenv("MONITORING_MAX_RETRIES", "3")),

            circuit_breaker_failure_threshold=int(os.getenv("MONITORING_CB_FAILURE_THRESHOLD", "5")),
            circuit_breaker_recovery_timeout=int(os.getenv("MONITORING_CB_RECOVERY_TIMEOUT", "60")),

            track_audio_metadata=os.getenv("TRACK_AUDIO_METADATA", "true").lower() == "true",
            track_performance_metrics=os.getenv("TRACK_PERFORMANCE_METRICS", "true").lower() == "true",
            track_quality_metrics=os.getenv("TRACK_QUALITY_METRICS", "true").lower() == "true",
            track_business_metrics=os.getenv("TRACK_BUSINESS_METRICS", "true").lower() == "true",

            log_level=os.getenv("MONITORING_LOG_LEVEL", "INFO"),
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MonitoringConfig":
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})

    def validate(self) -> bool:
        """Validate configuration."""
        if self.send_to_vertex and not self.project_id:
            raise ValueError("project_id is required when send_to_vertex is True")

        if self.credentials_path and not Path(self.credentials_path).exists():
            raise ValueError(f"Credentials file not found: {self.credentials_path}")

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "project_id": self.project_id,
            "location": self.location,
            "experiment_name": self.experiment_name,
            "monitoring_enabled": self.monitoring_enabled,
            "send_to_vertex": self.send_to_vertex,
            "offline_mode": self.offline_mode,
            "credentials_path": self.credentials_path,
            "batch_size": self.batch_size,
            "batch_timeout_seconds": self.batch_timeout_seconds,
            "max_retries": self.max_retries,
            "circuit_breaker_failure_threshold": self.circuit_breaker_failure_threshold,
            "circuit_breaker_recovery_timeout": self.circuit_breaker_recovery_timeout,
            "track_audio_metadata": self.track_audio_metadata,
            "track_performance_metrics": self.track_performance_metrics,
            "track_quality_metrics": self.track_quality_metrics,
            "track_business_metrics": self.track_business_metrics,
            "log_level": self.log_level
        }