"""
Pydantic models for metrics and monitoring data structures.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field


class TranscriptStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


class AudioMetadata(BaseModel):
    duration_seconds: Optional[float] = None
    file_size_bytes: Optional[int] = None
    format: Optional[str] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None


class PerformanceMetrics(BaseModel):
    request_start_time: datetime
    request_end_time: datetime
    total_duration_ms: float
    audio_processing_time_ms: Optional[float] = None
    model_inference_time_ms: Optional[float] = None
    pipeline_duration_ms: Optional[float] = None


class QualityMetrics(BaseModel):
    transcript_length: Optional[int] = None
    token_count: Optional[int] = None
    confidence_scores: Optional[List[float]] = None
    average_confidence: Optional[float] = None
    word_count: Optional[int] = None


class BusinessMetrics(BaseModel):
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    api_call_count: int = 1
    cost_estimate: Optional[float] = None


class ErrorInfo(BaseModel):
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    stack_trace: Optional[str] = None


class TranscriptMetrics(BaseModel):
    request_id: str = Field(default_factory=lambda: f"req_{datetime.now().timestamp()}")
    timestamp: datetime = Field(default_factory=datetime.now)
    status: TranscriptStatus

    audio_metadata: AudioMetadata
    performance_metrics: PerformanceMetrics
    quality_metrics: QualityMetrics
    business_metrics: BusinessMetrics

    error_info: Optional[ErrorInfo] = None
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MonitoringBatch(BaseModel):
    batch_id: str
    metrics: List[TranscriptMetrics]
    batch_timestamp: datetime = Field(default_factory=datetime.now)
    batch_size: int = Field(default=0)

    def __post_init__(self):
        self.batch_size = len(self.metrics)


class VertexAIEvaluationRequest(BaseModel):
    project_id: str
    location: str
    experiment_name: str
    metrics: TranscriptMetrics

    def to_vertex_format(self) -> Dict[str, Any]:
        return {
            "displayName": f"transcript_eval_{self.metrics.request_id}",
            "metrics": {
                "performance": {
                    "total_duration_ms": self.metrics.performance_metrics.total_duration_ms,
                    "audio_processing_time_ms": self.metrics.performance_metrics.audio_processing_time_ms,
                    "model_inference_time_ms": self.metrics.performance_metrics.model_inference_time_ms,
                },
                "quality": {
                    "transcript_length": self.metrics.quality_metrics.transcript_length,
                    "average_confidence": self.metrics.quality_metrics.average_confidence,
                    "word_count": self.metrics.quality_metrics.word_count,
                },
                "business": {
                    "user_id": self.metrics.business_metrics.user_id,
                    "api_call_count": self.metrics.business_metrics.api_call_count,
                }
            },
            "labels": {
                "status": self.metrics.status.value,
                "user_id": self.metrics.business_metrics.user_id or "unknown",
                "request_id": self.metrics.request_id
            }
        }