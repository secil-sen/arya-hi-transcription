"""
Vertex AI client wrapper for monitoring and evaluation.
"""

import asyncio
import logging
import os
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import json

try:
    from google.cloud import aiplatform
    from google.cloud import monitoring_v3
    from google.cloud import logging as gcp_logging
    from google.auth import default
    try:
        import vertexai
        from vertexai.preview.experiments import log_metrics, log_params, start_run, end_run, init
        VERTEX_EXPERIMENTS_AVAILABLE = True
    except ImportError:
        VERTEX_EXPERIMENTS_AVAILABLE = False
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    VERTEX_EXPERIMENTS_AVAILABLE = False

from .models import TranscriptMetrics, VertexAIEvaluationRequest, MonitoringBatch
from .config import MonitoringConfig
from .exceptions import VertexAIError, ConfigurationError


logger = logging.getLogger(__name__)


def _serialize_for_logging(data: Any) -> Any:
    """Convert datetime objects and other non-JSON-serializable objects to appropriate formats."""
    if data is None:
        return None
    elif isinstance(data, datetime):
        return data.isoformat()
    elif isinstance(data, dict):
        return {k: _serialize_for_logging(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [_serialize_for_logging(item) for item in data]
    elif hasattr(data, 'dict') and callable(getattr(data, 'dict')):  # Pydantic models
        try:
            # Use Pydantic's JSON serialization which handles datetime properly
            return _serialize_for_logging(data.dict())
        except Exception:
            return str(data)
    elif hasattr(data, '__dict__'):  # Regular objects
        return _serialize_for_logging(data.__dict__)
    else:
        # For any other type, try to convert to string if it's not JSON serializable
        try:
            json.dumps(data)
            return data
        except (TypeError, ValueError):
            return str(data)


class CircuitBreaker:
    """Simple circuit breaker implementation."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def is_open(self) -> bool:
        if self.state == "OPEN":
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout):
                self.state = "HALF_OPEN"
                return False
            return True
        return False

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class VertexAIClient:
    """Async Vertex AI client with circuit breaker and retry logic."""

    def __init__(self, config: MonitoringConfig):
        if not VERTEX_AI_AVAILABLE:
            if config.send_to_vertex:
                logger.warning(
                    "Vertex AI dependencies not available. "
                    "Monitoring will work in offline mode only."
                )
            # Don't raise error, just log warning

        self.config = config
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_failure_threshold,
            recovery_timeout=config.circuit_breaker_recovery_timeout
        )

        self._aiplatform_client = None
        self._monitoring_client = None
        self._logging_client = None
        self._initialized = False

    async def _initialize_clients(self):
        """Initialize Google Cloud clients."""
        if self._initialized or not VERTEX_AI_AVAILABLE:
            return

        try:
            if self.config.credentials_path:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.config.credentials_path

            # Initialize aiplatform
            aiplatform.init(
                project=self.config.project_id,
                location=self.config.location
            )

            # Initialize Vertex AI for experiments
            if VERTEX_EXPERIMENTS_AVAILABLE:
                vertexai.init(
                    project=self.config.project_id,
                    location=self.config.location
                )

                # Initialize experiments
                init(
                    experiment="transcriptmonitoringv1",
                    project=self.config.project_id,
                    location=self.config.location
                )
                logger.info("Vertex AI Experiments initialized successfully")

            self._monitoring_client = monitoring_v3.MetricServiceAsyncClient()
            self._logging_client = gcp_logging.Client()

            self._initialized = True
            logger.info("Vertex AI clients initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI clients: {e}")
            raise VertexAIError(f"Client initialization failed: {e}")

    @asynccontextmanager
    async def _circuit_breaker_context(self):
        """Context manager for circuit breaker."""
        if self.circuit_breaker.is_open():
            raise VertexAIError("Circuit breaker is OPEN - Vertex AI temporarily unavailable")

        try:
            yield
            self.circuit_breaker.record_success()
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise e

    async def send_metrics_batch(self, batch: MonitoringBatch) -> bool:
        """Send a batch of metrics to Vertex AI with graceful error handling."""
        if not self.config.send_to_vertex:
            logger.info("Vertex AI sending disabled, skipping batch")
            return True

        await self._initialize_clients()

        success_count = 0
        total_operations = 3  # monitoring, evaluation, logs

        async with self._circuit_breaker_context():
            # Try each operation independently - don't fail entire batch if one fails

            # 1. Send monitoring metrics
            try:
                await self._send_monitoring_metrics(batch)
                success_count += 1
                logger.debug(f"Monitoring metrics sent successfully for batch {batch.batch_id}")
            except Exception as e:
                logger.warning(f"Failed to send monitoring metrics for batch {batch.batch_id}: {e}")
                # Continue with other operations

            # 2. Send evaluation data
            try:
                await self._send_evaluation_data(batch)
                success_count += 1
                logger.debug(f"Evaluation data sent successfully for batch {batch.batch_id}")
            except Exception as e:
                logger.warning(f"Failed to send evaluation data for batch {batch.batch_id}: {e}")
                # Continue with other operations

            # 3. Send structured logs
            try:
                await self._send_structured_logs(batch)
                success_count += 1
                logger.debug(f"Structured logs sent successfully for batch {batch.batch_id}")
            except Exception as e:
                logger.warning(f"Failed to send structured logs for batch {batch.batch_id}: {e}")
                # Continue - this is not critical

            # Consider success if at least one operation succeeded
            if success_count > 0:
                logger.info(f"Batch {batch.batch_id} sent with {success_count}/{total_operations} operations successful")
                return True
            else:
                # Only fail if all operations failed
                logger.error(f"All operations failed for batch {batch.batch_id}")
                raise VertexAIError(f"All batch operations failed for {batch.batch_id}")

    async def _send_monitoring_metrics(self, batch: MonitoringBatch):
        """Send metrics to Google Cloud Monitoring."""
        if not self._monitoring_client:
            return

        # Group metrics by type to prevent duplicates
        metric_groups = {}
        project_name = f"projects/{self.config.project_id}"

        for metrics in batch.metrics:
            # Create unique key for deduplication
            base_labels = {
                "user_id": metrics.business_metrics.user_id or "unknown",
                "request_id": metrics.request_id[:8]  # Short request ID for labeling
            }

            # Performance metrics
            if metrics.performance_metrics:
                metric_key = f"transcript/performance/duration_{metrics.timestamp.isoformat()}"
                if metric_key not in metric_groups:
                    metric_groups[metric_key] = self._create_time_series(
                        "transcript/performance/duration",
                        metrics.performance_metrics.total_duration_ms,
                        metrics.timestamp,
                        base_labels
                    )

            # Quality metrics
            if metrics.quality_metrics and metrics.quality_metrics.average_confidence:
                metric_key = f"transcript/quality/confidence_{metrics.timestamp.isoformat()}"
                if metric_key not in metric_groups:
                    metric_groups[metric_key] = self._create_time_series(
                        "transcript/quality/confidence",
                        metrics.quality_metrics.average_confidence,
                        metrics.timestamp,
                        base_labels
                    )

        if metric_groups:
            try:
                await self._monitoring_client.create_time_series(
                    name=project_name,
                    time_series=list(metric_groups.values())
                )
                logger.info(f"Successfully sent {len(metric_groups)} unique metrics to Cloud Monitoring")
            except Exception as e:
                # Log error but don't fail the entire process
                logger.warning(f"Failed to send metrics to Cloud Monitoring: {e}")
                if "400" not in str(e):  # Only retry on non-client errors
                    raise

    def _create_time_series(self, metric_type: str, value: float, timestamp: datetime, labels: Dict[str, str]):
        """Create a time series for monitoring with proper resource labels."""
        from google.cloud.monitoring_v3 import TimeSeries, Point, TimeInterval
        from google.protobuf.timestamp_pb2 import Timestamp
        import socket

        series = TimeSeries()
        series.metric.type = f"custom.googleapis.com/{metric_type}"
        series.resource.type = "gce_instance"

        # Required resource labels for gce_instance
        series.resource.labels["project_id"] = self.config.project_id
        series.resource.labels["zone"] = f"{self.config.location}-a"  # Default zone
        series.resource.labels["instance_id"] = os.getenv("INSTANCE_ID", socket.gethostname())

        # Add metric labels
        for label_key, label_value in labels.items():
            # Ensure label values are strings and not too long
            series.metric.labels[label_key] = str(label_value)[:100]

        point = Point()
        point.value.double_value = float(value)

        timestamp_pb = Timestamp()
        timestamp_pb.FromDatetime(timestamp)
        point.interval = TimeInterval({"end_time": timestamp_pb})

        series.points = [point]
        return series

    async def _send_evaluation_data(self, batch: MonitoringBatch):
        """Send evaluation data to Vertex AI Experiments with proper run creation."""
        if not VERTEX_EXPERIMENTS_AVAILABLE:
            logger.debug("Vertex AI Experiments not available, logging evaluation data locally")
            for metrics in batch.metrics:
                eval_request = VertexAIEvaluationRequest(
                    project_id=self.config.project_id,
                    location=self.config.location,
                    experiment_name=self.config.experiment_name,
                    metrics=metrics
                )
                eval_data = eval_request.to_vertex_format()
                logger.info(f"Evaluation data for {metrics.request_id}: {eval_data}")
            return

        try:
            for metrics in batch.metrics:
                run_id = f"transcript_{metrics.request_id}"

                # Create and execute experiment run
                await asyncio.get_event_loop().run_in_executor(
                    None, self._create_experiment_run, metrics, run_id
                )
                logger.info(f"Created Vertex AI Experiments run {run_id} for request {metrics.request_id}")

        except Exception as e:
            logger.warning(f"Failed to send evaluation data to Vertex AI Experiments: {e}")
            # Fall back to local logging
            for metrics in batch.metrics:
                eval_request = VertexAIEvaluationRequest(
                    project_id=self.config.project_id,
                    location=self.config.location,
                    experiment_name=self.config.experiment_name,
                    metrics=metrics
                )
                eval_data = eval_request.to_vertex_format()
                logger.info(f"Evaluation data for {metrics.request_id}: {eval_data}")

    def _create_experiment_run(self, metrics: TranscriptMetrics, run_id: str):
        """Create an experiment run with metrics and parameters."""
        try:
            with start_run(run_id=run_id) as run:
                # Log parameters first
                params = {}

                # Business parameters
                if metrics.business_metrics:
                    if metrics.business_metrics.user_id:
                        params["user_id"] = metrics.business_metrics.user_id
                    if metrics.business_metrics.session_id:
                        params["session_id"] = metrics.business_metrics.session_id

                # Audio metadata parameters
                if metrics.audio_metadata:
                    if metrics.audio_metadata.duration_seconds:
                        params["audio_duration_seconds"] = metrics.audio_metadata.duration_seconds
                    if metrics.audio_metadata.file_size_bytes:
                        params["audio_file_size_bytes"] = metrics.audio_metadata.file_size_bytes
                    if metrics.audio_metadata.sample_rate:
                        params["audio_sample_rate"] = metrics.audio_metadata.sample_rate
                    if metrics.audio_metadata.channels:
                        params["audio_channels"] = metrics.audio_metadata.channels

                # System parameters
                params.update({
                    "request_id": metrics.request_id,
                    "status": metrics.status.value,
                    "timestamp": metrics.timestamp.isoformat()
                })

                if params:
                    log_params(params)

                # Log metrics
                experiment_metrics = {}

                # Performance metrics
                if metrics.performance_metrics:
                    experiment_metrics.update({
                        "duration_ms": metrics.performance_metrics.total_duration_ms,
                        "processing_time_ms": metrics.performance_metrics.audio_processing_time_ms or 0,
                        "inference_time_ms": metrics.performance_metrics.model_inference_time_ms or 0,
                    })

                # Quality metrics
                if metrics.quality_metrics:
                    if metrics.quality_metrics.word_count is not None:
                        experiment_metrics["word_count"] = metrics.quality_metrics.word_count
                    if metrics.quality_metrics.average_confidence is not None:
                        experiment_metrics["confidence"] = metrics.quality_metrics.average_confidence
                    if metrics.quality_metrics.transcript_length is not None:
                        experiment_metrics["transcript_length"] = metrics.quality_metrics.transcript_length
                    if metrics.quality_metrics.token_count is not None:
                        experiment_metrics["token_count"] = metrics.quality_metrics.token_count

                # Success rate metric
                experiment_metrics["success_rate"] = 1.0 if metrics.status.value == "success" else 0.0

                # Error metrics
                if metrics.error_info:
                    experiment_metrics["has_error"] = 1.0
                    # Add error type as parameter since it's categorical
                    if metrics.error_info.error_type:
                        log_params({"error_type": metrics.error_info.error_type})
                else:
                    experiment_metrics["has_error"] = 0.0

                if experiment_metrics:
                    log_metrics(experiment_metrics)

            logger.debug(f"Successfully created experiment run {run_id}")

        except Exception as e:
            logger.warning(f"Failed to create experiment run {run_id}: {e}")
            raise

    async def _send_structured_logs(self, batch: MonitoringBatch):
        """Send structured logs to Google Cloud Logging with proper datetime serialization."""
        if not self._logging_client:
            return

        logger_name = f"transcript_monitoring_{self.config.experiment_name}"
        cloud_logger = self._logging_client.logger(logger_name)

        for metrics in batch.metrics:
            # Create log entry and serialize all datetime objects
            raw_log_entry = {
                "request_id": metrics.request_id,
                "timestamp": metrics.timestamp.isoformat(),
                "status": metrics.status.value if hasattr(metrics.status, 'value') else str(metrics.status),
                "performance": metrics.performance_metrics.dict() if metrics.performance_metrics else None,
                "quality": metrics.quality_metrics.dict() if metrics.quality_metrics else None,
                "business": metrics.business_metrics.dict() if metrics.business_metrics else None,
                "error": metrics.error_info.dict() if metrics.error_info else None
            }

            # Serialize all datetime objects and other non-JSON-serializable objects
            log_entry = _serialize_for_logging(raw_log_entry)

            # Map status to proper severity levels
            severity_mapping = {
                "success": "INFO",
                "partial": "WARNING",
                "failure": "ERROR"
            }

            status_str = str(metrics.status).lower()
            severity = severity_mapping.get(status_str, "INFO")

            try:
                cloud_logger.log_struct(
                    log_entry,
                    severity=severity
                )
                logger.debug(f"Successfully sent log entry for request {metrics.request_id} with severity {severity}")
            except Exception as e:
                logger.warning(f"Failed to send log entry for request {metrics.request_id}: {e}")
                # Continue with other log entries

    async def send_single_metric(self, metrics: TranscriptMetrics) -> bool:
        """Send a single metric immediately."""
        batch = MonitoringBatch(
            batch_id=f"single_{metrics.request_id}",
            metrics=[metrics]
        )
        return await self.send_metrics_batch(batch)

    async def health_check(self) -> bool:
        """Perform a health check on Vertex AI services."""
        if not self.config.send_to_vertex:
            return True

        try:
            await self._initialize_clients()
            # Simple health check - just verify we can create a client
            return not self.circuit_breaker.is_open()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get current client status."""
        return {
            "initialized": self._initialized,
            "vertex_ai_available": VERTEX_AI_AVAILABLE,
            "circuit_breaker_state": self.circuit_breaker.state,
            "circuit_breaker_failures": self.circuit_breaker.failure_count,
            "send_to_vertex": self.config.send_to_vertex
        }