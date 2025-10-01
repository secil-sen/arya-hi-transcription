"""
Decorators and context managers for seamless transcript monitoring integration.
"""

import asyncio
import functools
import logging
import uuid
from typing import Callable, Any, Optional, Dict, Union
from contextlib import asynccontextmanager

from .config import MonitoringConfig
from .metrics_collector import MetricsCollector
from .vertex_client import VertexAIClient
from .background_sender import BackgroundMetricsSender
from .exceptions import MonitoringError
from .models import TranscriptStatus

logger = logging.getLogger(__name__)


class MonitoringManager:
    """Singleton manager for monitoring components."""

    _instance: Optional['MonitoringManager'] = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.config = MonitoringConfig.from_env()
            self.metrics_collector = MetricsCollector(self.config)
            self.vertex_client = VertexAIClient(self.config)
            self.background_sender = BackgroundMetricsSender(self.config, self.vertex_client)
            self._started = False
            MonitoringManager._initialized = True

    async def start(self):
        """Start monitoring services."""
        if not self._started and self.config.monitoring_enabled:
            try:
                await self.background_sender.start()
                self._started = True
                logger.info("Monitoring manager started")
            except Exception as e:
                logger.error(f"Failed to start monitoring manager: {e}")

    async def stop(self):
        """Stop monitoring services."""
        if self._started:
            try:
                await self.background_sender.stop()
                self._started = False
                logger.info("Monitoring manager stopped")
            except Exception as e:
                logger.error(f"Error stopping monitoring manager: {e}")

    def is_enabled(self) -> bool:
        """Check if monitoring is enabled."""
        return self.config.monitoring_enabled

    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            "config": self.config.to_dict(),
            "background_sender": self.background_sender.get_stats(),
            "vertex_client": self.vertex_client.get_status(),
            "active_contexts": self.metrics_collector.get_active_context_count()
        }


def transcript_monitor(
    project_id: Optional[str] = None,
    user_context: Optional[Callable] = None,
    session_context: Optional[Callable] = None,
    track_audio_metadata: bool = True,
    send_to_vertex: bool = True,
    custom_config: Optional[Dict[str, Any]] = None
):
    """
    Decorator for monitoring transcript functions.

    Args:
        project_id: Override project ID for this function
        user_context: Function to extract user_id from function args/kwargs
        session_context: Function to extract session_id from function args/kwargs
        track_audio_metadata: Whether to track audio metadata
        send_to_vertex: Whether to send metrics to Vertex AI
        custom_config: Custom configuration overrides

    Usage:
        @transcript_monitor(
            user_context=lambda *args, **kwargs: kwargs.get('user_id'),
            track_audio_metadata=True
        )
        async def transcribe_audio(audio_data: bytes, user_id: str) -> dict:
            # Your transcription logic
            return {"transcript": "...", "confidence": 0.95}
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            manager = MonitoringManager()

            # Start manager if not already started
            if not manager._started:
                await manager.start()
                logger.debug("Monitoring manager started with Vertex AI Experiments integration")

            if not manager.is_enabled():
                # If monitoring is disabled, just call the function
                return await func(*args, **kwargs)

            # Generate request ID
            request_id = f"req_{uuid.uuid4().hex[:12]}"

            # Extract user and session context
            user_id = None
            session_id = None

            try:
                if user_context:
                    user_id = user_context(*args, **kwargs)
                if session_context:
                    session_id = session_context(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to extract context: {e}")

            # Create metrics context
            context = manager.metrics_collector.create_context(
                request_id=request_id,
                user_id=user_id,
                session_id=session_id
            )

            # Add audio metadata if available
            if track_audio_metadata and args:
                try:
                    # Assume first argument might be audio data
                    audio_data = args[0]
                    manager.metrics_collector.add_audio_metadata(
                        request_id,
                        audio_data=audio_data
                    )
                except Exception as e:
                    logger.debug(f"Could not extract audio metadata: {e}")

            # Execute function with monitoring
            result = None
            error = None

            try:
                # Mark start of all processing phases
                manager.metrics_collector.mark_audio_processing_start(request_id)
                manager.metrics_collector.mark_model_inference_start(request_id)
                manager.metrics_collector.mark_pipeline_start(request_id)

                # Call the actual function
                result = await func(*args, **kwargs)

                # Mark end of all processing phases
                manager.metrics_collector.mark_pipeline_end(request_id)
                manager.metrics_collector.mark_model_inference_end(request_id)
                manager.metrics_collector.mark_audio_processing_end(request_id)

                return result

            except Exception as e:
                error = e
                raise

            finally:
                # Finalize metrics
                try:
                    # Enhanced debug logging for result analysis
                    logger.info(f"SEGMENT DEBUG: Decorator finalizing metrics for {request_id}")
                    logger.info(f"SEGMENT DEBUG: Result type: {type(result)}")
                    logger.info(f"SEGMENT DEBUG: Function result: {result}")

                    if result:
                        if isinstance(result, dict):
                            logger.info(f"SEGMENT DEBUG: Result keys: {list(result.keys())}")
                            logger.info(f"SEGMENT DEBUG: Has text_corrected: {'text_corrected' in result}")
                            logger.info(f"SEGMENT DEBUG: Has confidence: {'confidence' in result}")
                            logger.info(f"SEGMENT DEBUG: text_corrected value: {result.get('text_corrected', 'NOT_FOUND')}")
                            logger.info(f"SEGMENT DEBUG: confidence value: {result.get('confidence', 'NOT_FOUND')}")
                        logger.info(f"SEGMENT DEBUG: Result preview: {str(result)[:200]}...")
                    else:
                        logger.info("SEGMENT DEBUG: Result is None or empty")

                    metrics = manager.metrics_collector.finalize_metrics(
                        request_id=request_id,
                        result=result,
                        error=error
                    )

                    # Queue metrics for background sending only if metrics were created
                    if metrics is not None:
                        await manager.background_sender.queue_metrics(metrics)
                        logger.debug(f"Queued metrics for {request_id}")
                    else:
                        logger.info(f"No metrics created for {request_id} (empty result skipped)")

                except Exception as e:
                    logger.error(f"Failed to finalize metrics for {request_id}: {e}")

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, create a simple async wrapper
            async def async_func(*args, **kwargs):
                return func(*args, **kwargs)

            # Run in event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            return loop.run_until_complete(async_wrapper(*args, **kwargs))

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class TranscriptMonitor:
    """Context manager for transcript monitoring."""

    def __init__(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ):
        self.user_id = user_id
        self.session_id = session_id
        self.request_id = request_id or f"ctx_{uuid.uuid4().hex[:12]}"
        self.custom_metadata = custom_metadata or {}

        self.manager = MonitoringManager()
        self.context = None
        self.result = None
        self.error = None

    async def __aenter__(self):
        """Async context manager entry."""
        # Start manager if not already started
        if not self.manager._started:
            await self.manager.start()
            logger.debug("Monitoring manager started with Vertex AI Experiments integration")

        if self.manager.is_enabled():
            # Create metrics context
            self.context = self.manager.metrics_collector.create_context(
                request_id=self.request_id,
                user_id=self.user_id,
                session_id=self.session_id
            )

            # Add custom metadata
            if self.custom_metadata:
                self.manager.metrics_collector.add_custom_metadata(
                    self.request_id,
                    **self.custom_metadata
                )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if not self.manager.is_enabled() or not self.context:
            return

        try:
            # Set error if exception occurred
            if exc_type is not None:
                self.error = exc_val

            # Debug logging for result analysis
            logger.debug(f"Context manager finalizing metrics for {self.request_id}")
            logger.debug(f"Result type: {type(self.result)}")
            if self.result:
                if isinstance(self.result, dict):
                    logger.debug(f"Result keys: {list(self.result.keys())}")
                logger.debug(f"Result preview: {str(self.result)[:200]}...")
            else:
                logger.debug("Result is None or empty")

            # Finalize metrics
            metrics = self.manager.metrics_collector.finalize_metrics(
                request_id=self.request_id,
                result=self.result,
                error=self.error
            )

            # Queue metrics for background sending only if metrics were created
            if metrics is not None:
                await self.manager.background_sender.queue_metrics(metrics)
                logger.debug(f"Queued metrics for {self.request_id}")
            else:
                logger.info(f"No metrics created for {self.request_id} (empty result skipped)")

        except Exception as e:
            logger.error(f"Failed to finalize context metrics for {self.request_id}: {e}")

    def __enter__(self):
        """Sync context manager entry."""
        return asyncio.run(self.__aenter__())

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit."""
        return asyncio.run(self.__aexit__(exc_type, exc_val, exc_tb))

    def mark_audio_processing_start(self):
        """Mark start of audio processing."""
        if self.manager.is_enabled():
            self.manager.metrics_collector.mark_audio_processing_start(self.request_id)

    def mark_audio_processing_end(self):
        """Mark end of audio processing."""
        if self.manager.is_enabled():
            self.manager.metrics_collector.mark_audio_processing_end(self.request_id)

    def mark_model_inference_start(self):
        """Mark start of model inference."""
        if self.manager.is_enabled():
            self.manager.metrics_collector.mark_model_inference_start(self.request_id)

    def mark_model_inference_end(self):
        """Mark end of model inference."""
        if self.manager.is_enabled():
            self.manager.metrics_collector.mark_model_inference_end(self.request_id)

    def mark_pipeline_start(self):
        """Mark start of pipeline processing."""
        if self.manager.is_enabled():
            self.manager.metrics_collector.mark_pipeline_start(self.request_id)

    def mark_pipeline_end(self):
        """Mark end of pipeline processing."""
        if self.manager.is_enabled():
            self.manager.metrics_collector.mark_pipeline_end(self.request_id)

    def add_audio_metadata(self, audio_data: Any = None, **metadata):
        """Add audio metadata."""
        if self.manager.is_enabled():
            self.manager.metrics_collector.add_audio_metadata(
                self.request_id,
                audio_data=audio_data,
                **metadata
            )

    def add_custom_metadata(self, **metadata):
        """Add custom metadata."""
        if self.manager.is_enabled():
            self.manager.metrics_collector.add_custom_metadata(
                self.request_id,
                **metadata
            )

    def set_result(self, result: Any):
        """Set the operation result."""
        self.result = result


# Global monitoring manager instance
_global_manager: Optional[MonitoringManager] = None


async def start_monitoring():
    """Start global monitoring services."""
    global _global_manager
    if _global_manager is None:
        _global_manager = MonitoringManager()
    await _global_manager.start()


async def stop_monitoring():
    """Stop global monitoring services."""
    global _global_manager
    if _global_manager:
        await _global_manager.stop()


def get_monitoring_stats() -> Dict[str, Any]:
    """Get global monitoring statistics."""
    global _global_manager
    if _global_manager:
        return _global_manager.get_stats()
    return {"error": "Monitoring not initialized"}


async def health_check() -> Dict[str, Any]:
    """Perform health check on monitoring system."""
    global _global_manager
    if not _global_manager:
        return {"healthy": False, "error": "Not initialized"}

    try:
        vertex_health = await _global_manager.vertex_client.health_check()
        sender_health = await _global_manager.background_sender.health_check()

        return {
            "healthy": vertex_health and sender_health,
            "vertex_ai": vertex_health,
            "background_sender": sender_health,
            "stats": _global_manager.get_stats()
        }
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e)
        }