"""
Metrics collector for gathering performance, quality, and business metrics.
"""

import time
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, Optional, List, Callable
from dataclasses import dataclass, field

from .models import (
    TranscriptMetrics,
    PerformanceMetrics,
    QualityMetrics,
    BusinessMetrics,
    AudioMetadata,
    ErrorInfo,
    TranscriptStatus
)
from .config import MonitoringConfig
from .exceptions import MetricsCollectionError

logger = logging.getLogger(__name__)


@dataclass
class MetricsContext:
    """Context for metrics collection during request processing."""
    request_id: str
    start_time: float = field(default_factory=time.time)
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    audio_processing_start: Optional[float] = None
    audio_processing_end: Optional[float] = None
    model_inference_start: Optional[float] = None
    model_inference_end: Optional[float] = None
    pipeline_start: Optional[float] = None
    pipeline_end: Optional[float] = None

    audio_metadata: Optional[Dict[str, Any]] = None
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and processes metrics from transcript operations."""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._active_contexts: Dict[str, MetricsContext] = {}

    def create_context(self, request_id: str, user_id: Optional[str] = None, session_id: Optional[str] = None) -> MetricsContext:
        """Create a new metrics collection context."""
        context = MetricsContext(
            request_id=request_id,
            user_id=user_id,
            session_id=session_id
        )
        self._active_contexts[request_id] = context
        return context

    def get_context(self, request_id: str) -> Optional[MetricsContext]:
        """Get an existing metrics context."""
        return self._active_contexts.get(request_id)

    def mark_audio_processing_start(self, request_id: str):
        """Mark the start of audio processing."""
        context = self._active_contexts.get(request_id)
        if context:
            context.audio_processing_start = time.time()

    def mark_audio_processing_end(self, request_id: str):
        """Mark the end of audio processing."""
        context = self._active_contexts.get(request_id)
        if context:
            context.audio_processing_end = time.time()

    def mark_model_inference_start(self, request_id: str):
        """Mark the start of model inference."""
        context = self._active_contexts.get(request_id)
        if context:
            context.model_inference_start = time.time()

    def mark_model_inference_end(self, request_id: str):
        """Mark the end of model inference."""
        context = self._active_contexts.get(request_id)
        if context:
            context.model_inference_end = time.time()

    def mark_pipeline_start(self, request_id: str):
        """Mark the start of pipeline processing."""
        context = self._active_contexts.get(request_id)
        if context:
            context.pipeline_start = time.time()

    def mark_pipeline_end(self, request_id: str):
        """Mark the end of pipeline processing."""
        context = self._active_contexts.get(request_id)
        if context:
            context.pipeline_end = time.time()

    def add_audio_metadata(self, request_id: str, audio_data: Any = None, **metadata):
        """Add audio metadata to the context."""
        context = self._active_contexts.get(request_id)
        if not context:
            return

        if context.audio_metadata is None:
            context.audio_metadata = {}

        # Extract metadata from audio data if provided
        if audio_data and self.config.track_audio_metadata:
            try:
                extracted = self._extract_audio_metadata(audio_data)
                context.audio_metadata.update(extracted)
            except Exception as e:
                logger.warning(f"Failed to extract audio metadata: {e}")

        # Add custom metadata
        context.audio_metadata.update(metadata)

    def add_custom_metadata(self, request_id: str, **metadata):
        """Add custom metadata to the context."""
        context = self._active_contexts.get(request_id)
        if context:
            context.custom_metadata.update(metadata)

    def _extract_audio_metadata(self, audio_data: Any) -> Dict[str, Any]:
        """Extract metadata from audio data."""
        metadata = {}

        try:
            # Handle bytes data
            if isinstance(audio_data, bytes):
                metadata["file_size_bytes"] = len(audio_data)

                # Try to extract more detailed info using librosa/pydub if available
                try:
                    import io
                    import librosa
                    import soundfile as sf

                    # Try to load audio data
                    audio_buffer = io.BytesIO(audio_data)
                    y, sr = librosa.load(audio_buffer, sr=None)

                    metadata.update({
                        "duration_seconds": len(y) / sr,
                        "sample_rate": sr,
                        "channels": 1 if len(y.shape) == 1 else y.shape[0]
                    })

                except ImportError:
                    logger.debug("librosa not available for audio metadata extraction")
                except Exception as e:
                    logger.debug(f"Could not extract detailed audio metadata: {e}")

            # Handle file path
            elif isinstance(audio_data, (str, type(None))):
                if audio_data and hasattr(audio_data, 'stat'):
                    metadata["file_size_bytes"] = audio_data.stat().st_size

        except Exception as e:
            logger.warning(f"Error extracting audio metadata: {e}")

        return metadata

    def finalize_metrics(
        self,
        request_id: str,
        result: Any = None,
        error: Exception = None,
        status: Optional[TranscriptStatus] = None
    ) -> Optional[TranscriptMetrics]:
        """Finalize and create metrics for a completed request."""

        context = self._active_contexts.get(request_id)
        if not context:
            raise MetricsCollectionError(f"No metrics context found for request {request_id}")

        # Check if result is meaningfully empty (e.g., empty segment)
        if self._is_empty_result(result) and not error:
            logger.info(f"EMPTY RESULT: Skipping metrics for {request_id} - result appears to be empty segment")
            # Clean up context but don't create metrics
            del self._active_contexts[request_id]
            return None

        end_time = time.time()

        try:
            # Determine status
            final_status = status
            if final_status is None:
                if error:
                    final_status = TranscriptStatus.FAILURE
                elif result:
                    final_status = TranscriptStatus.SUCCESS
                else:
                    final_status = TranscriptStatus.PARTIAL

            # Build performance metrics
            performance_metrics = self._build_performance_metrics(context, end_time)

            # Build quality metrics
            quality_metrics = self._build_quality_metrics(result)

            # Build business metrics
            business_metrics = self._build_business_metrics(context)

            # Build audio metadata
            audio_metadata = self._build_audio_metadata(context)

            # Build error info
            error_info = self._build_error_info(error) if error else None

            metrics = TranscriptMetrics(
                request_id=request_id,
                timestamp=datetime.now(),
                status=final_status,
                audio_metadata=audio_metadata,
                performance_metrics=performance_metrics,
                quality_metrics=quality_metrics,
                business_metrics=business_metrics,
                error_info=error_info,
                custom_metadata=context.custom_metadata
            )

            # Clean up context
            del self._active_contexts[request_id]

            return metrics

        except Exception as e:
            logger.error(f"Error finalizing metrics for {request_id}: {e}")
            # Clean up context even on error
            if request_id in self._active_contexts:
                del self._active_contexts[request_id]
            raise MetricsCollectionError(f"Failed to finalize metrics: {e}")

    def _build_performance_metrics(self, context: MetricsContext, end_time: float) -> PerformanceMetrics:
        """Build performance metrics with realistic timing estimates."""
        # Total duration: from request start to request end
        total_duration_ms = (end_time - context.start_time) * 1000

        # Check if we have specific timing measurements
        has_specific_audio_timing = (context.audio_processing_start and context.audio_processing_end and
                                   context.audio_processing_start != context.audio_processing_end)
        has_specific_model_timing = (context.model_inference_start and context.model_inference_end and
                                   context.model_inference_start != context.model_inference_end)
        has_specific_pipeline_timing = (context.pipeline_start and context.pipeline_end and
                                      context.pipeline_start != context.pipeline_end)

        if has_specific_audio_timing:
            audio_processing_time_ms = (context.audio_processing_end - context.audio_processing_start) * 1000
            logger.debug(f"Using measured audio processing time: {audio_processing_time_ms}ms")
        else:
            # Estimate: Audio processing typically takes 20-30% of total time
            audio_processing_time_ms = total_duration_ms * 0.25
            logger.debug(f"Estimated audio processing time: {audio_processing_time_ms}ms")

        if has_specific_model_timing:
            model_inference_time_ms = (context.model_inference_end - context.model_inference_start) * 1000
            logger.debug(f"Using measured model inference time: {model_inference_time_ms}ms")
        else:
            # Estimate: Model inference typically takes 60-70% of total time
            model_inference_time_ms = total_duration_ms * 0.65
            logger.debug(f"Estimated model inference time: {model_inference_time_ms}ms")

        if has_specific_pipeline_timing:
            pipeline_duration_ms = (context.pipeline_end - context.pipeline_start) * 1000
            logger.debug(f"Using measured pipeline duration: {pipeline_duration_ms}ms")
        else:
            # Pipeline is typically 90-95% of total (excludes setup/teardown)
            pipeline_duration_ms = total_duration_ms * 0.95
            logger.debug(f"Estimated pipeline duration: {pipeline_duration_ms}ms")

        # Ensure no negative values and realistic constraints
        total_duration_ms = max(0, total_duration_ms)
        audio_processing_time_ms = max(0, min(audio_processing_time_ms, total_duration_ms))
        model_inference_time_ms = max(0, min(model_inference_time_ms, total_duration_ms))
        pipeline_duration_ms = max(0, min(pipeline_duration_ms, total_duration_ms))

        # Debug logging for timing validation
        logger.debug(f"TIMING DEBUG: Total: {total_duration_ms}ms, Pipeline: {pipeline_duration_ms}ms, "
                    f"Audio: {audio_processing_time_ms}ms, Model: {model_inference_time_ms}ms")
        logger.debug(f"TIMING SOURCE: Audio={'measured' if has_specific_audio_timing else 'estimated'}, "
                    f"Model={'measured' if has_specific_model_timing else 'estimated'}, "
                    f"Pipeline={'measured' if has_specific_pipeline_timing else 'estimated'}")

        return PerformanceMetrics(
            request_start_time=datetime.fromtimestamp(context.start_time),
            request_end_time=datetime.fromtimestamp(end_time),
            total_duration_ms=total_duration_ms,
            audio_processing_time_ms=audio_processing_time_ms,
            model_inference_time_ms=model_inference_time_ms,
            pipeline_duration_ms=pipeline_duration_ms
        )

    def _build_quality_metrics(self, result: Any) -> QualityMetrics:
        """Build quality metrics from result."""
        quality_metrics = QualityMetrics()

        if not result:
            logger.debug("No result provided for quality metrics extraction")
            return quality_metrics

        # Enhanced debug logging to understand result structure
        logger.info(f"METRICS DEBUG: Extracting quality metrics from result type: {type(result)}")
        if isinstance(result, dict):
            logger.info(f"METRICS DEBUG: Result keys: {list(result.keys())}")
            # Log actual values for debugging
            for key, value in result.items():
                if isinstance(value, str):
                    if len(value) > 100:
                        logger.info(f"METRICS DEBUG:   {key}: '{value[:100]}...' (length: {len(value)})")
                    else:
                        logger.info(f"METRICS DEBUG:   {key}: '{value}' (length: {len(value)})")
                else:
                    logger.info(f"METRICS DEBUG:   {key}: {value} (type: {type(value)})")
        elif hasattr(result, '__dict__'):
            logger.info(f"METRICS DEBUG: Result attributes: {list(result.__dict__.keys())}")

        try:
            transcript_text = ""
            confidence_value = None

            # Handle different result formats
            if isinstance(result, dict):
                # Extract transcript text from various possible keys (prioritize text_corrected)
                transcript_text = (result.get("text_corrected") or
                                 result.get("transcript") or
                                 result.get("text") or
                                 result.get("transcription") or
                                 result.get("output") or "")

                logger.debug(f"Text field search results:")
                logger.debug(f"  text_corrected: {result.get('text_corrected', 'NOT_FOUND')}")
                logger.debug(f"  transcript: {result.get('transcript', 'NOT_FOUND')}")
                logger.debug(f"  text: {result.get('text', 'NOT_FOUND')}")
                logger.debug(f"  Selected transcript_text: '{transcript_text[:100] if transcript_text else 'EMPTY'}...'")

                # If no text found in standard fields, scan all fields for text-like content
                if not transcript_text:
                    logger.debug("No transcript found in standard fields, scanning all fields...")
                    for key, value in result.items():
                        if isinstance(value, str) and len(value.strip()) > 10:  # Likely text content
                            logger.debug(f"  Found potential text in '{key}': '{value[:100]}...'")
                            if not transcript_text:  # Use first text-like field found
                                transcript_text = value
                                logger.debug(f"  Using '{key}' as transcript text")

                # Extract confidence scores from various formats with comprehensive scanning
                confidence_value = None

                # Primary confidence field search
                confidence_fields = ['confidence', 'confidence_score', 'avg_confidence', 'score', 'conf']
                for field in confidence_fields:
                    if field in result and result[field] is not None:
                        confidence_value = result[field]
                        logger.info(f"CONFIDENCE DEBUG: Found confidence in '{field}': {confidence_value}")
                        break

                # If no confidence found in standard fields, scan all numeric fields
                if confidence_value is None:
                    logger.info("CONFIDENCE DEBUG: No confidence in standard fields, scanning all numeric fields...")
                    for key, value in result.items():
                        if isinstance(value, (int, float)) and 0 <= value <= 1 and 'confidence' in key.lower():
                            confidence_value = value
                            logger.info(f"CONFIDENCE DEBUG: Found potential confidence in '{key}': {value}")
                            break

                logger.info(f"CONFIDENCE DEBUG: Field search results:")
                for field in confidence_fields:
                    logger.info(f"  {field}: {result.get(field, 'NOT_FOUND')}")
                logger.info(f"CONFIDENCE DEBUG: Selected confidence_value: {confidence_value}")

                # Handle segments/chunks format (common in speech recognition)
                segments = result.get("segments") or result.get("chunks")
                if segments and isinstance(segments, list):
                    segment_confidences = []
                    total_text = ""

                    for segment in segments:
                        if isinstance(segment, dict):
                            seg_text = segment.get("text", "")
                            total_text += seg_text + " "

                            seg_conf = segment.get("confidence")
                            if seg_conf is not None:
                                segment_confidences.append(float(seg_conf))

                    # Use segments data if available and main transcript is empty
                    if total_text.strip() and not transcript_text.strip():
                        transcript_text = total_text.strip()

                    # Use segment confidences if available and no main confidence
                    if segment_confidences and confidence_value is None:
                        quality_metrics.confidence_scores = segment_confidences
                        quality_metrics.average_confidence = sum(segment_confidences) / len(segment_confidences)

                # Extract token count from various possible keys
                tokens = (result.get("tokens") or
                         result.get("token_count") or
                         result.get("num_tokens"))
                if tokens and isinstance(tokens, (int, list)):
                    if isinstance(tokens, list):
                        quality_metrics.token_count = len(tokens)
                    else:
                        quality_metrics.token_count = int(tokens)

            elif hasattr(result, 'text') or hasattr(result, 'transcript'):
                # Handle result objects with text attribute
                transcript_text = getattr(result, 'text', None) or getattr(result, 'transcript', "")

                # Extract confidence if available
                if hasattr(result, 'confidence'):
                    confidence_value = getattr(result, 'confidence')

                # Check for segments attribute
                if hasattr(result, 'segments'):
                    segments = getattr(result, 'segments')
                    if segments and isinstance(segments, list):
                        segment_confidences = []
                        for segment in segments:
                            if hasattr(segment, 'confidence'):
                                seg_conf = getattr(segment, 'confidence')
                                if seg_conf is not None:
                                    segment_confidences.append(float(seg_conf))

                        if segment_confidences and confidence_value is None:
                            quality_metrics.confidence_scores = segment_confidences
                            quality_metrics.average_confidence = sum(segment_confidences) / len(segment_confidences)

            # Handle string results directly
            elif isinstance(result, str) and result.strip():
                transcript_text = result

            # Process transcript text if we found any
            if transcript_text and transcript_text.strip():
                cleaned_text = transcript_text.strip()
                logger.debug(f"Found transcript text (length: {len(cleaned_text)}): {cleaned_text[:100]}...")

                # Basic text metrics
                quality_metrics.transcript_length = len(cleaned_text)
                words = cleaned_text.split()
                quality_metrics.word_count = len(words)

                # Estimate token count if not provided
                if quality_metrics.token_count is None:
                    quality_metrics.token_count = self._estimate_token_count(cleaned_text)

                logger.debug(f"Calculated metrics - Length: {quality_metrics.transcript_length}, Words: {quality_metrics.word_count}, Tokens: {quality_metrics.token_count}")

            # Process confidence if we found any
            if confidence_value is not None and quality_metrics.average_confidence is None:
                logger.debug(f"Found confidence value: {confidence_value} (type: {type(confidence_value)})")

                if isinstance(confidence_value, (int, float)):
                    quality_metrics.average_confidence = float(confidence_value)
                    quality_metrics.confidence_scores = [float(confidence_value)]
                    logger.debug(f"Set confidence: {quality_metrics.average_confidence}")
                elif isinstance(confidence_value, list) and len(confidence_value) > 0:
                    valid_confidences = [float(c) for c in confidence_value if c is not None and isinstance(c, (int, float))]
                    if valid_confidences:
                        quality_metrics.confidence_scores = valid_confidences
                        quality_metrics.average_confidence = sum(valid_confidences) / len(valid_confidences)
                        logger.debug(f"Set average confidence from list: {quality_metrics.average_confidence}")
            else:
                logger.debug(f"No confidence value found or already set (confidence_value: {confidence_value}, current avg: {quality_metrics.average_confidence})")

            # Set defaults for empty values to avoid nulls
            if quality_metrics.average_confidence is None:
                quality_metrics.average_confidence = 0.0
                logger.debug("Set default confidence to 0.0")
            if quality_metrics.confidence_scores is None:
                quality_metrics.confidence_scores = [0.0]
                logger.debug("Set default confidence scores to [0.0]")
            if quality_metrics.transcript_length is None:
                quality_metrics.transcript_length = 0
                logger.debug("Set default transcript length to 0")
            if quality_metrics.word_count is None:
                quality_metrics.word_count = 0
                logger.debug("Set default word count to 0")
            if quality_metrics.token_count is None:
                quality_metrics.token_count = 0
                logger.debug("Set default token count to 0")

            logger.debug(f"Final quality metrics: length={quality_metrics.transcript_length}, "
                        f"words={quality_metrics.word_count}, tokens={quality_metrics.token_count}, "
                        f"confidence={quality_metrics.average_confidence}")

        except Exception as e:
            logger.warning(f"Error extracting quality metrics: {e}")
            # Set safe defaults on error
            quality_metrics.average_confidence = 0.0
            quality_metrics.confidence_scores = [0.0]
            quality_metrics.transcript_length = 0
            quality_metrics.word_count = 0
            quality_metrics.token_count = 0

        return quality_metrics

    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count from text using rough heuristics."""
        if not text or not text.strip():
            return 0

        # Rough estimation: ~4 characters per token for most languages
        # This is a conservative estimate that works reasonably well for Turkish and English
        char_count = len(text.strip())
        estimated_tokens = max(1, char_count // 4)

        return estimated_tokens

    def _is_empty_result(self, result: Any) -> bool:
        """Check if result is meaningfully empty (e.g., empty segment)."""
        if not result:
            return True

        if isinstance(result, dict):
            # Check if all text fields are empty or None
            text_fields = ['text_corrected', 'transcript', 'text', 'transcription', 'output']
            has_meaningful_text = False

            for field in text_fields:
                text_value = result.get(field)
                if text_value and isinstance(text_value, str) and text_value.strip():
                    has_meaningful_text = True
                    break

            # Also check for any string value that might be meaningful text
            if not has_meaningful_text:
                for key, value in result.items():
                    if isinstance(value, str) and len(value.strip()) > 5:  # At least 5 chars
                        has_meaningful_text = True
                        break

            return not has_meaningful_text

        elif isinstance(result, str):
            return len(result.strip()) == 0

        # If it's some other type and not None, assume it's meaningful
        return False

    def _build_business_metrics(self, context: MetricsContext) -> BusinessMetrics:
        """Build business metrics from context."""
        return BusinessMetrics(
            user_id=context.user_id,
            session_id=context.session_id,
            api_call_count=1,
            cost_estimate=None  # Could be calculated based on usage
        )

    def _build_audio_metadata(self, context: MetricsContext) -> AudioMetadata:
        """Build audio metadata from context."""
        audio_meta = AudioMetadata()

        if context.audio_metadata:
            audio_meta.duration_seconds = context.audio_metadata.get("duration_seconds")
            audio_meta.file_size_bytes = context.audio_metadata.get("file_size_bytes")
            audio_meta.format = context.audio_metadata.get("format")
            audio_meta.sample_rate = context.audio_metadata.get("sample_rate")
            audio_meta.channels = context.audio_metadata.get("channels")

        return audio_meta

    def _build_error_info(self, error: Exception) -> ErrorInfo:
        """Build error info from exception."""
        return ErrorInfo(
            error_type=type(error).__name__,
            error_message=str(error),
            error_code=getattr(error, 'code', None),
            stack_trace=traceback.format_exc()
        )

    def cleanup_stale_contexts(self, max_age_seconds: int = 300):
        """Clean up stale contexts that haven't been finalized."""
        current_time = time.time()
        stale_contexts = []

        for request_id, context in self._active_contexts.items():
            if current_time - context.start_time > max_age_seconds:
                stale_contexts.append(request_id)

        for request_id in stale_contexts:
            logger.warning(f"Cleaning up stale context for request {request_id}")
            del self._active_contexts[request_id]

    def get_active_context_count(self) -> int:
        """Get the number of active contexts."""
        return len(self._active_contexts)