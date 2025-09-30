"""
Centralized error handling and logging configuration.
"""

import logging
import traceback
from typing import Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass


class ErrorCode(Enum):
    """Standard error codes for transcript service."""

    # Input validation errors
    INVALID_REQUEST = "INVALID_REQUEST"
    MISSING_AUDIO_FILE = "MISSING_AUDIO_FILE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"

    # Processing errors
    AUDIO_EXTRACTION_FAILED = "AUDIO_EXTRACTION_FAILED"
    TRANSCRIPTION_FAILED = "TRANSCRIPTION_FAILED"
    DIARIZATION_FAILED = "DIARIZATION_FAILED"
    ENRICHMENT_FAILED = "ENRICHMENT_FAILED"

    # System errors
    MODEL_LOADING_FAILED = "MODEL_LOADING_FAILED"
    INSUFFICIENT_MEMORY = "INSUFFICIENT_MEMORY"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"

    # External service errors
    GEMINI_API_ERROR = "GEMINI_API_ERROR"
    S3_ACCESS_ERROR = "S3_ACCESS_ERROR"
    MONITORING_ERROR = "MONITORING_ERROR"


@dataclass
class ErrorContext:
    """Context information for errors."""

    error_code: ErrorCode
    message: str
    user_message: str
    details: Optional[Dict[str, Any]] = None
    recoverable: bool = True
    retry_after_seconds: Optional[int] = None


class TranscriptException(Exception):
    """Base exception class for transcript service."""

    def __init__(self, context: ErrorContext, original_error: Optional[Exception] = None):
        self.context = context
        self.original_error = original_error
        super().__init__(context.message)


class ErrorHandler:
    """Centralized error handling."""

    @staticmethod
    def handle_error(
        error: Exception,
        error_code: ErrorCode,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ) -> ErrorContext:
        """
        Handle and log error with proper context.

        Args:
            error: Original exception
            error_code: Standardized error code
            user_message: User-friendly error message
            context: Additional context information
            recoverable: Whether the error is recoverable

        Returns:
            ErrorContext: Structured error context
        """
        logger = logging.getLogger(__name__)

        error_context = ErrorContext(
            error_code=error_code,
            message=str(error),
            user_message=user_message,
            details=context or {},
            recoverable=recoverable
        )

        # Log error with appropriate level
        if recoverable:
            logger.warning(
                f"Recoverable error [{error_code.value}]: {user_message}",
                extra={
                    "error_code": error_code.value,
                    "original_error": str(error),
                    "context": context,
                    "stack_trace": traceback.format_exc()
                }
            )
        else:
            logger.error(
                f"Critical error [{error_code.value}]: {user_message}",
                extra={
                    "error_code": error_code.value,
                    "original_error": str(error),
                    "context": context,
                    "stack_trace": traceback.format_exc()
                }
            )

        return error_context

    @staticmethod
    def create_user_response(error_context: ErrorContext) -> Dict[str, Any]:
        """Create user-friendly error response."""
        response = {
            "success": False,
            "error": {
                "code": error_context.error_code.value,
                "message": error_context.user_message,
                "recoverable": error_context.recoverable
            }
        }

        if error_context.retry_after_seconds:
            response["error"]["retry_after_seconds"] = error_context.retry_after_seconds

        return response


def configure_logging(log_level: str = "INFO", log_format: Optional[str] = None):
    """Configure application logging."""

    if not log_format:
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[%(filename)s:%(lineno)d] - %(message)s"
        )

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("/tmp/claude/transcript_service.log", mode="a")
        ]
    )

    # Configure specific loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# Decorator for error handling
def handle_service_errors(
    error_code: ErrorCode,
    user_message: str,
    recoverable: bool = True
):
    """Decorator to handle service errors consistently."""

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_context = ErrorHandler.handle_error(
                    error=e,
                    error_code=error_code,
                    user_message=user_message,
                    recoverable=recoverable
                )
                raise TranscriptException(error_context, e)

        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = ErrorHandler.handle_error(
                    error=e,
                    error_code=error_code,
                    user_message=user_message,
                    recoverable=recoverable
                )
                raise TranscriptException(error_context, e)

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Common error scenarios
class CommonErrors:
    """Pre-defined error contexts for common scenarios."""

    @staticmethod
    def invalid_audio_file(filename: str) -> ErrorContext:
        return ErrorContext(
            error_code=ErrorCode.MISSING_AUDIO_FILE,
            message=f"Audio file not found: {filename}",
            user_message="The provided audio file could not be found or accessed.",
            recoverable=False
        )

    @staticmethod
    def transcription_timeout() -> ErrorContext:
        return ErrorContext(
            error_code=ErrorCode.TIMEOUT_ERROR,
            message="Transcription process timed out",
            user_message="The transcription process took too long. Please try with a shorter audio file.",
            recoverable=True,
            retry_after_seconds=60
        )

    @staticmethod
    def gemini_api_error(api_error: str) -> ErrorContext:
        return ErrorContext(
            error_code=ErrorCode.GEMINI_API_ERROR,
            message=f"Gemini API error: {api_error}",
            user_message="Content enrichment service is temporarily unavailable.",
            recoverable=True,
            retry_after_seconds=30
        )