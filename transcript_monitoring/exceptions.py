"""
Custom exceptions for transcript monitoring.
"""

class MonitoringError(Exception):
    """Base exception for monitoring-related errors."""
    pass


class VertexAIError(MonitoringError):
    """Exception raised when Vertex AI operations fail."""
    pass


class ConfigurationError(MonitoringError):
    """Exception raised for configuration-related issues."""
    pass


class MetricsCollectionError(MonitoringError):
    """Exception raised during metrics collection."""
    pass


class BackgroundSenderError(MonitoringError):
    """Exception raised in background metrics sender."""
    pass