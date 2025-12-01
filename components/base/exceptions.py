"""
Custom exceptions for component services.
"""


class ComponentError(Exception):
    """Base exception for all component errors."""

    def __init__(self, message: str, component: str = None, details: dict = None):
        self.message = message
        self.component = component
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "component": self.component,
            "details": self.details,
        }


class ConfigurationError(ComponentError):
    """Raised when component configuration is invalid or missing."""

    def __init__(self, message: str, component: str = None, missing_keys: list = None):
        details = {"missing_keys": missing_keys} if missing_keys else {}
        super().__init__(message, component, details)


class ProcessingError(ComponentError):
    """Raised when component processing fails."""

    def __init__(
        self,
        message: str,
        component: str = None,
        stage: str = None,
        original_error: Exception = None,
    ):
        details = {
            "stage": stage,
            "original_error": str(original_error) if original_error else None,
        }
        super().__init__(message, component, details)
        self.original_error = original_error


class ValidationError(ComponentError):
    """Raised when input validation fails."""

    def __init__(
        self,
        message: str,
        component: str = None,
        field: str = None,
        value: any = None,
    ):
        details = {"field": field, "value": str(value) if value else None}
        super().__init__(message, component, details)


class ExternalServiceError(ComponentError):
    """Raised when an external service (OpenAI, FAISS) fails."""

    def __init__(
        self,
        message: str,
        component: str = None,
        service: str = None,
        status_code: int = None,
    ):
        details = {"service": service, "status_code": status_code}
        super().__init__(message, component, details)
