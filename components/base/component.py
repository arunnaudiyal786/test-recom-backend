"""
Base component abstract class.

All service components inherit from this class to ensure
consistent interface across the system.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Dict, Any
from pydantic import BaseModel

# Generic type variables for request/response
TRequest = TypeVar("TRequest", bound=BaseModel)
TResponse = TypeVar("TResponse", bound=BaseModel)


class BaseComponent(ABC, Generic[TRequest, TResponse]):
    """
    Abstract base class for all service components.

    Each component must implement:
    - process(): Main processing logic
    - health_check(): Health status check
    - component_name: Unique identifier

    Usage:
        class MyService(BaseComponent[MyRequest, MyResponse]):
            async def process(self, request: MyRequest) -> MyResponse:
                # Implementation
                pass
    """

    @abstractmethod
    async def process(self, request: TRequest) -> TResponse:
        """
        Process a request and return a response.

        This is the main entry point for the component.
        All business logic should be implemented here.

        Args:
            request: Pydantic model containing input data

        Returns:
            Pydantic model containing output data

        Raises:
            ProcessingError: If processing fails
            ValidationError: If input validation fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check component health status.

        Returns:
            Dict with at least:
            - status: "healthy" | "unhealthy" | "degraded"
            - component: Component name
            - details: Optional additional info
        """
        pass

    @property
    @abstractmethod
    def component_name(self) -> str:
        """
        Return unique component identifier.

        This is used for logging, metrics, and error reporting.

        Returns:
            String identifier (e.g., "embedding", "retrieval")
        """
        pass

    async def __call__(self, request: TRequest) -> TResponse:
        """
        Allow component to be called directly.

        Makes the component callable: response = await component(request)
        """
        return await self.process(request)
