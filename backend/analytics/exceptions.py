"""
Analytics Custom Exceptions
===========================

Custom exception classes for the EvalX analytics module.
These exceptions provide specific error handling for analytics
operations with meaningful error messages and context.

All exceptions inherit from AnalyticsError for easy catching
of analytics-specific errors.
"""

from typing import Optional, Any, Dict


class AnalyticsError(Exception):
    """
    Base exception for all analytics-related errors.
    
    All other analytics exceptions inherit from this class,
    allowing for catch-all error handling when needed.
    
    Attributes:
        message: Human-readable error description.
        error_code: Machine-readable error identifier.
        context: Additional context data for debugging.
    
    Example:
        try:
            result = await calculate_metrics(event_id)
        except AnalyticsError as e:
            logger.error(f"Analytics failed: {e.message}")
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "ANALYTICS_ERROR",
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the AnalyticsError.
        
        Args:
            message: Human-readable error description.
            error_code: Machine-readable error identifier.
            context: Additional context data for debugging.
        """
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """Return string representation with error code."""
        return f"[{self.error_code}] {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for API responses.
        
        Returns:
            Dictionary with error details.
        """
        return {
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context
        }


class InsufficientDataError(AnalyticsError):
    """
    Raised when there is not enough data to perform analytics.
    
    This exception is raised when analytics operations require
    a minimum amount of data that is not available, such as
    calculating statistics with zero submissions.
    
    Attributes:
        required_count: Minimum data points required.
        actual_count: Actual data points available.
    
    Example:
        if len(submissions) < MIN_SUBMISSIONS:
            raise InsufficientDataError(
                "Need at least 5 submissions for calibration",
                required_count=5,
                actual_count=len(submissions)
            )
    """
    
    def __init__(
        self,
        message: str,
        required_count: int = 0,
        actual_count: int = 0,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize InsufficientDataError.
        
        Args:
            message: Human-readable error description.
            required_count: Minimum data points required.
            actual_count: Actual data points available.
            context: Additional context data for debugging.
        """
        self.required_count = required_count
        self.actual_count = actual_count
        
        # Merge counts into context
        full_context = context or {}
        full_context.update({
            "required_count": required_count,
            "actual_count": actual_count
        })
        
        super().__init__(
            message=message,
            error_code="INSUFFICIENT_DATA",
            context=full_context
        )


class ExportError(AnalyticsError):
    """
    Raised when data export operations fail.
    
    This exception handles errors during CSV/JSON export,
    including invalid column selections, format errors,
    and file generation failures.
    
    Attributes:
        export_format: The requested export format.
        invalid_columns: List of invalid column names requested.
    
    Example:
        if invalid_cols := set(columns) - set(valid_columns):
            raise ExportError(
                "Invalid columns requested",
                export_format="csv",
                invalid_columns=list(invalid_cols)
            )
    """
    
    def __init__(
        self,
        message: str,
        export_format: str = "unknown",
        invalid_columns: Optional[list] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize ExportError.
        
        Args:
            message: Human-readable error description.
            export_format: The requested export format.
            invalid_columns: List of invalid column names.
            context: Additional context data for debugging.
        """
        self.export_format = export_format
        self.invalid_columns = invalid_columns or []
        
        # Merge into context
        full_context = context or {}
        full_context.update({
            "export_format": export_format,
            "invalid_columns": self.invalid_columns
        })
        
        super().__init__(
            message=message,
            error_code="EXPORT_ERROR",
            context=full_context
        )


class ThemeDetectionError(AnalyticsError):
    """
    Raised when theme detection from event description fails.
    
    This exception is raised when the system cannot reliably
    classify an event or submission into a theme category
    based on the available text content.
    
    Attributes:
        event_id: The event that failed classification.
        attempted_text: Text that was analyzed.
    
    Example:
        if not detected_theme:
            raise ThemeDetectionError(
                "Could not detect theme from event description",
                event_id=event_id,
                attempted_text=event.get("description", "")[:100]
            )
    """
    
    def __init__(
        self,
        message: str,
        event_id: Optional[str] = None,
        attempted_text: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize ThemeDetectionError.
        
        Args:
            message: Human-readable error description.
            event_id: The event that failed classification.
            attempted_text: Text that was analyzed (truncated).
            context: Additional context data for debugging.
        """
        self.event_id = event_id
        self.attempted_text = attempted_text
        
        # Merge into context
        full_context = context or {}
        full_context.update({
            "event_id": event_id,
            "attempted_text": (attempted_text[:100] + "...") if attempted_text and len(attempted_text) > 100 else attempted_text
        })
        
        super().__init__(
            message=message,
            error_code="THEME_DETECTION_ERROR",
            context=full_context
        )


class InvalidEventError(AnalyticsError):
    """
    Raised when an event cannot be found or is invalid.
    
    This exception is raised when analytics operations
    reference an event that doesn't exist, is inaccessible
    to the current user, or has invalid data.
    
    Attributes:
        event_id: The invalid event identifier.
        reason: Specific reason for invalidity.
    
    Example:
        if not event:
            raise InvalidEventError(
                "Event not found",
                event_id=event_id,
                reason="not_found"
            )
    """
    
    def __init__(
        self,
        message: str,
        event_id: Optional[str] = None,
        reason: str = "unknown",
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize InvalidEventError.
        
        Args:
            message: Human-readable error description.
            event_id: The invalid event identifier.
            reason: Specific reason for invalidity.
            context: Additional context data for debugging.
        """
        self.event_id = event_id
        self.reason = reason
        
        # Merge into context
        full_context = context or {}
        full_context.update({
            "event_id": event_id,
            "reason": reason
        })
        
        super().__init__(
            message=message,
            error_code="INVALID_EVENT",
            context=full_context
        )


class InvalidTeamError(AnalyticsError):
    """
    Raised when a team cannot be found or user is not a member.
    
    This exception is raised when analytics operations
    reference a team that doesn't exist or the user
    does not have access to.
    
    Attributes:
        team_id: The invalid team identifier.
        event_id: The event context.
        reason: Specific reason for invalidity.
    
    Example:
        if user_id not in team_member_ids:
            raise InvalidTeamError(
                "User is not a member of this team",
                team_id=team_id,
                event_id=event_id,
                reason="not_member"
            )
    """
    
    def __init__(
        self,
        message: str,
        team_id: Optional[str] = None,
        event_id: Optional[str] = None,
        reason: str = "unknown",
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize InvalidTeamError.
        
        Args:
            message: Human-readable error description.
            team_id: The invalid team identifier.
            event_id: The event context.
            reason: Specific reason for invalidity.
            context: Additional context data for debugging.
        """
        self.team_id = team_id
        self.event_id = event_id
        self.reason = reason
        
        # Merge into context
        full_context = context or {}
        full_context.update({
            "team_id": team_id,
            "event_id": event_id,
            "reason": reason
        })
        
        super().__init__(
            message=message,
            error_code="INVALID_TEAM",
            context=full_context
        )


class CalculationError(AnalyticsError):
    """
    Raised when a mathematical calculation fails.
    
    This exception handles division by zero, invalid
    statistical operations, and numerical computation errors.
    
    Attributes:
        operation: The operation that failed.
        input_values: Values that caused the failure.
    
    Example:
        if std_dev == 0:
            raise CalculationError(
                "Cannot calculate z-score with zero standard deviation",
                operation="z_score",
                input_values={"std_dev": std_dev}
            )
    """
    
    def __init__(
        self,
        message: str,
        operation: str = "unknown",
        input_values: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize CalculationError.
        
        Args:
            message: Human-readable error description.
            operation: The operation that failed.
            input_values: Values that caused the failure.
            context: Additional context data for debugging.
        """
        self.operation = operation
        self.input_values = input_values or {}
        
        # Merge into context
        full_context = context or {}
        full_context.update({
            "operation": operation,
            "input_values": self.input_values
        })
        
        super().__init__(
            message=message,
            error_code="CALCULATION_ERROR",
            context=full_context
        )


class AuthorizationError(AnalyticsError):
    """
    Raised when user lacks permission for analytics operation.
    
    This exception is raised when a user attempts to access
    analytics they don't have permission to view.
    
    Attributes:
        user_id: The user attempting access.
        resource_type: Type of resource being accessed.
        resource_id: ID of the resource.
    
    Example:
        if event.organizerId != user_id:
            raise AuthorizationError(
                "Only event organizer can view calibration metrics",
                user_id=user_id,
                resource_type="event",
                resource_id=event_id
            )
    """
    
    def __init__(
        self,
        message: str,
        user_id: Optional[str] = None,
        resource_type: str = "unknown",
        resource_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize AuthorizationError.
        
        Args:
            message: Human-readable error description.
            user_id: The user attempting access.
            resource_type: Type of resource being accessed.
            resource_id: ID of the resource.
            context: Additional context data for debugging.
        """
        self.user_id = user_id
        self.resource_type = resource_type
        self.resource_id = resource_id
        
        # Merge into context
        full_context = context or {}
        full_context.update({
            "user_id": user_id,
            "resource_type": resource_type,
            "resource_id": resource_id
        })
        
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            context=full_context
        )
