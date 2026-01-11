"""
Domain Evaluation Exceptions
============================

Custom exception classes for domain-specific evaluation error handling.
These exceptions provide granular error categorization for better debugging
and error recovery.

Exception Hierarchy:
    DomainEvaluationError (base)
    ├── UnsupportedDomainError
    ├── PatternAnalysisError
    ├── DomainDetectionError
    ├── EvaluatorConfigError
    └── ScoreCalculationError

Author: EvalX Team
"""

from typing import Optional, Dict, Any


class DomainEvaluationError(Exception):
    """
    Base exception for all domain evaluation errors.
    
    Attributes:
        message: Human-readable error description
        error_code: Machine-readable error identifier
        context: Additional contextual information
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: str = "DOMAIN_EVAL_ERROR",
        context: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.error_code,
            "message": self.message,
            "context": self.context
        }
    
    def __str__(self) -> str:
        if self.context:
            return f"[{self.error_code}] {self.message} | Context: {self.context}"
        return f"[{self.error_code}] {self.message}"


class UnsupportedDomainError(DomainEvaluationError):
    """
    Raised when attempting to evaluate an unsupported or unknown domain.
    
    Example:
        >>> raise UnsupportedDomainError("quantum_computing")
        UnsupportedDomainError: Domain 'quantum_computing' is not supported
    """
    
    def __init__(self, domain: str, supported_domains: Optional[list] = None):
        self.domain = domain
        self.supported_domains = supported_domains or [
            "web3", "ml_ai", "fintech", "iot", "ar_vr"
        ]
        message = (
            f"Domain '{domain}' is not supported. "
            f"Supported domains: {', '.join(self.supported_domains)}"
        )
        super().__init__(
            message=message,
            error_code="UNSUPPORTED_DOMAIN",
            context={"domain": domain, "supported": self.supported_domains}
        )


class PatternAnalysisError(DomainEvaluationError):
    """
    Raised when pattern analysis fails for a specific file or pattern.
    
    Example:
        >>> raise PatternAnalysisError("contracts/Token.sol", "reentrancy_check")
        PatternAnalysisError: Failed to analyze pattern 'reentrancy_check' in 'contracts/Token.sol'
    """
    
    def __init__(
        self, 
        file_path: str, 
        pattern_name: str, 
        reason: Optional[str] = None
    ):
        self.file_path = file_path
        self.pattern_name = pattern_name
        self.reason = reason
        
        message = f"Failed to analyze pattern '{pattern_name}' in '{file_path}'"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            error_code="PATTERN_ANALYSIS_ERROR",
            context={
                "file_path": file_path,
                "pattern_name": pattern_name,
                "reason": reason
            }
        )


class DomainDetectionError(DomainEvaluationError):
    """
    Raised when domain auto-detection fails or produces ambiguous results.
    
    Example:
        >>> raise DomainDetectionError("Multiple domains detected with equal confidence")
    """
    
    def __init__(
        self, 
        message: str,
        detected_domains: Optional[Dict[str, float]] = None
    ):
        self.detected_domains = detected_domains
        super().__init__(
            message=message,
            error_code="DOMAIN_DETECTION_ERROR",
            context={"detected_domains": detected_domains}
        )


class EvaluatorConfigError(DomainEvaluationError):
    """
    Raised when evaluator configuration is invalid or incomplete.
    
    Example:
        >>> raise EvaluatorConfigError("Missing required config key", "api_key")
    """
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        self.config_key = config_key
        super().__init__(
            message=message,
            error_code="EVALUATOR_CONFIG_ERROR",
            context={"config_key": config_key}
        )


class ScoreCalculationError(DomainEvaluationError):
    """
    Raised when score calculation fails due to invalid data or computation errors.
    
    Example:
        >>> raise ScoreCalculationError("Division by zero in metric calculation", "security_score")
    """
    
    def __init__(self, message: str, metric_name: Optional[str] = None):
        self.metric_name = metric_name
        super().__init__(
            message=message,
            error_code="SCORE_CALCULATION_ERROR",
            context={"metric_name": metric_name}
        )


class RepositoryAccessError(DomainEvaluationError):
    """
    Raised when repository access fails (path not found, permissions, etc.).
    
    Example:
        >>> raise RepositoryAccessError("/path/to/repo", "Directory not found")
    """
    
    def __init__(self, repo_path: str, reason: str):
        self.repo_path = repo_path
        self.reason = reason
        super().__init__(
            message=f"Cannot access repository at '{repo_path}': {reason}",
            error_code="REPOSITORY_ACCESS_ERROR",
            context={"repo_path": repo_path, "reason": reason}
        )


class EvaluationTimeoutError(DomainEvaluationError):
    """
    Raised when evaluation exceeds the configured timeout threshold.
    
    Example:
        >>> raise EvaluationTimeoutError(300, 500)
        EvaluationTimeoutError: Evaluation timed out after 300s (limit: 500s)
    """
    
    def __init__(self, elapsed_seconds: float, timeout_seconds: float):
        self.elapsed_seconds = elapsed_seconds
        self.timeout_seconds = timeout_seconds
        super().__init__(
            message=f"Evaluation timed out after {elapsed_seconds:.1f}s (limit: {timeout_seconds}s)",
            error_code="EVALUATION_TIMEOUT",
            context={
                "elapsed_seconds": elapsed_seconds,
                "timeout_seconds": timeout_seconds
            }
        )
