"""
Domain Evaluation Orchestrator
==============================

High-level orchestration layer for domain-specific code evaluation.
Handles domain auto-detection, evaluator selection, and result aggregation.

Features:
    - Automatic domain detection from repository contents
    - Multi-domain evaluation support
    - Result aggregation and normalization
    - Integration with existing evaluation pipeline

Usage:
    from evaluators.orchestrator import DomainOrchestrator
    
    orchestrator = DomainOrchestrator()
    result = await orchestrator.evaluate_repository(
        repo_path="/path/to/repo",
        metadata={"submission_id": "123"}
    )

Author: EvalX Team
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import asyncio
from datetime import datetime
import uuid

from evaluators.base import BaseDomainEvaluator
from evaluators.registry import DomainRegistry, get_registry
from evaluators.models import (
    DomainType,
    DomainEvaluationResult,
    DomainScore,
    EvaluationMetadata,
)
from evaluators.exceptions import (
    DomainEvaluationError,
    DomainDetectionError,
    RepositoryAccessError,
)
from evaluators.detector_utils import (
    detect_domain_from_files,
    get_secondary_domains,
)


logger = logging.getLogger(__name__)


class DomainOrchestrator:
    """
    Orchestrator for domain-specific code evaluation.
    
    This class provides high-level coordination for evaluating repositories
    against domain-specific patterns and best practices. It handles domain
    detection, evaluator selection, parallel evaluation, and result aggregation.
    
    Features:
        - Automatic domain detection based on file analysis
        - Manual domain override support
        - Multi-domain evaluation for hybrid projects
        - Configurable evaluation behavior
        - Integration-ready result format
    
    Attributes:
        registry: DomainRegistry instance for evaluator access
        config: Configuration dictionary for evaluation settings
    
    Example:
        >>> orchestrator = DomainOrchestrator()
        >>> 
        >>> # Auto-detect domain
        >>> result = await orchestrator.evaluate_repository("/path/to/repo")
        >>> 
        >>> # Specify domain explicitly
        >>> result = await orchestrator.evaluate_repository(
        ...     "/path/to/repo",
        ...     domain=DomainType.WEB3
        ... )
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        "enable_auto_detection": True,
        "evaluate_secondary_domains": False,
        "detection_confidence_threshold": 0.5,
        "include_raw_analysis": False,
        "max_evaluation_time_seconds": 300,
        "parallel_evaluation": True,
    }
    
    def __init__(
        self,
        registry: Optional[DomainRegistry] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            registry: Optional custom DomainRegistry instance
            config: Optional configuration overrides
        """
        self.registry = registry or get_registry()
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        
        logger.info(
            f"DomainOrchestrator initialized with "
            f"{len(self.registry)} registered evaluators"
        )
    
    async def evaluate_repository(
        self,
        repo_path: str,
        domain: Optional[DomainType] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DomainEvaluationResult:
        """
        Evaluate a repository against domain-specific patterns.
        
        This is the main entry point for domain evaluation. It handles:
        1. Domain detection (if not specified)
        2. Evaluator instantiation
        3. Pattern analysis and scoring
        4. Result generation
        
        Args:
            repo_path: Absolute path to the repository root
            domain: Optional explicit domain type (skips auto-detection)
            metadata: Optional metadata about the submission
        
        Returns:
            DomainEvaluationResult containing complete evaluation data
        
        Raises:
            RepositoryAccessError: If repository path is invalid
            DomainDetectionError: If auto-detection fails
            DomainEvaluationError: If evaluation fails
        
        Example:
            >>> result = await orchestrator.evaluate_repository(
            ...     repo_path="/projects/my-defi-app",
            ...     metadata={"submission_id": "abc123", "team": "alpha"}
            ... )
            >>> print(f"Domain: {result.detected_domain.value}")
            >>> print(f"Score: {result.score.overall_score}")
        """
        evaluation_id = str(uuid.uuid4())[:8]
        start_time = datetime.utcnow()
        
        logger.info(f"[{evaluation_id}] Starting repository evaluation: {repo_path}")
        
        # Validate repository path
        repo = Path(repo_path)
        if not repo.exists():
            raise RepositoryAccessError(repo_path, "Path does not exist")
        if not repo.is_dir():
            raise RepositoryAccessError(repo_path, "Path is not a directory")
        
        # Step 1: Determine domain
        detected_domain, detection_confidence = await self._detect_or_use_domain(
            repo_path,
            domain,
            evaluation_id
        )
        
        # Step 2: Get appropriate evaluator
        evaluator = self.registry.get_evaluator(
            detected_domain,
            repo_path=repo_path,
            config={"include_raw": self.config.get("include_raw_analysis", False)}
        )
        
        # Step 3: Run evaluation
        logger.info(
            f"[{evaluation_id}] Running {detected_domain.value} evaluation"
        )
        
        try:
            result = await evaluator.evaluate(repo_path=repo_path, metadata=metadata)
        except Exception as e:
            logger.error(f"[{evaluation_id}] Evaluation failed: {str(e)}")
            raise DomainEvaluationError(
                f"Evaluation failed for {detected_domain.value}: {str(e)}",
                context={"repo_path": repo_path, "domain": detected_domain.value}
            )
        
        # Step 4: Enrich result with orchestrator metadata
        result.confidence = detection_confidence
        
        # Handle secondary domains if enabled
        if self.config.get("evaluate_secondary_domains", False):
            secondary = get_secondary_domains(
                repo_path,
                detected_domain,
                threshold=0.3
            )
            result.secondary_domains = secondary
        
        # Update metadata
        if result.metadata:
            result.metadata.evaluation_id = evaluation_id
            result.metadata.completed_at = datetime.utcnow()
            duration_ms = int(
                (result.metadata.completed_at - start_time).total_seconds() * 1000
            )
            result.metadata.duration_ms = duration_ms
        
        logger.info(
            f"[{evaluation_id}] Evaluation complete. "
            f"Domain: {detected_domain.value}, "
            f"Score: {result.score.overall_score:.1f}"
        )
        
        return result
    
    async def evaluate_multiple_domains(
        self,
        repo_path: str,
        domains: List[DomainType],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[DomainType, DomainEvaluationResult]:
        """
        Evaluate a repository against multiple domains.
        
        Useful for hybrid projects that span multiple technology domains.
        
        Args:
            repo_path: Path to the repository
            domains: List of domains to evaluate against
            metadata: Optional submission metadata
        
        Returns:
            Dictionary mapping DomainType to evaluation results
        
        Example:
            >>> results = await orchestrator.evaluate_multiple_domains(
            ...     "/path/to/repo",
            ...     domains=[DomainType.ML_AI, DomainType.FINTECH]
            ... )
            >>> for domain, result in results.items():
            ...     print(f"{domain.value}: {result.score.overall_score}")
        """
        results: Dict[DomainType, DomainEvaluationResult] = {}
        
        if self.config.get("parallel_evaluation", True):
            # Run evaluations in parallel
            tasks = [
                self.evaluate_repository(
                    repo_path, 
                    domain=domain, 
                    metadata=metadata
                )
                for domain in domains
            ]
            
            completed = await asyncio.gather(*tasks, return_exceptions=True)
            
            for domain, result in zip(domains, completed):
                if isinstance(result, Exception):
                    logger.error(
                        f"Evaluation failed for {domain.value}: {str(result)}"
                    )
                    continue
                results[domain] = result
        else:
            # Run sequentially
            for domain in domains:
                try:
                    result = await self.evaluate_repository(
                        repo_path,
                        domain=domain,
                        metadata=metadata
                    )
                    results[domain] = result
                except Exception as e:
                    logger.error(f"Evaluation failed for {domain.value}: {str(e)}")
                    continue
        
        return results
    
    async def detect_domain(
        self,
        repo_path: str
    ) -> Tuple[DomainType, float]:
        """
        Detect the primary domain of a repository.
        
        Uses multiple signals including file extensions, dependency files,
        README content, and code pattern analysis.
        
        Args:
            repo_path: Path to the repository
        
        Returns:
            Tuple of (detected DomainType, confidence score)
        
        Raises:
            DomainDetectionError: If detection fails or confidence too low
        """
        domain, confidence = detect_domain_from_files(repo_path)
        
        threshold = self.config.get("detection_confidence_threshold", 0.5)
        
        if domain == DomainType.UNKNOWN or confidence < threshold:
            raise DomainDetectionError(
                f"Domain detection confidence ({confidence:.2%}) below threshold ({threshold:.0%})",
                detected_domains={domain.value: confidence}
            )
        
        return domain, confidence
    
    async def _detect_or_use_domain(
        self,
        repo_path: str,
        explicit_domain: Optional[DomainType],
        evaluation_id: str
    ) -> Tuple[DomainType, float]:
        """
        Get domain either from explicit specification or auto-detection.
        
        Args:
            repo_path: Repository path for detection
            explicit_domain: Optional explicitly specified domain
            evaluation_id: Evaluation ID for logging
        
        Returns:
            Tuple of (domain type, confidence)
        """
        if explicit_domain is not None:
            # User specified domain explicitly
            logger.info(
                f"[{evaluation_id}] Using explicit domain: {explicit_domain.value}"
            )
            return explicit_domain, 1.0
        
        if not self.config.get("enable_auto_detection", True):
            raise DomainDetectionError(
                "Auto-detection disabled and no domain specified"
            )
        
        # Auto-detect domain
        logger.info(f"[{evaluation_id}] Auto-detecting domain")
        
        domain, confidence = detect_domain_from_files(repo_path)
        
        threshold = self.config.get("detection_confidence_threshold", 0.5)
        
        if domain == DomainType.UNKNOWN:
            logger.warning(
                f"[{evaluation_id}] Could not detect domain. "
                "Defaulting to generic evaluation."
            )
            # Try to find any matching domain above a lower threshold
            # For now, return UNKNOWN with low confidence
            return DomainType.UNKNOWN, 0.0
        
        if confidence < threshold:
            logger.warning(
                f"[{evaluation_id}] Low detection confidence: {confidence:.2%}"
            )
        
        logger.info(
            f"[{evaluation_id}] Detected domain: {domain.value} "
            f"(confidence: {confidence:.2%})"
        )
        
        return domain, confidence
    
    def get_supported_domains(self) -> List[DomainType]:
        """Get list of supported domain types."""
        return self.registry.get_supported_domains()
    
    def format_result_for_storage(
        self,
        result: DomainEvaluationResult
    ) -> Dict[str, Any]:
        """
        Format evaluation result for database storage.
        
        Converts the DomainEvaluationResult to a dictionary format
        suitable for storing in MongoDB as part of submission evaluation.
        
        Args:
            result: The evaluation result to format
        
        Returns:
            Dictionary ready for database storage
        
        Example:
            >>> result = await orchestrator.evaluate_repository("/path/to/repo")
            >>> storage_data = orchestrator.format_result_for_storage(result)
            >>> # Store in submissions collection under 'evaluation.domain_evaluation'
        """
        return {
            "domain_evaluation": {
                "detected_domain": result.detected_domain.value,
                "secondary_domains": [d.value for d in result.secondary_domains],
                "confidence": result.confidence,
                "score": {
                    "overall": result.score.overall_score,
                    "architecture": result.score.architecture_score,
                    "security": result.score.security_score,
                    "best_practices": result.score.best_practices_score,
                    "innovation": result.score.innovation_score,
                    "completeness": result.score.completeness_score,
                    "grade": result.score.calculate_grade(),
                    "breakdown": result.score.breakdown,
                },
                "patterns_found_count": len(result.patterns_found),
                "patterns_summary": [
                    {
                        "name": p.pattern_name,
                        "category": p.category,
                        "file": p.file_path,
                    }
                    for p in result.patterns_found[:20]  # Limit stored patterns
                ],
                "recommendations": result.recommendations[:5],
                "strengths": result.strengths[:5],
                "weaknesses": result.weaknesses[:5],
                "metadata": {
                    "evaluation_id": result.metadata.evaluation_id if result.metadata else None,
                    "duration_ms": result.metadata.duration_ms if result.metadata else None,
                    "files_analyzed": result.metadata.files_analyzed if result.metadata else None,
                    "evaluated_at": datetime.utcnow().isoformat(),
                },
            }
        }
    
    def __repr__(self) -> str:
        domains = [d.value for d in self.get_supported_domains()]
        return f"DomainOrchestrator(domains={domains})"
