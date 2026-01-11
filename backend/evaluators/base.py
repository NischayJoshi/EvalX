"""
Base Domain Evaluator
=====================

Abstract base class defining the interface for all domain-specific evaluators.
Each domain evaluator must implement these methods to ensure consistent
evaluation behavior across different technology domains.

Design Principles:
    - Template Method Pattern: Common workflow, customizable steps
    - Strategy Pattern: Interchangeable evaluation strategies
    - Single Responsibility: Each evaluator handles one domain

Author: EvalX Team
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import logging
import asyncio
import time
from datetime import datetime
import uuid

from evaluators.models import (
    DomainEvaluationResult,
    DomainScore,
    PatternMatch,
    DomainType,
    EvaluationMetadata,
)
from evaluators.exceptions import (
    PatternAnalysisError,
    ScoreCalculationError,
    RepositoryAccessError,
)


logger = logging.getLogger(__name__)


class BaseDomainEvaluator(ABC):
    """
    Abstract base class for domain-specific code evaluators.

    This class provides the foundational structure and common functionality
    for evaluating codebases against domain-specific patterns, best practices,
    and security requirements.

    Subclasses must implement:
        - domain_type: Property returning the DomainType
        - get_file_extensions(): File types to analyze
        - get_patterns(): Domain-specific patterns to detect
        - calculate_domain_score(): Custom scoring logic

    Attributes:
        repo_path: Path to the repository being evaluated
        config: Optional configuration dictionary
        patterns_found: List of detected patterns during evaluation
        files_analyzed: Set of files processed during evaluation

    Example:
        >>> class Web3Evaluator(BaseDomainEvaluator):
        ...     @property
        ...     def domain_type(self) -> DomainType:
        ...         return DomainType.WEB3
        ...
        ...     def get_file_extensions(self) -> List[str]:
        ...         return ['.sol', '.vy', '.rs']
    """

    # Class-level configuration
    DEFAULT_TIMEOUT_SECONDS: int = 300
    MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10MB
    MAX_FILES_TO_ANALYZE: int = 500

    def __init__(
        self, repo_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the domain evaluator.

        Args:
            repo_path: Absolute path to the repository root
            config: Optional configuration overrides
        """
        self.repo_path = Path(repo_path) if repo_path else None
        self.config = config or {}
        self.patterns_found: List[PatternMatch] = []
        self.files_analyzed: Set[str] = set()
        self._evaluation_id: str = ""
        self._start_time: float = 0

        # Apply configuration overrides
        self.timeout_seconds = self.config.get(
            "timeout_seconds", self.DEFAULT_TIMEOUT_SECONDS
        )
        self.max_file_size = self.config.get(
            "max_file_size_bytes", self.MAX_FILE_SIZE_BYTES
        )

        logger.info(
            f"Initialized {self.__class__.__name__} for domain: {self.domain_type.value}"
        )

    # =========================================================================
    # Abstract Properties and Methods (Must be implemented by subclasses)
    # =========================================================================

    @property
    @abstractmethod
    def domain_type(self) -> DomainType:
        """
        Return the domain type this evaluator handles.

        Returns:
            DomainType enum value representing the domain
        """
        pass

    @abstractmethod
    def get_file_extensions(self) -> List[str]:
        """
        Return file extensions relevant to this domain.

        Returns:
            List of file extensions including the dot (e.g., ['.sol', '.vy'])
        """
        pass

    @abstractmethod
    def get_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Return domain-specific patterns to detect.

        Returns:
            Dictionary mapping pattern names to pattern configurations:
            {
                "pattern_name": {
                    "regex": r"pattern_regex",
                    "category": "security|architecture|best_practice",
                    "severity": "info|low|medium|high|critical",
                    "description": "What this pattern indicates",
                    "weight": 1.0  # Scoring weight
                }
            }
        """
        pass

    @abstractmethod
    async def calculate_domain_score(
        self, patterns: List[PatternMatch], file_analysis: Dict[str, Any]
    ) -> DomainScore:
        """
        Calculate the domain-specific score based on patterns and analysis.

        Args:
            patterns: List of detected patterns
            file_analysis: Additional file-level analysis data

        Returns:
            DomainScore with detailed scoring breakdown
        """
        pass

    # =========================================================================
    # Template Method: Main Evaluation Workflow
    # =========================================================================

    async def evaluate(
        self, repo_path: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> DomainEvaluationResult:
        """
        Execute the complete domain evaluation workflow.

        This is the main entry point for evaluation. It orchestrates the
        entire evaluation process using the Template Method pattern.

        Args:
            repo_path: Optional path override for the repository
            metadata: Optional metadata about the submission

        Returns:
            DomainEvaluationResult containing complete evaluation data

        Raises:
            RepositoryAccessError: If repository path is invalid
            PatternAnalysisError: If pattern analysis fails
            ScoreCalculationError: If scoring calculation fails
        """
        # Initialize evaluation session
        self._evaluation_id = str(uuid.uuid4())[:8]
        self._start_time = time.time()

        logger.info(
            f"[{self._evaluation_id}] Starting {self.domain_type.value} evaluation"
        )

        # Resolve repository path
        if repo_path:
            self.repo_path = Path(repo_path)

        if not self.repo_path or not self.repo_path.exists():
            raise RepositoryAccessError(
                str(self.repo_path), "Repository path does not exist"
            )

        try:
            # Step 1: Discover relevant files
            files = await self._discover_files()
            logger.info(f"[{self._evaluation_id}] Found {len(files)} relevant files")

            # Step 2: Analyze files for patterns
            patterns = await self._analyze_patterns(files)
            logger.info(f"[{self._evaluation_id}] Detected {len(patterns)} patterns")

            # Step 3: Perform file-level analysis
            file_analysis = await self._analyze_files(files)

            # Step 4: Calculate domain score
            score = await self.calculate_domain_score(patterns, file_analysis)

            # Step 5: Generate recommendations
            recommendations = await self._generate_recommendations(patterns, score)

            # Step 6: Identify strengths and weaknesses
            strengths, weaknesses = await self._identify_insights(
                patterns, file_analysis
            )

            # Step 7: Build final result
            elapsed_ms = int((time.time() - self._start_time) * 1000)

            evaluation_metadata = EvaluationMetadata(
                evaluation_id=self._evaluation_id,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                duration_ms=elapsed_ms,
                files_analyzed=len(self.files_analyzed),
                patterns_checked=len(self.get_patterns()),
                evaluator_version="1.0.0",
            )

            result = DomainEvaluationResult(
                detected_domain=self.domain_type,
                confidence=self._calculate_confidence(patterns),
                score=score,
                patterns_found=patterns,
                recommendations=recommendations,
                strengths=strengths,
                weaknesses=weaknesses,
                metadata=evaluation_metadata,
                raw_analysis=file_analysis if self.config.get("include_raw") else None,
            )

            logger.info(
                f"[{self._evaluation_id}] Evaluation complete. "
                f"Score: {score.overall_score:.1f}"
            )

            return result

        except Exception as e:
            logger.error(f"[{self._evaluation_id}] Evaluation failed: {str(e)}")
            raise

    # =========================================================================
    # Protected Methods (Can be overridden by subclasses)
    # =========================================================================

    async def _discover_files(self) -> List[Path]:
        """
        Discover files relevant to this domain evaluator.

        Returns:
            List of file paths to analyze
        """
        extensions = set(self.get_file_extensions())
        files: List[Path] = []

        # Common directories to skip
        skip_dirs = {
            "node_modules",
            ".git",
            "__pycache__",
            "venv",
            "env",
            ".env",
            "dist",
            "build",
            ".next",
            "coverage",
            ".pytest_cache",
            ".mypy_cache",
        }

        for file_path in self.repo_path.rglob("*"):
            # Skip directories and hidden files
            if file_path.is_dir():
                continue

            # Check if in a skipped directory
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue

            # Check file extension
            if file_path.suffix.lower() in extensions:
                # Check file size
                try:
                    if file_path.stat().st_size <= self.max_file_size:
                        files.append(file_path)
                except OSError:
                    continue

            # Limit total files
            if len(files) >= self.MAX_FILES_TO_ANALYZE:
                logger.warning(
                    f"[{self._evaluation_id}] File limit reached: {self.MAX_FILES_TO_ANALYZE}"
                )
                break

        return files

    async def _analyze_patterns(self, files: List[Path]) -> List[PatternMatch]:
        """
        Analyze files for domain-specific patterns.

        Args:
            files: List of file paths to analyze

        Returns:
            List of detected PatternMatch objects
        """
        import re

        patterns_config = self.get_patterns()
        detected_patterns: List[PatternMatch] = []

        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                relative_path = str(file_path.relative_to(self.repo_path))
                self.files_analyzed.add(relative_path)

                for pattern_name, pattern_config in patterns_config.items():
                    regex = pattern_config.get("regex", "")
                    if not regex:
                        continue

                    try:
                        matches = list(re.finditer(regex, content, re.MULTILINE))
                        for match in matches:
                            # Calculate line number
                            line_num = content[: match.start()].count("\n") + 1

                            # Extract context (surrounding lines)
                            lines = content.split("\n")
                            start_line = max(0, line_num - 2)
                            end_line = min(len(lines), line_num + 1)
                            context = "\n".join(lines[start_line:end_line])

                            pattern_match = PatternMatch(
                                pattern_name=pattern_name,
                                file_path=relative_path,
                                line_number=line_num,
                                confidence=pattern_config.get("confidence", 0.9),
                                context=context[:200],  # Limit context length
                                category=pattern_config.get("category", "general"),
                                severity=pattern_config.get("severity", "info"),
                            )
                            detected_patterns.append(pattern_match)

                    except re.error as e:
                        logger.warning(f"Invalid regex for pattern {pattern_name}: {e}")
                        continue

            except Exception as e:
                logger.warning(
                    f"[{self._evaluation_id}] Error analyzing {file_path}: {e}"
                )
                continue

        self.patterns_found = detected_patterns
        return detected_patterns

    async def _analyze_files(self, files: List[Path]) -> Dict[str, Any]:
        """
        Perform additional file-level analysis.

        Args:
            files: List of files to analyze

        Returns:
            Dictionary containing analysis results
        """
        analysis = {
            "total_files": len(files),
            "files_by_extension": {},
            "total_lines": 0,
            "avg_file_size": 0,
        }

        total_size = 0

        for file_path in files:
            try:
                ext = file_path.suffix.lower()
                analysis["files_by_extension"][ext] = (
                    analysis["files_by_extension"].get(ext, 0) + 1
                )

                content = file_path.read_text(encoding="utf-8", errors="ignore")
                analysis["total_lines"] += content.count("\n") + 1
                total_size += len(content.encode("utf-8"))

            except Exception:
                continue

        if files:
            analysis["avg_file_size"] = total_size / len(files)

        return analysis

    async def _generate_recommendations(
        self, patterns: List[PatternMatch], score: DomainScore
    ) -> List[str]:
        """
        Generate improvement recommendations based on patterns and scores.

        Args:
            patterns: Detected patterns
            score: Calculated domain score

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Score-based recommendations
        if score.security_score < 70:
            recommendations.append(
                "Improve security practices - review authentication, "
                "input validation, and access control patterns"
            )

        if score.architecture_score < 70:
            recommendations.append(
                "Consider refactoring for better separation of concerns "
                "and modular architecture"
            )

        if score.best_practices_score < 70:
            recommendations.append(
                f"Review {self.domain_type.value} best practices and "
                "industry standards for this domain"
            )

        # Pattern-based recommendations (override in subclasses)
        high_severity_patterns = [
            p for p in patterns if p.severity in ("high", "critical")
        ]
        if high_severity_patterns:
            recommendations.append(
                f"Address {len(high_severity_patterns)} high-severity patterns "
                "identified in the codebase"
            )

        return recommendations

    async def _identify_insights(
        self, patterns: List[PatternMatch], file_analysis: Dict[str, Any]
    ) -> tuple[List[str], List[str]]:
        """
        Identify strengths and weaknesses from the evaluation.

        Args:
            patterns: Detected patterns
            file_analysis: File-level analysis data

        Returns:
            Tuple of (strengths list, weaknesses list)
        """
        strengths = []
        weaknesses = []

        # File structure insights
        if file_analysis.get("total_files", 0) > 10:
            strengths.append("Good project structure with multiple modules")

        # Pattern-based insights
        security_patterns = [p for p in patterns if p.category == "security"]
        if len(security_patterns) > 5:
            strengths.append("Strong security awareness with multiple safeguards")
        elif len(security_patterns) == 0:
            weaknesses.append("Limited security patterns detected")

        architecture_patterns = [p for p in patterns if p.category == "architecture"]
        if architecture_patterns:
            strengths.append("Well-defined architectural patterns")

        return strengths, weaknesses

    def _calculate_confidence(self, patterns: List[PatternMatch]) -> float:
        """
        Calculate confidence score for domain detection.

        Args:
            patterns: Detected patterns

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not patterns:
            return 0.3  # Low confidence if no patterns

        # Average confidence of detected patterns
        avg_confidence = sum(p.confidence for p in patterns) / len(patterns)

        # Bonus for pattern diversity
        unique_categories = len(set(p.category for p in patterns))
        diversity_bonus = min(0.1, unique_categories * 0.02)

        return min(1.0, avg_confidence + diversity_bonus)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def reset(self) -> None:
        """Reset evaluator state for a new evaluation."""
        self.patterns_found = []
        self.files_analyzed = set()
        self._evaluation_id = ""
        self._start_time = 0

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"domain={self.domain_type.value}, "
            f"repo_path={self.repo_path})"
        )
