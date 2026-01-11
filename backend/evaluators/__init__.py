"""
Domain-Specific Evaluators Module
================================

A comprehensive evaluation system for analyzing codebases against domain-specific
patterns, best practices, and industry standards.

Supported Domains:
    - Web3/Blockchain: Smart contracts, DeFi protocols, NFT marketplaces
    - ML/AI: Machine learning pipelines, model training, inference systems
    - Fintech: Payment processing, banking systems, compliance frameworks
    - IoT: Embedded systems, sensor networks, device management
    - AR/VR: Immersive experiences, 3D rendering, spatial computing

Usage:
    from evaluators import DomainOrchestrator
    
    orchestrator = DomainOrchestrator()
    results = await orchestrator.evaluate_repository(repo_path, metadata)

Architecture:
    - BaseDomainEvaluator: Abstract base class for all domain evaluators
    - DomainRegistry: Maps domains to their respective evaluator classes
    - DomainOrchestrator: Auto-detects domain and orchestrates evaluation
    - Individual Evaluators: Specialized analyzers for each domain

Author: EvalX Team
Version: 1.0.0
"""

from evaluators.base import BaseDomainEvaluator
from evaluators.orchestrator import DomainOrchestrator
from evaluators.registry import DomainRegistry
from evaluators.models import (
    DomainEvaluationResult,
    DomainScore,
    PatternMatch,
    DomainType,
)
from evaluators.exceptions import (
    DomainEvaluationError,
    UnsupportedDomainError,
    PatternAnalysisError,
)

__all__ = [
    # Core Classes
    "BaseDomainEvaluator",
    "DomainOrchestrator",
    "DomainRegistry",
    # Models
    "DomainEvaluationResult",
    "DomainScore",
    "PatternMatch",
    "DomainType",
    # Exceptions
    "DomainEvaluationError",
    "UnsupportedDomainError",
    "PatternAnalysisError",
]

__version__ = "1.0.0"
__author__ = "EvalX Team"
