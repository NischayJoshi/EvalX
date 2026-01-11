"""
Domain Evaluators Unit Tests
============================

Comprehensive test suite for domain-specific evaluator modules.
Tests cover models, base classes, individual evaluators, registry,
and orchestrator functionality.

Run tests with:
    pytest tests/test_domain_evaluators.py -v
    pytest tests/test_domain_evaluators.py -v --cov=evaluators

Author: EvalX Team
"""

import pytest
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch, AsyncMock
import tempfile
import os

# Import evaluator modules
from evaluators.models import (
    DomainType,
    PatternMatch,
    DomainScore,
    DomainEvaluationResult,
    EvaluationMetadata,
)
from evaluators.exceptions import (
    DomainEvaluationError,
    UnsupportedDomainError,
    PatternAnalysisError,
    RepositoryAccessError,
)
from evaluators.base import BaseDomainEvaluator
from evaluators.registry import DomainRegistry, get_registry
from evaluators.orchestrator import DomainOrchestrator
from evaluators.detector_utils import (
    detect_domain_from_files,
    calculate_weighted_score,
    normalize_score,
)
from evaluators.constants import (
    WEB3_PATTERNS,
    ML_AI_PATTERNS,
    FINTECH_PATTERNS,
    IOT_PATTERNS,
    AR_VR_PATTERNS,
)

# Import domain evaluators
from evaluators.domain.web3_evaluator import Web3Evaluator
from evaluators.domain.ml_evaluator import MLEvaluator
from evaluators.domain.fintech_evaluator import FintechEvaluator
from evaluators.domain.iot_evaluator import IoTEvaluator
from evaluators.domain.arvr_evaluator import ARVREvaluator


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_patterns() -> List[PatternMatch]:
    """Create sample pattern matches for testing."""
    return [
        PatternMatch(
            pattern_name="reentrancy_guard",
            file_path="contracts/Token.sol",
            line_number=42,
            confidence=0.95,
            context="modifier nonReentrant() {",
            category="security",
            severity="high",
        ),
        PatternMatch(
            pattern_name="access_control",
            file_path="contracts/Token.sol",
            line_number=15,
            confidence=0.92,
            context="modifier onlyOwner() {",
            category="security",
            severity="high",
        ),
        PatternMatch(
            pattern_name="erc20_implementation",
            file_path="contracts/Token.sol",
            line_number=1,
            confidence=0.98,
            context="contract Token is ERC20 {",
            category="architecture",
            severity="info",
        ),
    ]


@pytest.fixture
def sample_file_analysis() -> Dict[str, Any]:
    """Create sample file analysis data."""
    return {
        "total_files": 15,
        "files_by_extension": {
            ".sol": 5,
            ".ts": 8,
            ".json": 2,
        },
        "total_lines": 2500,
        "avg_file_size": 5000,
    }


@pytest.fixture
def temp_repo_dir():
    """Create a temporary repository directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some sample files
        contracts_dir = Path(tmpdir) / "contracts"
        contracts_dir.mkdir()

        # Create a sample Solidity file
        sol_file = contracts_dir / "Token.sol"
        sol_file.write_text("""
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token is ERC20, ReentrancyGuard, Ownable {
    modifier nonReentrant() {
        // reentrancy guard implementation
    }
    
    function transfer(address to, uint256 amount) public override nonReentrant {
        super.transfer(to, amount);
        emit Transfer(msg.sender, to, amount);
    }
}
        """)

        # Create a README
        readme = Path(tmpdir) / "README.md"
        readme.write_text("# DeFi Token\n\nA blockchain smart contract project.")

        # Create package.json
        package_json = Path(tmpdir) / "package.json"
        package_json.write_text("""
{
    "dependencies": {
        "ethers": "^5.0.0",
        "hardhat": "^2.0.0"
    }
}
        """)

        yield tmpdir


# =============================================================================
# Model Tests
# =============================================================================


class TestDomainType:
    """Tests for DomainType enum."""

    def test_domain_type_values(self):
        """Test that all domain types have correct values."""
        assert DomainType.WEB3.value == "web3"
        assert DomainType.ML_AI.value == "ml_ai"
        assert DomainType.FINTECH.value == "fintech"
        assert DomainType.IOT.value == "iot"
        assert DomainType.AR_VR.value == "ar_vr"
        assert DomainType.UNKNOWN.value == "unknown"

    def test_from_string_valid(self):
        """Test DomainType.from_string with valid inputs."""
        assert DomainType.from_string("web3") == DomainType.WEB3
        assert DomainType.from_string("blockchain") == DomainType.WEB3
        assert DomainType.from_string("ml") == DomainType.ML_AI
        assert DomainType.from_string("ai") == DomainType.ML_AI
        assert DomainType.from_string("fintech") == DomainType.FINTECH
        assert DomainType.from_string("iot") == DomainType.IOT
        assert DomainType.from_string("ar") == DomainType.AR_VR
        assert DomainType.from_string("vr") == DomainType.AR_VR

    def test_from_string_unknown(self):
        """Test DomainType.from_string with unknown input."""
        assert DomainType.from_string("invalid") == DomainType.UNKNOWN
        assert DomainType.from_string("random") == DomainType.UNKNOWN


class TestPatternMatch:
    """Tests for PatternMatch model."""

    def test_pattern_match_creation(self):
        """Test creating a PatternMatch instance."""
        pattern = PatternMatch(
            pattern_name="test_pattern",
            file_path="src/test.py",
            line_number=10,
            confidence=0.9,
            category="security",
            severity="high",
        )

        assert pattern.pattern_name == "test_pattern"
        assert pattern.file_path == "src/test.py"
        assert pattern.line_number == 10
        assert pattern.confidence == 0.9
        assert pattern.category == "security"
        assert pattern.severity == "high"

    def test_pattern_match_defaults(self):
        """Test PatternMatch default values."""
        pattern = PatternMatch(pattern_name="test", file_path="test.py")

        assert pattern.confidence == 1.0
        assert pattern.category == "general"
        assert pattern.severity == "info"
        assert pattern.line_number is None
        assert pattern.context is None


class TestDomainScore:
    """Tests for DomainScore model."""

    def test_domain_score_creation(self):
        """Test creating a DomainScore instance."""
        score = DomainScore(
            domain=DomainType.WEB3,
            overall_score=85.5,
            architecture_score=80.0,
            security_score=90.0,
            best_practices_score=85.0,
            innovation_score=82.0,
            completeness_score=78.0,
        )

        assert score.domain == DomainType.WEB3
        assert score.overall_score == 85.5
        assert score.security_score == 90.0

    def test_calculate_grade(self):
        """Test grade calculation from scores."""
        # A+ grade (90+)
        score = DomainScore(domain=DomainType.WEB3, overall_score=95)
        assert score.calculate_grade() == "A+"

        # A grade (85-89)
        score = DomainScore(domain=DomainType.WEB3, overall_score=87)
        assert score.calculate_grade() == "A"

        # B+ grade (75-79)
        score = DomainScore(domain=DomainType.WEB3, overall_score=77)
        assert score.calculate_grade() == "B+"

        # C grade (55-59)
        score = DomainScore(domain=DomainType.WEB3, overall_score=57)
        assert score.calculate_grade() == "C"

        # F grade (<40)
        score = DomainScore(domain=DomainType.WEB3, overall_score=35)
        assert score.calculate_grade() == "F"


class TestDomainEvaluationResult:
    """Tests for DomainEvaluationResult model."""

    def test_to_markdown_report(self, sample_patterns):
        """Test generating markdown report."""
        score = DomainScore(
            domain=DomainType.WEB3,
            overall_score=78.5,
            architecture_score=80.0,
            security_score=75.0,
            best_practices_score=80.0,
            innovation_score=70.0,
            completeness_score=85.0,
        )

        result = DomainEvaluationResult(
            detected_domain=DomainType.WEB3,
            confidence=0.92,
            score=score,
            patterns_found=sample_patterns,
            recommendations=["Add more tests", "Improve security"],
            strengths=["Good architecture"],
            weaknesses=["Needs more documentation"],
        )

        report = result.to_markdown_report()

        assert "# Domain Evaluation Report: WEB3" in report
        assert "78.5/100" in report
        assert "B+" in report
        assert "Architecture: 80.0/100" in report
        assert "Good architecture" in report
        assert "Needs more documentation" in report


# =============================================================================
# Exception Tests
# =============================================================================


class TestExceptions:
    """Tests for custom exceptions."""

    def test_unsupported_domain_error(self):
        """Test UnsupportedDomainError creation."""
        error = UnsupportedDomainError("quantum")

        assert "quantum" in str(error)
        assert error.domain == "quantum"
        assert "web3" in error.supported_domains

        error_dict = error.to_dict()
        assert error_dict["error"] == "UNSUPPORTED_DOMAIN"

    def test_pattern_analysis_error(self):
        """Test PatternAnalysisError creation."""
        error = PatternAnalysisError("test.py", "security_check", "File not readable")

        assert "test.py" in str(error)
        assert "security_check" in str(error)
        assert error.file_path == "test.py"
        assert error.pattern_name == "security_check"

    def test_repository_access_error(self):
        """Test RepositoryAccessError creation."""
        error = RepositoryAccessError("/invalid/path", "Directory not found")

        assert "/invalid/path" in str(error)
        assert error.repo_path == "/invalid/path"


# =============================================================================
# Registry Tests
# =============================================================================


class TestDomainRegistry:
    """Tests for DomainRegistry."""

    def test_registry_initialization(self):
        """Test registry initializes with default evaluators."""
        registry = DomainRegistry()

        assert len(registry) == 5
        assert DomainType.WEB3 in registry
        assert DomainType.ML_AI in registry
        assert DomainType.FINTECH in registry
        assert DomainType.IOT in registry
        assert DomainType.AR_VR in registry

    def test_get_evaluator(self):
        """Test getting evaluator instances."""
        registry = DomainRegistry()

        evaluator = registry.get_evaluator(DomainType.WEB3)

        assert isinstance(evaluator, Web3Evaluator)
        assert evaluator.domain_type == DomainType.WEB3

    def test_get_evaluator_with_path(self, temp_repo_dir):
        """Test getting evaluator with repo path."""
        registry = DomainRegistry()

        evaluator = registry.get_evaluator(DomainType.WEB3, repo_path=temp_repo_dir)

        assert evaluator.repo_path == Path(temp_repo_dir)

    def test_unsupported_domain_raises(self):
        """Test that unsupported domain raises exception."""
        registry = DomainRegistry()

        with pytest.raises(UnsupportedDomainError):
            registry.get_evaluator(DomainType.UNKNOWN)

    def test_get_supported_domains(self):
        """Test getting list of supported domains."""
        registry = DomainRegistry()
        supported = registry.get_supported_domains()

        assert DomainType.WEB3 in supported
        assert DomainType.ML_AI in supported
        assert len(supported) == 5

    def test_is_domain_supported(self):
        """Test checking if domain is supported."""
        registry = DomainRegistry()

        assert registry.is_domain_supported(DomainType.WEB3) is True
        assert registry.is_domain_supported(DomainType.UNKNOWN) is False

    def test_get_evaluator_info(self):
        """Test getting evaluator information."""
        registry = DomainRegistry()
        info = registry.get_evaluator_info(DomainType.WEB3)

        assert info["domain"] == "web3"
        assert info["class_name"] == "Web3Evaluator"
        assert ".sol" in info["file_extensions"]


# =============================================================================
# Individual Evaluator Tests
# =============================================================================


class TestWeb3Evaluator:
    """Tests for Web3Evaluator."""

    def test_domain_type(self):
        """Test Web3Evaluator domain type."""
        evaluator = Web3Evaluator()
        assert evaluator.domain_type == DomainType.WEB3

    def test_file_extensions(self):
        """Test Web3Evaluator file extensions."""
        evaluator = Web3Evaluator()
        extensions = evaluator.get_file_extensions()

        assert ".sol" in extensions
        assert ".vy" in extensions
        assert ".rs" in extensions

    def test_patterns(self):
        """Test Web3Evaluator patterns."""
        evaluator = Web3Evaluator()
        patterns = evaluator.get_patterns()

        assert "reentrancy_guard" in patterns
        assert "access_control" in patterns
        assert "erc20_implementation" in patterns

    @pytest.mark.asyncio
    async def test_calculate_score(self, sample_patterns, sample_file_analysis):
        """Test Web3 score calculation."""
        evaluator = Web3Evaluator()

        score = await evaluator.calculate_domain_score(
            sample_patterns, sample_file_analysis
        )

        assert score.domain == DomainType.WEB3
        assert 0 <= score.overall_score <= 100
        assert score.security_score > 50  # Has security patterns


class TestMLEvaluator:
    """Tests for MLEvaluator."""

    def test_domain_type(self):
        """Test MLEvaluator domain type."""
        evaluator = MLEvaluator()
        assert evaluator.domain_type == DomainType.ML_AI

    def test_file_extensions(self):
        """Test MLEvaluator file extensions."""
        evaluator = MLEvaluator()
        extensions = evaluator.get_file_extensions()

        assert ".py" in extensions
        assert ".ipynb" in extensions
        assert ".yaml" in extensions

    def test_patterns(self):
        """Test MLEvaluator patterns."""
        evaluator = MLEvaluator()
        patterns = evaluator.get_patterns()

        assert "pytorch_usage" in patterns
        assert "tensorflow_usage" in patterns
        assert "training_loop" in patterns


class TestFintechEvaluator:
    """Tests for FintechEvaluator."""

    def test_domain_type(self):
        """Test FintechEvaluator domain type."""
        evaluator = FintechEvaluator()
        assert evaluator.domain_type == DomainType.FINTECH

    def test_patterns(self):
        """Test FintechEvaluator patterns."""
        evaluator = FintechEvaluator()
        patterns = evaluator.get_patterns()

        assert "payment_gateway" in patterns
        assert "pci_compliance" in patterns
        assert "kyc_aml" in patterns


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_calculate_weighted_score(self):
        """Test weighted score calculation."""
        metrics = {"security": 80, "architecture": 70, "innovation": 90}
        weights = {"security": 0.4, "architecture": 0.3, "innovation": 0.3}

        score = calculate_weighted_score(metrics, weights)

        # (80*0.4 + 70*0.3 + 90*0.3) = 32 + 21 + 27 = 80
        assert score == pytest.approx(80.0, rel=0.01)

    def test_normalize_score(self):
        """Test score normalization."""
        assert normalize_score(50) == 50
        assert normalize_score(-10) == 0
        assert normalize_score(150) == 100
        assert normalize_score(75.5) == 75.5


# =============================================================================
# Pattern Constants Tests
# =============================================================================


class TestPatternConstants:
    """Tests for pattern constant definitions."""

    def test_web3_patterns_structure(self):
        """Test Web3 patterns have required fields."""
        for name, pattern in WEB3_PATTERNS.items():
            assert "regex" in pattern, f"Pattern {name} missing regex"
            assert "category" in pattern, f"Pattern {name} missing category"
            assert "severity" in pattern, f"Pattern {name} missing severity"

    def test_ml_patterns_structure(self):
        """Test ML patterns have required fields."""
        for name, pattern in ML_AI_PATTERNS.items():
            assert "regex" in pattern
            assert "category" in pattern

    def test_fintech_patterns_structure(self):
        """Test Fintech patterns have required fields."""
        for name, pattern in FINTECH_PATTERNS.items():
            assert "regex" in pattern
            assert "category" in pattern

    def test_pattern_categories_valid(self):
        """Test all patterns have valid categories."""
        valid_categories = {
            "security",
            "architecture",
            "best_practice",
            "framework",
            "general",
        }

        all_patterns = {
            **WEB3_PATTERNS,
            **ML_AI_PATTERNS,
            **FINTECH_PATTERNS,
            **IOT_PATTERNS,
            **AR_VR_PATTERNS,
        }

        for name, pattern in all_patterns.items():
            category = pattern.get("category", "general")
            assert category in valid_categories, (
                f"Pattern {name} has invalid category: {category}"
            )


# =============================================================================
# Integration Tests
# =============================================================================


class TestOrchestratorIntegration:
    """Integration tests for DomainOrchestrator."""

    def test_orchestrator_initialization(self):
        """Test orchestrator initializes correctly."""
        orchestrator = DomainOrchestrator()

        assert orchestrator.registry is not None
        assert len(orchestrator.get_supported_domains()) == 5

    def test_format_result_for_storage(self, sample_patterns):
        """Test formatting result for database storage."""
        orchestrator = DomainOrchestrator()

        score = DomainScore(
            domain=DomainType.WEB3,
            overall_score=78.5,
        )

        result = DomainEvaluationResult(
            detected_domain=DomainType.WEB3,
            confidence=0.92,
            score=score,
            patterns_found=sample_patterns,
            recommendations=["Test recommendation"],
            strengths=["Test strength"],
            weaknesses=["Test weakness"],
            metadata=EvaluationMetadata(evaluation_id="test123", files_analyzed=10),
        )

        storage_format = orchestrator.format_result_for_storage(result)

        assert "domain_evaluation" in storage_format
        assert storage_format["domain_evaluation"]["detected_domain"] == "web3"
        assert storage_format["domain_evaluation"]["confidence"] == 0.92
        assert storage_format["domain_evaluation"]["score"]["overall"] == 78.5


# =============================================================================
# Main Test Runner
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
