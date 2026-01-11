"""
Web3/Blockchain Domain Evaluator
================================

Comprehensive evaluator for blockchain and Web3 projects including:
- Smart contract analysis (Solidity, Vyper, Rust/Anchor)
- DeFi protocol patterns (AMM, lending, staking)
- Security vulnerability detection
- Token standard compliance (ERC-20, ERC-721, ERC-1155)
- Gas optimization patterns
- Upgrade patterns and governance

Scoring Weights:
    - Security: 35% (critical for smart contracts)
    - Architecture: 25% (modular, upgradeable patterns)
    - Best Practices: 20% (gas optimization, events, etc.)
    - Innovation: 10% (novel mechanisms)
    - Completeness: 10% (test coverage, documentation)

Author: EvalX Team
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import re

from evaluators.base import BaseDomainEvaluator
from evaluators.models import (
    DomainType,
    DomainScore,
    PatternMatch,
)
from evaluators.constants import WEB3_PATTERNS
from evaluators.detector_utils import (
    calculate_weighted_score,
    normalize_score,
    count_pattern_categories,
)


logger = logging.getLogger(__name__)


class Web3Evaluator(BaseDomainEvaluator):
    """
    Domain evaluator specialized for Web3 and blockchain projects.

    This evaluator analyzes smart contracts, DeFi protocols, and blockchain
    infrastructure code for security vulnerabilities, architectural patterns,
    and adherence to industry best practices.

    Features:
        - Multi-language support (Solidity, Vyper, Rust/Anchor, Move)
        - Security vulnerability detection (reentrancy, overflow, access control)
        - Token standard compliance checking
        - DeFi pattern recognition (AMM, flash loans, staking)
        - Gas optimization analysis
        - Upgrade pattern validation

    Example:
        >>> evaluator = Web3Evaluator("/path/to/defi-protocol")
        >>> result = await evaluator.evaluate()
        >>> print(f"Security Score: {result.score.security_score}")
    """

    # Scoring weights for Web3 domain
    SCORING_WEIGHTS = {
        "security": 0.35,
        "architecture": 0.25,
        "best_practices": 0.20,
        "innovation": 0.10,
        "completeness": 0.10,
    }

    # Critical security patterns that should be present
    REQUIRED_SECURITY_PATTERNS = [
        "reentrancy_guard",
        "access_control",
    ]

    # Patterns indicating advanced DeFi implementations
    ADVANCED_DEFI_PATTERNS = [
        "flash_loan",
        "liquidity_pool",
        "oracle_integration",
        "staking_mechanism",
    ]

    @property
    def domain_type(self) -> DomainType:
        """Return the Web3 domain type."""
        return DomainType.WEB3

    def get_file_extensions(self) -> List[str]:
        """
        Return file extensions relevant to Web3 development.

        Includes:
            - .sol: Solidity smart contracts
            - .vy: Vyper smart contracts
            - .rs: Rust (Anchor/Solana programs)
            - .move: Move language (Aptos/Sui)
            - .ts/.js: Deployment scripts, tests, frontend
            - .json: ABI files, configurations
        """
        return [
            ".sol",  # Solidity
            ".vy",  # Vyper
            ".rs",  # Rust (Anchor)
            ".move",  # Move
            ".ts",  # TypeScript (scripts/tests)
            ".js",  # JavaScript
            ".json",  # ABI, configs
        ]

    def get_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Return Web3-specific detection patterns."""
        return WEB3_PATTERNS

    async def calculate_domain_score(
        self, patterns: List[PatternMatch], file_analysis: Dict[str, Any]
    ) -> DomainScore:
        """
        Calculate comprehensive Web3 domain score.

        Scoring methodology:
            1. Security Score (35%): Presence of security patterns,
               absence of vulnerabilities, access control coverage
            2. Architecture Score (25%): Modular design, upgrade patterns,
               proper separation of concerns
            3. Best Practices Score (20%): Gas optimization, event emission,
               proper documentation
            4. Innovation Score (10%): Advanced DeFi patterns, novel mechanisms
            5. Completeness Score (10%): Test coverage, documentation, deployment scripts

        Args:
            patterns: Detected patterns from codebase analysis
            file_analysis: File-level statistics and analysis

        Returns:
            DomainScore with detailed breakdown
        """
        logger.info(f"[{self._evaluation_id}] Calculating Web3 domain score")

        # Categorize patterns
        category_counts = count_pattern_categories(patterns)

        # 1. Security Score
        security_score = await self._calculate_security_score(patterns)

        # 2. Architecture Score
        architecture_score = await self._calculate_architecture_score(
            patterns, file_analysis
        )

        # 3. Best Practices Score
        best_practices_score = await self._calculate_best_practices_score(patterns)

        # 4. Innovation Score
        innovation_score = await self._calculate_innovation_score(patterns)

        # 5. Completeness Score
        completeness_score = await self._calculate_completeness_score(file_analysis)

        # Calculate weighted overall score
        metrics = {
            "security": security_score,
            "architecture": architecture_score,
            "best_practices": best_practices_score,
            "innovation": innovation_score,
            "completeness": completeness_score,
        }

        overall_score = calculate_weighted_score(metrics, self.SCORING_WEIGHTS)

        # Build detailed breakdown
        breakdown = {
            "security_patterns_found": category_counts.get("security", 0),
            "architecture_patterns_found": category_counts.get("architecture", 0),
            "has_access_control": any(
                p.pattern_name == "access_control" for p in patterns
            ),
            "has_reentrancy_guard": any(
                p.pattern_name == "reentrancy_guard" for p in patterns
            ),
            "uses_upgradeable_proxy": any(
                p.pattern_name == "upgradeable_proxy" for p in patterns
            ),
            "defi_features_count": sum(
                1 for p in patterns if p.pattern_name in self.ADVANCED_DEFI_PATTERNS
            ),
            "total_sol_files": file_analysis.get("files_by_extension", {}).get(
                ".sol", 0
            ),
        }

        return DomainScore(
            domain=self.domain_type,
            overall_score=normalize_score(overall_score),
            architecture_score=normalize_score(architecture_score),
            security_score=normalize_score(security_score),
            best_practices_score=normalize_score(best_practices_score),
            innovation_score=normalize_score(innovation_score),
            completeness_score=normalize_score(completeness_score),
            breakdown=breakdown,
        )

    async def _calculate_security_score(self, patterns: List[PatternMatch]) -> float:
        """
        Calculate security score based on pattern analysis.

        Scoring factors:
            - Presence of security patterns (+)
            - Critical patterns (reentrancy, access control) (+++)
            - Signature verification patterns (+)
            - Timelock patterns for governance (+)
            - Missing critical patterns (-)
        """
        base_score = 50.0  # Start at baseline

        security_patterns = [p for p in patterns if p.category == "security"]

        # Bonus for each security pattern found
        base_score += len(security_patterns) * 5

        # Critical pattern bonuses
        pattern_names = {p.pattern_name for p in patterns}

        if "reentrancy_guard" in pattern_names:
            base_score += 15
        else:
            base_score -= 10  # Penalty for missing critical pattern

        if "access_control" in pattern_names:
            base_score += 12
        else:
            base_score -= 8

        if "pausable_pattern" in pattern_names:
            base_score += 8

        if "signature_verification" in pattern_names:
            base_score += 10

        if "timelock_pattern" in pattern_names:
            base_score += 8

        if "safe_math" in pattern_names:
            base_score += 5

        # High severity pattern bonus
        high_severity = [
            p for p in security_patterns if p.severity in ("high", "critical")
        ]
        base_score += len(high_severity) * 3

        return normalize_score(base_score)

    async def _calculate_architecture_score(
        self, patterns: List[PatternMatch], file_analysis: Dict[str, Any]
    ) -> float:
        """
        Calculate architecture quality score.

        Factors:
            - Token standard implementations (ERC-20, ERC-721, etc.)
            - Upgradeable patterns
            - Modular contract structure
            - Proper interface usage
        """
        base_score = 50.0

        arch_patterns = [p for p in patterns if p.category == "architecture"]
        pattern_names = {p.pattern_name for p in patterns}

        # Token standard implementations
        if "erc20_implementation" in pattern_names:
            base_score += 10
        if "erc721_nft" in pattern_names:
            base_score += 10
        if "erc1155_multitoken" in pattern_names:
            base_score += 12

        # Upgradeable patterns
        if "upgradeable_proxy" in pattern_names:
            base_score += 10

        # DeFi patterns
        if "liquidity_pool" in pattern_names:
            base_score += 8
        if "staking_mechanism" in pattern_names:
            base_score += 7
        if "oracle_integration" in pattern_names:
            base_score += 8

        # File structure bonus
        sol_files = file_analysis.get("files_by_extension", {}).get(".sol", 0)
        if sol_files > 5:
            base_score += 8  # Multiple contracts indicate modular design

        return normalize_score(base_score)

    async def _calculate_best_practices_score(
        self, patterns: List[PatternMatch]
    ) -> float:
        """
        Calculate best practices adherence score.

        Factors:
            - Event emission for off-chain tracking
            - Gas optimization techniques
            - Proper documentation (NatSpec)
            - Testing infrastructure
        """
        base_score = 50.0

        pattern_names = {p.pattern_name for p in patterns}
        bp_patterns = [p for p in patterns if p.category == "best_practice"]

        base_score += len(bp_patterns) * 5

        if "event_emission" in pattern_names:
            base_score += 12

        if "gas_optimization" in pattern_names:
            base_score += 10

        return normalize_score(base_score)

    async def _calculate_innovation_score(self, patterns: List[PatternMatch]) -> float:
        """
        Calculate innovation score based on advanced features.

        Factors:
            - Flash loan implementations
            - Novel DeFi mechanisms
            - Cross-chain patterns
            - Advanced cryptographic operations
        """
        base_score = 40.0  # Lower baseline for innovation

        pattern_names = {p.pattern_name for p in patterns}

        # Advanced DeFi features
        if "flash_loan" in pattern_names:
            base_score += 20

        if "liquidity_pool" in pattern_names:
            base_score += 15

        if "oracle_integration" in pattern_names:
            base_score += 12

        if "staking_mechanism" in pattern_names:
            base_score += 10

        # Multiple advanced features bonus
        advanced_count = sum(
            1 for p in self.ADVANCED_DEFI_PATTERNS if p in pattern_names
        )
        if advanced_count >= 3:
            base_score += 10

        return normalize_score(base_score)

    async def _calculate_completeness_score(
        self, file_analysis: Dict[str, Any]
    ) -> float:
        """
        Calculate project completeness score.

        Factors:
            - Test file presence
            - Deployment scripts
            - Documentation
            - Multiple contract files
        """
        base_score = 50.0

        total_files = file_analysis.get("total_files", 0)
        files_by_ext = file_analysis.get("files_by_extension", {})

        # Contract files
        sol_files = files_by_ext.get(".sol", 0)
        if sol_files >= 3:
            base_score += 10
        if sol_files >= 5:
            base_score += 5

        # Test/script files (TypeScript/JavaScript)
        ts_files = files_by_ext.get(".ts", 0)
        js_files = files_by_ext.get(".js", 0)
        if ts_files + js_files >= 5:
            base_score += 15

        # JSON configs (hardhat config, deployment info)
        json_files = files_by_ext.get(".json", 0)
        if json_files >= 2:
            base_score += 5

        # Total codebase size bonus
        total_lines = file_analysis.get("total_lines", 0)
        if total_lines > 1000:
            base_score += 10
        if total_lines > 5000:
            base_score += 5

        return normalize_score(base_score)

    async def _generate_recommendations(
        self, patterns: List[PatternMatch], score: DomainScore
    ) -> List[str]:
        """Generate Web3-specific recommendations."""
        recommendations = []
        pattern_names = {p.pattern_name for p in patterns}

        # Security recommendations
        if score.security_score < 70:
            recommendations.append(
                "🔒 Strengthen smart contract security by implementing "
                "comprehensive access control and reentrancy guards"
            )

        if "reentrancy_guard" not in pattern_names:
            recommendations.append(
                "⚠️ Critical: Implement reentrancy guards (OpenZeppelin's "
                "ReentrancyGuard) to prevent reentrancy attacks"
            )

        if "access_control" not in pattern_names:
            recommendations.append(
                "⚠️ Add role-based access control for privileged functions "
                "using OpenZeppelin's AccessControl or Ownable"
            )

        if "pausable_pattern" not in pattern_names:
            recommendations.append(
                "💡 Consider implementing Pausable pattern for emergency stops"
            )

        # Architecture recommendations
        if score.architecture_score < 70:
            recommendations.append(
                "📐 Improve contract architecture with better separation "
                "of concerns and interface abstractions"
            )

        if "upgradeable_proxy" not in pattern_names:
            recommendations.append(
                "💡 Consider upgradeable proxy pattern (UUPS or Transparent) "
                "for future-proofing smart contracts"
            )

        # Best practices recommendations
        if "event_emission" not in pattern_names:
            recommendations.append(
                "📢 Add event emissions for all state changes to enable "
                "off-chain indexing and tracking"
            )

        if "gas_optimization" not in pattern_names:
            recommendations.append(
                "⛽ Optimize gas usage with calldata parameters, immutable "
                "variables, and unchecked arithmetic where safe"
            )

        # Innovation suggestions
        if score.innovation_score < 60:
            recommendations.append(
                "🚀 Consider implementing advanced DeFi features like "
                "flash loans, liquidity pools, or yield strategies"
            )

        return recommendations[:8]  # Limit to top 8 recommendations

    async def _identify_insights(
        self, patterns: List[PatternMatch], file_analysis: Dict[str, Any]
    ) -> tuple[List[str], List[str]]:
        """Identify Web3-specific strengths and weaknesses."""
        strengths = []
        weaknesses = []

        pattern_names = {p.pattern_name for p in patterns}
        security_patterns = [p for p in patterns if p.category == "security"]

        # Strengths
        if len(security_patterns) >= 3:
            strengths.append(
                "Strong security posture with multiple protection mechanisms"
            )

        if "reentrancy_guard" in pattern_names and "access_control" in pattern_names:
            strengths.append(
                "Critical security patterns (reentrancy + access control) implemented"
            )

        if "upgradeable_proxy" in pattern_names:
            strengths.append("Upgradeable architecture for future improvements")

        if any(p in pattern_names for p in self.ADVANCED_DEFI_PATTERNS):
            strengths.append("Advanced DeFi functionality implemented")

        if "event_emission" in pattern_names:
            strengths.append("Proper event emission for off-chain indexing")

        # Weaknesses
        if "reentrancy_guard" not in pattern_names:
            weaknesses.append(
                "Missing reentrancy protection - critical vulnerability risk"
            )

        if "access_control" not in pattern_names:
            weaknesses.append("No access control patterns detected")

        if len(security_patterns) < 2:
            weaknesses.append("Limited security mechanisms implemented")

        sol_files = file_analysis.get("files_by_extension", {}).get(".sol", 0)
        if sol_files < 2:
            weaknesses.append("Limited smart contract modularity")

        return strengths, weaknesses
