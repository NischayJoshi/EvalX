"""
Fintech Domain Evaluator
========================

Comprehensive evaluator for financial technology projects including:
- Payment processing patterns (Stripe, PayPal, etc.)
- Banking and ledger systems
- Compliance frameworks (PCI-DSS, KYC/AML)
- Security patterns for financial data
- Open banking integrations

Scoring Weights:
    - Security: 35% (critical for financial applications)
    - Architecture: 25% (transaction handling, ledger design)
    - Compliance: 20% (regulatory adherence)
    - Best Practices: 10% (audit trails, idempotency)
    - Completeness: 10% (documentation, testing)

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
from evaluators.constants import FINTECH_PATTERNS
from evaluators.detector_utils import (
    calculate_weighted_score,
    normalize_score,
    count_pattern_categories,
)


logger = logging.getLogger(__name__)


class FintechEvaluator(BaseDomainEvaluator):
    """
    Domain evaluator specialized for Financial Technology projects.
    
    This evaluator analyzes fintech codebases for security best practices,
    compliance patterns, proper transaction handling, and adherence to
    financial industry standards.
    
    Features:
        - Payment gateway integration detection (Stripe, PayPal, etc.)
        - Banking system pattern recognition
        - Compliance framework detection (PCI, KYC/AML)
        - Security pattern analysis (encryption, tokenization)
        - Audit trail and transaction logging
        - Open banking API detection
    
    Example:
        >>> evaluator = FintechEvaluator("/path/to/payment-service")
        >>> result = await evaluator.evaluate()
        >>> print(f"Compliance Score: {result.score.best_practices_score}")
    """
    
    # Scoring weights for Fintech domain (security-heavy)
    SCORING_WEIGHTS = {
        "security": 0.35,
        "architecture": 0.25,
        "compliance": 0.20,
        "best_practices": 0.10,
        "completeness": 0.10,
    }
    
    # Critical security patterns for financial applications
    CRITICAL_SECURITY_PATTERNS = [
        "pci_compliance",
        "encryption_at_rest",
        "transaction_security",
        "fraud_detection",
    ]
    
    # Compliance-related patterns
    COMPLIANCE_PATTERNS = [
        "pci_compliance",
        "kyc_aml",
        "regulatory_reporting",
        "audit_logging",
    ]
    
    # Payment integration patterns
    PAYMENT_PATTERNS = [
        "payment_gateway",
        "payment_processing",
        "recurring_billing",
    ]
    
    @property
    def domain_type(self) -> DomainType:
        """Return the Fintech domain type."""
        return DomainType.FINTECH
    
    def get_file_extensions(self) -> List[str]:
        """
        Return file extensions relevant to Fintech development.
        
        Includes:
            - .py: Python (backend services)
            - .java: Java (enterprise systems)
            - .ts/.js: TypeScript/JavaScript
            - .go: Go (high-performance services)
            - .cs: C# (.NET financial services)
            - .rb: Ruby (payment integrations)
            - .sql: Database schemas
        """
        return [
            '.py',       # Python
            '.java',     # Java
            '.ts',       # TypeScript
            '.js',       # JavaScript
            '.go',       # Go
            '.cs',       # C#
            '.rb',       # Ruby
            '.sql',      # SQL schemas
            '.json',     # Configs
            '.yaml',     # Configs
            '.yml',      # Configs
        ]
    
    def get_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Return Fintech-specific detection patterns."""
        return FINTECH_PATTERNS
    
    async def calculate_domain_score(
        self,
        patterns: List[PatternMatch],
        file_analysis: Dict[str, Any]
    ) -> DomainScore:
        """
        Calculate comprehensive Fintech domain score.
        
        Scoring methodology:
            1. Security Score (35%): Encryption, tokenization, 
               fraud detection, secure transaction handling
            2. Architecture Score (25%): Ledger design, transaction
               patterns, idempotency, event sourcing
            3. Compliance Score (20%): PCI-DSS, KYC/AML, audit trails
            4. Best Practices Score (10%): Error handling, logging
            5. Completeness Score (10%): Tests, documentation
        
        Args:
            patterns: Detected patterns from codebase analysis
            file_analysis: File-level statistics and analysis
        
        Returns:
            DomainScore with detailed breakdown
        """
        logger.info(f"[{self._evaluation_id}] Calculating Fintech domain score")
        
        # Categorize patterns
        category_counts = count_pattern_categories(patterns)
        
        # 1. Security Score
        security_score = await self._calculate_security_score(patterns)
        
        # 2. Architecture Score
        architecture_score = await self._calculate_architecture_score(
            patterns, 
            file_analysis
        )
        
        # 3. Compliance Score (maps to best_practices in DomainScore)
        compliance_score = await self._calculate_compliance_score(patterns)
        
        # 4. Best Practices Score (maps to innovation for variety)
        best_practices_score = await self._calculate_best_practices_score(patterns)
        
        # 5. Completeness Score
        completeness_score = await self._calculate_completeness_score(
            file_analysis
        )
        
        # Calculate weighted overall score
        metrics = {
            "security": security_score,
            "architecture": architecture_score,
            "compliance": compliance_score,
            "best_practices": best_practices_score,
            "completeness": completeness_score,
        }
        
        # Remap weights for standard DomainScore
        overall_score = (
            security_score * 0.35 +
            architecture_score * 0.25 +
            compliance_score * 0.20 +
            best_practices_score * 0.10 +
            completeness_score * 0.10
        )
        
        # Detect payment provider
        pattern_names = {p.pattern_name for p in patterns}
        payment_provider = self._detect_payment_provider(pattern_names, patterns)
        
        # Build detailed breakdown
        breakdown = {
            "primary_payment_provider": payment_provider,
            "security_patterns_found": category_counts.get("security", 0),
            "architecture_patterns_found": category_counts.get("architecture", 0),
            "has_pci_compliance": "pci_compliance" in pattern_names,
            "has_kyc_aml": "kyc_aml" in pattern_names,
            "has_fraud_detection": "fraud_detection" in pattern_names,
            "has_audit_logging": "audit_logging" in pattern_names,
            "has_encryption": "encryption_at_rest" in pattern_names,
            "compliance_maturity": self._calculate_compliance_maturity(pattern_names),
            "payment_integration_count": sum(
                1 for p in self.PAYMENT_PATTERNS if p in pattern_names
            ),
        }
        
        return DomainScore(
            domain=self.domain_type,
            overall_score=normalize_score(overall_score),
            architecture_score=normalize_score(architecture_score),
            security_score=normalize_score(security_score),
            best_practices_score=normalize_score(compliance_score),  # Compliance as BP
            innovation_score=normalize_score(best_practices_score),  # BP as innovation
            completeness_score=normalize_score(completeness_score),
            breakdown=breakdown,
        )
    
    def _detect_payment_provider(
        self, 
        pattern_names: set, 
        patterns: List[PatternMatch]
    ) -> str:
        """Detect primary payment provider from patterns."""
        # Check pattern contexts for provider names
        for pattern in patterns:
            if pattern.pattern_name == "payment_gateway":
                context = pattern.context or ""
                context_lower = context.lower()
                if "stripe" in context_lower:
                    return "Stripe"
                if "paypal" in context_lower:
                    return "PayPal"
                if "braintree" in context_lower:
                    return "Braintree"
                if "adyen" in context_lower:
                    return "Adyen"
                if "square" in context_lower:
                    return "Square"
        
        if "plaid_integration" in pattern_names:
            return "Plaid (Banking)"
        
        return "Custom/Unknown"
    
    def _calculate_compliance_maturity(self, pattern_names: set) -> float:
        """Calculate compliance maturity level (0-100)."""
        maturity = 0
        
        for pattern in self.COMPLIANCE_PATTERNS:
            if pattern in pattern_names:
                maturity += 25
        
        return min(100, maturity)
    
    async def _calculate_security_score(
        self, 
        patterns: List[PatternMatch]
    ) -> float:
        """
        Calculate security score for financial applications.
        
        Factors:
            - PCI compliance patterns
            - Encryption implementation
            - Fraud detection mechanisms
            - Transaction security (idempotency, etc.)
            - Tokenization patterns
        """
        base_score = 40.0
        
        security_patterns = [p for p in patterns if p.category == "security"]
        pattern_names = {p.pattern_name for p in patterns}
        
        # Critical security bonuses
        base_score += len(security_patterns) * 4
        
        # PCI compliance (critical)
        if "pci_compliance" in pattern_names:
            base_score += 18
        else:
            base_score -= 10  # Penalty for missing PCI
        
        # Encryption
        if "encryption_at_rest" in pattern_names:
            base_score += 15
        
        # Transaction security
        if "transaction_security" in pattern_names:
            base_score += 12
        
        # Fraud detection
        if "fraud_detection" in pattern_names:
            base_score += 12
        
        # KYC/AML
        if "kyc_aml" in pattern_names:
            base_score += 10
        
        # High severity patterns
        critical_patterns = [
            p for p in security_patterns 
            if p.severity in ("critical", "high")
        ]
        base_score += len(critical_patterns) * 3
        
        return normalize_score(base_score)
    
    async def _calculate_architecture_score(
        self,
        patterns: List[PatternMatch],
        file_analysis: Dict[str, Any]
    ) -> float:
        """
        Calculate financial architecture quality score.
        
        Factors:
            - Payment integration patterns
            - Account/ledger management
            - Transaction handling
            - Double-entry bookkeeping
            - Event sourcing patterns
        """
        base_score = 45.0
        
        arch_patterns = [p for p in patterns if p.category == "architecture"]
        pattern_names = {p.pattern_name for p in patterns}
        
        base_score += len(arch_patterns) * 4
        
        # Payment patterns
        if "payment_gateway" in pattern_names:
            base_score += 12
        
        if "payment_processing" in pattern_names:
            base_score += 10
        
        if "recurring_billing" in pattern_names:
            base_score += 8
        
        # Banking patterns
        if "account_management" in pattern_names:
            base_score += 10
        
        if "double_entry_bookkeeping" in pattern_names:
            base_score += 12
        
        if "transaction_ledger" in pattern_names:
            base_score += 10
        
        # Open banking
        if "open_banking_api" in pattern_names:
            base_score += 8
        
        if "plaid_integration" in pattern_names:
            base_score += 8
        
        return normalize_score(base_score)
    
    async def _calculate_compliance_score(
        self, 
        patterns: List[PatternMatch]
    ) -> float:
        """
        Calculate regulatory compliance score.
        
        Factors:
            - PCI-DSS patterns
            - KYC/AML implementation
            - Regulatory reporting
            - Audit trail coverage
        """
        base_score = 40.0
        
        pattern_names = {p.pattern_name for p in patterns}
        
        # PCI compliance
        if "pci_compliance" in pattern_names:
            base_score += 20
        
        # KYC/AML
        if "kyc_aml" in pattern_names:
            base_score += 18
        
        # Regulatory reporting
        if "regulatory_reporting" in pattern_names:
            base_score += 15
        
        # Audit logging (essential for compliance)
        if "audit_logging" in pattern_names:
            base_score += 15
        
        # Transaction ledger (audit trail)
        if "transaction_ledger" in pattern_names:
            base_score += 10
        
        return normalize_score(base_score)
    
    async def _calculate_best_practices_score(
        self, 
        patterns: List[PatternMatch]
    ) -> float:
        """
        Calculate fintech best practices score.
        
        Factors:
            - Idempotency patterns
            - Error handling
            - Interest calculations
            - Proper logging
        """
        base_score = 50.0
        
        pattern_names = {p.pattern_name for p in patterns}
        
        # Transaction security includes idempotency
        if "transaction_security" in pattern_names:
            base_score += 15
        
        # Interest calculation (domain expertise)
        if "interest_calculation" in pattern_names:
            base_score += 10
        
        # Audit logging (best practice)
        if "audit_logging" in pattern_names:
            base_score += 12
        
        # Multiple payment integrations
        payment_count = sum(
            1 for p in self.PAYMENT_PATTERNS 
            if p in pattern_names
        )
        base_score += payment_count * 5
        
        return normalize_score(base_score)
    
    async def _calculate_completeness_score(
        self, 
        file_analysis: Dict[str, Any]
    ) -> float:
        """
        Calculate project completeness score.
        
        Factors:
            - Code file count
            - Configuration files
            - Multi-language support
            - Total codebase size
        """
        base_score = 50.0
        
        files_by_ext = file_analysis.get("files_by_extension", {})
        
        # Primary language files
        total_code_files = sum([
            files_by_ext.get(".py", 0),
            files_by_ext.get(".java", 0),
            files_by_ext.get(".ts", 0),
            files_by_ext.get(".js", 0),
            files_by_ext.get(".go", 0),
            files_by_ext.get(".cs", 0),
        ])
        
        if total_code_files >= 5:
            base_score += 10
        if total_code_files >= 15:
            base_score += 10
        
        # Config files
        config_files = files_by_ext.get(".yaml", 0) + files_by_ext.get(".yml", 0)
        if config_files >= 1:
            base_score += 8
        
        # SQL schemas
        sql_files = files_by_ext.get(".sql", 0)
        if sql_files >= 1:
            base_score += 8
        
        # Total codebase
        total_lines = file_analysis.get("total_lines", 0)
        if total_lines > 1000:
            base_score += 8
        if total_lines > 5000:
            base_score += 5
        
        return normalize_score(base_score)
    
    async def _generate_recommendations(
        self,
        patterns: List[PatternMatch],
        score: DomainScore
    ) -> List[str]:
        """Generate Fintech-specific recommendations."""
        recommendations = []
        pattern_names = {p.pattern_name for p in patterns}
        
        # Critical security recommendations
        if score.security_score < 70:
            recommendations.append(
                "🔒 Critical: Strengthen security posture - financial applications "
                "require comprehensive security measures"
            )
        
        if "pci_compliance" not in pattern_names:
            recommendations.append(
                "⚠️ Critical: Implement PCI-DSS compliance patterns including "
                "card tokenization and secure data handling"
            )
        
        if "encryption_at_rest" not in pattern_names:
            recommendations.append(
                "🔐 Add encryption at rest for all sensitive financial data "
                "using AES-256 or equivalent"
            )
        
        # Compliance recommendations
        if "kyc_aml" not in pattern_names:
            recommendations.append(
                "📋 Implement KYC/AML verification flows for regulatory compliance"
            )
        
        if "audit_logging" not in pattern_names:
            recommendations.append(
                "📝 Add comprehensive audit logging with immutable records "
                "for compliance and forensics"
            )
        
        # Architecture recommendations
        if "transaction_security" not in pattern_names:
            recommendations.append(
                "🔄 Implement idempotency keys and transaction deduplication "
                "to prevent duplicate charges"
            )
        
        if "fraud_detection" not in pattern_names:
            recommendations.append(
                "🛡️ Add fraud detection mechanisms including velocity checks "
                "and anomaly detection"
            )
        
        if "double_entry_bookkeeping" not in pattern_names:
            recommendations.append(
                "📊 Consider double-entry bookkeeping for accurate financial "
                "record keeping and reconciliation"
            )
        
        # Architecture improvements
        if score.architecture_score < 70:
            recommendations.append(
                "📐 Improve architecture with proper separation of payment "
                "processing, ledger, and reporting concerns"
            )
        
        return recommendations[:8]
    
    async def _identify_insights(
        self,
        patterns: List[PatternMatch],
        file_analysis: Dict[str, Any]
    ) -> tuple[List[str], List[str]]:
        """Identify Fintech-specific strengths and weaknesses."""
        strengths = []
        weaknesses = []
        
        pattern_names = {p.pattern_name for p in patterns}
        security_patterns = [p for p in patterns if p.category == "security"]
        
        # Strengths
        if "pci_compliance" in pattern_names:
            strengths.append("PCI-DSS compliance patterns implemented")
        
        if len(security_patterns) >= 3:
            strengths.append("Strong security foundation with multiple safeguards")
        
        if "payment_gateway" in pattern_names:
            strengths.append("Professional payment gateway integration")
        
        if "audit_logging" in pattern_names:
            strengths.append("Comprehensive audit trail for compliance")
        
        if "fraud_detection" in pattern_names:
            strengths.append("Active fraud detection mechanisms in place")
        
        if "double_entry_bookkeeping" in pattern_names:
            strengths.append("Proper financial record-keeping with double-entry")
        
        # Weaknesses
        if "pci_compliance" not in pattern_names:
            weaknesses.append("Missing PCI-DSS compliance - critical for card handling")
        
        if "encryption_at_rest" not in pattern_names:
            weaknesses.append("No data encryption at rest detected")
        
        if len(security_patterns) < 2:
            weaknesses.append("Insufficient security patterns for financial application")
        
        if "audit_logging" not in pattern_names:
            weaknesses.append("Missing audit logging - compliance risk")
        
        if "transaction_security" not in pattern_names:
            weaknesses.append("No idempotency patterns - risk of duplicate transactions")
        
        return strengths, weaknesses
