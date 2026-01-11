"""
Domain Evaluation Models
=======================

Pydantic models and type definitions for domain-specific evaluation results.
These models ensure type safety and provide clear structure for evaluation outputs.

Models:
    - DomainType: Enum of supported domain categories
    - PatternMatch: Individual pattern detection result
    - DomainScore: Scoring breakdown for a domain
    - DomainEvaluationResult: Complete evaluation output
    - EvaluationMetadata: Contextual information about evaluation run

Author: EvalX Team
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class DomainType(str, Enum):
    """
    Enumeration of supported domain types for evaluation.
    
    Each domain has specialized patterns, best practices, and scoring criteria
    that are applied during the evaluation process.
    """
    WEB3 = "web3"
    ML_AI = "ml_ai"
    FINTECH = "fintech"
    IOT = "iot"
    AR_VR = "ar_vr"
    UNKNOWN = "unknown"
    
    @classmethod
    def from_string(cls, value: str) -> "DomainType":
        """Convert string to DomainType with fallback to UNKNOWN."""
        mapping = {
            "web3": cls.WEB3,
            "blockchain": cls.WEB3,
            "crypto": cls.WEB3,
            "ml": cls.ML_AI,
            "ai": cls.ML_AI,
            "machine_learning": cls.ML_AI,
            "fintech": cls.FINTECH,
            "finance": cls.FINTECH,
            "banking": cls.FINTECH,
            "iot": cls.IOT,
            "embedded": cls.IOT,
            "ar": cls.AR_VR,
            "vr": cls.AR_VR,
            "xr": cls.AR_VR,
        }
        return mapping.get(value.lower(), cls.UNKNOWN)


class PatternMatch(BaseModel):
    """
    Represents a single pattern detection result within the codebase.
    
    Attributes:
        pattern_name: Identifier for the detected pattern
        file_path: Relative path to the file containing the pattern
        line_number: Line number where pattern was detected (1-indexed)
        confidence: Detection confidence score (0.0 to 1.0)
        context: Code snippet or surrounding context
        category: Classification category (security, architecture, etc.)
        severity: Impact level (info, low, medium, high, critical)
    """
    pattern_name: str = Field(..., description="Name of the detected pattern")
    file_path: str = Field(..., description="File where pattern was found")
    line_number: Optional[int] = Field(None, description="Line number (1-indexed)")
    confidence: float = Field(
        default=1.0, 
        ge=0.0, 
        le=1.0, 
        description="Detection confidence"
    )
    context: Optional[str] = Field(None, description="Code context snippet")
    category: str = Field(default="general", description="Pattern category")
    severity: str = Field(default="info", description="Impact severity level")

    class Config:
        json_schema_extra = {
            "example": {
                "pattern_name": "smart_contract_reentrancy_guard",
                "file_path": "contracts/Token.sol",
                "line_number": 45,
                "confidence": 0.95,
                "context": "modifier nonReentrant() {",
                "category": "security",
                "severity": "high"
            }
        }


class DomainScore(BaseModel):
    """
    Detailed scoring breakdown for domain-specific evaluation.
    
    Attributes:
        domain: The evaluated domain type
        overall_score: Aggregate domain score (0-100)
        architecture_score: Code architecture quality score
        security_score: Security best practices adherence
        best_practices_score: Domain-specific best practices
        innovation_score: Novel approaches and implementations
        completeness_score: Feature completeness assessment
        breakdown: Detailed category-wise scoring
    """
    domain: DomainType = Field(..., description="Evaluated domain")
    overall_score: float = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Overall domain score"
    )
    architecture_score: float = Field(
        default=0.0, 
        ge=0, 
        le=100, 
        description="Architecture quality"
    )
    security_score: float = Field(
        default=0.0, 
        ge=0, 
        le=100, 
        description="Security adherence"
    )
    best_practices_score: float = Field(
        default=0.0, 
        ge=0, 
        le=100, 
        description="Best practices"
    )
    innovation_score: float = Field(
        default=0.0, 
        ge=0, 
        le=100, 
        description="Innovation level"
    )
    completeness_score: float = Field(
        default=0.0, 
        ge=0, 
        le=100, 
        description="Feature completeness"
    )
    breakdown: Dict[str, float] = Field(
        default_factory=dict, 
        description="Detailed scoring breakdown"
    )

    def calculate_grade(self) -> str:
        """Convert overall score to letter grade."""
        score = self.overall_score
        if score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "A-"
        elif score >= 75:
            return "B+"
        elif score >= 70:
            return "B"
        elif score >= 65:
            return "B-"
        elif score >= 60:
            return "C+"
        elif score >= 55:
            return "C"
        elif score >= 50:
            return "C-"
        elif score >= 40:
            return "D"
        else:
            return "F"


class EvaluationMetadata(BaseModel):
    """
    Metadata about the evaluation run for tracking and auditing.
    
    Attributes:
        evaluation_id: Unique identifier for this evaluation
        started_at: Timestamp when evaluation began
        completed_at: Timestamp when evaluation finished
        duration_ms: Total evaluation duration in milliseconds
        files_analyzed: Number of files processed
        patterns_checked: Number of patterns evaluated
        evaluator_version: Version of the evaluator used
    """
    evaluation_id: str = Field(..., description="Unique evaluation ID")
    started_at: datetime = Field(
        default_factory=datetime.utcnow, 
        description="Start timestamp"
    )
    completed_at: Optional[datetime] = Field(None, description="End timestamp")
    duration_ms: Optional[int] = Field(None, description="Duration in milliseconds")
    files_analyzed: int = Field(default=0, description="Files processed count")
    patterns_checked: int = Field(default=0, description="Patterns evaluated")
    evaluator_version: str = Field(default="1.0.0", description="Evaluator version")


class DomainEvaluationResult(BaseModel):
    """
    Complete result of a domain-specific evaluation.
    
    This is the primary output model containing all evaluation data,
    pattern matches, scores, recommendations, and metadata.
    
    Attributes:
        detected_domain: Primary domain detected in the codebase
        secondary_domains: Additional domains with partial presence
        confidence: Domain detection confidence (0.0 to 1.0)
        score: Detailed scoring breakdown
        patterns_found: List of detected patterns
        recommendations: Improvement suggestions
        strengths: Identified strong points
        weaknesses: Areas needing improvement
        metadata: Evaluation run metadata
        raw_analysis: Optional raw analysis data for debugging
    """
    detected_domain: DomainType = Field(
        ..., 
        description="Primary detected domain"
    )
    secondary_domains: List[DomainType] = Field(
        default_factory=list, 
        description="Secondary domains detected"
    )
    confidence: float = Field(
        default=1.0, 
        ge=0.0, 
        le=1.0, 
        description="Detection confidence"
    )
    score: DomainScore = Field(..., description="Evaluation scoring")
    patterns_found: List[PatternMatch] = Field(
        default_factory=list, 
        description="Detected patterns"
    )
    recommendations: List[str] = Field(
        default_factory=list, 
        description="Improvement recommendations"
    )
    strengths: List[str] = Field(
        default_factory=list, 
        description="Identified strengths"
    )
    weaknesses: List[str] = Field(
        default_factory=list, 
        description="Identified weaknesses"
    )
    metadata: Optional[EvaluationMetadata] = Field(
        None, 
        description="Evaluation metadata"
    )
    raw_analysis: Optional[Dict[str, Any]] = Field(
        None, 
        description="Raw analysis data"
    )

    def to_markdown_report(self) -> str:
        """Generate a Markdown-formatted evaluation report."""
        grade = self.score.calculate_grade()
        report_lines = [
            f"# Domain Evaluation Report: {self.detected_domain.value.upper()}",
            "",
            f"**Overall Score:** {self.score.overall_score:.1f}/100 ({grade})",
            f"**Detection Confidence:** {self.confidence * 100:.1f}%",
            "",
            "## Score Breakdown",
            "",
            f"- Architecture: {self.score.architecture_score:.1f}/100",
            f"- Security: {self.score.security_score:.1f}/100",
            f"- Best Practices: {self.score.best_practices_score:.1f}/100",
            f"- Innovation: {self.score.innovation_score:.1f}/100",
            f"- Completeness: {self.score.completeness_score:.1f}/100",
            "",
        ]
        
        if self.strengths:
            report_lines.extend(["## Strengths", ""])
            for strength in self.strengths:
                report_lines.append(f"- ✅ {strength}")
            report_lines.append("")
        
        if self.weaknesses:
            report_lines.extend(["## Areas for Improvement", ""])
            for weakness in self.weaknesses:
                report_lines.append(f"- ⚠️ {weakness}")
            report_lines.append("")
        
        if self.recommendations:
            report_lines.extend(["## Recommendations", ""])
            for i, rec in enumerate(self.recommendations, 1):
                report_lines.append(f"{i}. {rec}")
            report_lines.append("")
        
        if self.patterns_found:
            report_lines.extend([
                "## Detected Patterns",
                "",
                f"Found {len(self.patterns_found)} domain-specific patterns:",
                ""
            ])
            for pattern in self.patterns_found[:10]:  # Limit to top 10
                report_lines.append(
                    f"- **{pattern.pattern_name}** in `{pattern.file_path}` "
                    f"(confidence: {pattern.confidence * 100:.0f}%)"
                )
        
        return "\n".join(report_lines)

    class Config:
        json_schema_extra = {
            "example": {
                "detected_domain": "web3",
                "confidence": 0.92,
                "score": {
                    "domain": "web3",
                    "overall_score": 78.5,
                    "architecture_score": 82.0,
                    "security_score": 75.0,
                    "best_practices_score": 80.0,
                    "innovation_score": 70.0,
                    "completeness_score": 85.0
                },
                "recommendations": [
                    "Implement reentrancy guards in all external calls",
                    "Add comprehensive unit tests for smart contracts"
                ]
            }
        }
