"""
Analytics Constants and Configuration
======================================

This module contains all constants, thresholds, and configuration
values used across the analytics module. Centralizing these values
makes the system easier to tune and maintain.

Categories:
    - Theme Detection: Keywords and patterns for classifying events
    - Scoring: Thresholds for anomaly detection and grading
    - Time Analysis: Buckets and deadlines for submission patterns
    - Skill Mapping: Dimension definitions for radar charts
    - Export: Available columns and default configurations
"""

from typing import Dict, List, Set
import re


# =============================================================================
# THEME DETECTION PATTERNS
# =============================================================================

# Keywords used to detect hackathon themes from event descriptions
# Each theme has a list of regex patterns (case-insensitive matching)

THEME_PATTERNS: Dict[str, List[str]] = {
    "AI/ML": [
        r"\bai\b",
        r"\bartificial\s+intelligence\b",
        r"\bmachine\s+learning\b",
        r"\bml\b",
        r"\bdeep\s+learning\b",
        r"\bneural\s+network",
        r"\bnlp\b",
        r"\bnatural\s+language\s+processing\b",
        r"\bcomputer\s+vision\b",
        r"\bllm\b",
        r"\blarge\s+language\s+model",
        r"\bgpt\b",
        r"\btransformer",
        r"\breinforcement\s+learning\b",
        r"\bpredictive\s+analytics\b",
        r"\bdata\s+science\b",
        r"\bgenerat(ive|or)\b",
    ],
    "Web3": [
        r"\bweb3\b",
        r"\bblockchain\b",
        r"\bcrypto",
        r"\bnft\b",
        r"\bdecentrali[zs]ed\b",
        r"\bdefi\b",
        r"\bsmart\s+contract",
        r"\bethereum\b",
        r"\bsolana\b",
        r"\bsolidity\b",
        r"\bdao\b",
        r"\btoken",
        r"\bwallet\b",
        r"\bmetamask\b",
        r"\bweb\s*3\b",
        r"\bdapp\b",
    ],
    "IoT": [
        r"\biot\b",
        r"\binternet\s+of\s+things\b",
        r"\bsmart\s+home\b",
        r"\bsmart\s+city\b",
        r"\bsensor",
        r"\bembedded\b",
        r"\braspberry\s+pi\b",
        r"\barduino\b",
        r"\bhardware\b",
        r"\bwearable",
        r"\bconnected\s+device",
        r"\bedge\s+computing\b",
        r"\bmqtt\b",
        r"\brobotic",
        r"\bdrone",
    ],
    "Fintech": [
        r"\bfintech\b",
        r"\bfinancial\s+tech",
        r"\bpayment",
        r"\bbanking\b",
        r"\binvest",
        r"\btrading\b",
        r"\bstock",
        r"\binsurance\b",
        r"\binsurtech\b",
        r"\blending\b",
        r"\bloan",
        r"\bbudget",
        r"\bpersonal\s+finance\b",
        r"\bregtech\b",
        r"\bcompliance\b",
        r"\bkyc\b",
    ],
    "AR/VR": [
        r"\bar\b",
        r"\bvr\b",
        r"\baugmented\s+reality\b",
        r"\bvirtual\s+reality\b",
        r"\bmixed\s+reality\b",
        r"\bxr\b",
        r"\bmetaverse\b",
        r"\bimmersive\b",
        r"\b3d\b",
        r"\bholograph",
        r"\bunity\b",
        r"\bunreal\s+engine\b",
        r"\bspatial\b",
        r"\bheadset\b",
        r"\boculus\b",
    ],
    "Healthcare": [
        r"\bhealthcare\b",
        r"\bhealth\s+tech\b",
        r"\bmedical\b",
        r"\btelemedicine\b",
        r"\btelehealth\b",
        r"\bpatient\b",
        r"\bdiagnos",
        r"\behr\b",
        r"\belectronic\s+health\b",
        r"\bdrug\b",
        r"\bpharma",
        r"\bmental\s+health\b",
        r"\bfitness\b",
        r"\bwellness\b",
        r"\bbiotech\b",
    ],
    "EdTech": [
        r"\bedtech\b",
        r"\beducation\s+tech",
        r"\be-learning\b",
        r"\bonline\s+learning\b",
        r"\blms\b",
        r"\blearning\s+management\b",
        r"\bcourse",
        r"\btutorial",
        r"\bstudent",
        r"\bteacher",
        r"\bclassroom\b",
        r"\bquiz",
        r"\bassessment\b",
        r"\bskill\s+development\b",
    ],
    "Sustainability": [
        r"\bsustainab",
        r"\bgreen\s+tech\b",
        r"\bclimate\b",
        r"\benvironment",
        r"\bcarbon",
        r"\brenewable\b",
        r"\bsolar\b",
        r"\bwind\s+energy\b",
        r"\bevehicle\b",
        r"\belectric\s+vehicle\b",
        r"\brecycl",
        r"\bwaste\s+management\b",
        r"\beco-friendly\b",
        r"\bnet\s+zero\b",
    ],
}

# Compiled regex patterns for performance
COMPILED_THEME_PATTERNS: Dict[str, List[re.Pattern]] = {
    theme: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for theme, patterns in THEME_PATTERNS.items()
}

# Theme detection minimum confidence threshold (0.0 - 1.0)
THEME_CONFIDENCE_THRESHOLD: float = 0.1

# Default theme when no patterns match
DEFAULT_THEME: str = "General"


# =============================================================================
# SCORING THRESHOLDS AND CONFIGURATION
# =============================================================================

# Z-score threshold for anomaly detection
# Values beyond this are flagged as anomalies
ANOMALY_Z_SCORE_THRESHOLD: float = 2.0

# Minimum submissions required for statistical significance
MIN_SUBMISSIONS_FOR_CALIBRATION: int = 3
MIN_SUBMISSIONS_FOR_PERCENTILE: int = 2
MIN_EVENTS_FOR_TRENDS: int = 2

# Score grade boundaries
SCORE_GRADES: Dict[str, tuple] = {
    "A+": (95, 100),
    "A": (90, 94.99),
    "A-": (85, 89.99),
    "B+": (80, 84.99),
    "B": (75, 79.99),
    "B-": (70, 74.99),
    "C+": (65, 69.99),
    "C": (60, 64.99),
    "C-": (55, 59.99),
    "D": (50, 54.99),
    "F": (0, 49.99),
}

# Score distribution buckets for histograms
SCORE_BUCKETS: List[tuple] = [
    (0, 20, "0-20"),
    (21, 40, "21-40"),
    (41, 60, "41-60"),
    (61, 80, "61-80"),
    (81, 100, "81-100"),
]

# Coefficient of variation thresholds
CV_THRESHOLDS: Dict[str, float] = {
    "low": 15.0,      # Very consistent scoring
    "moderate": 25.0,  # Normal variation
    "high": 35.0,      # High variation (needs review)
}

# Historical trend classification thresholds
TREND_SLOPE_THRESHOLDS: Dict[str, float] = {
    "improving": 0.5,   # Slope > 0.5 points per event
    "declining": -0.5,  # Slope < -0.5 points per event
    # Between -0.5 and 0.5 is "stable"
}


# =============================================================================
# TIME ANALYSIS CONFIGURATION
# =============================================================================

# Days of the week for heatmap (0 = Monday)
DAYS_OF_WEEK: List[str] = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

# Hours of the day for heatmap (24-hour format)
HOURS_OF_DAY: List[int] = list(range(24))

# Submission timing thresholds (in hours before deadline)
SUBMISSION_TIMING: Dict[str, float] = {
    "early": 24.0,        # More than 24 hours before deadline
    "on_time": 0.0,       # Before deadline
    "last_minute": 1.0,   # Less than 1 hour before deadline
}

# Peak activity detection threshold (percentage of total)
PEAK_ACTIVITY_THRESHOLD: float = 10.0


# =============================================================================
# SKILL RADAR CONFIGURATION
# =============================================================================

# Skill dimensions for radar chart
# Maps internal field names to display names and descriptions
SKILL_DIMENSIONS: Dict[str, Dict[str, str]] = {
    "design": {
        "name": "Design",
        "description": "UI/UX design quality, visual appeal, and user experience",
        "source_fields": ["aiResult.rubric.design", "pptAnalysis.design_score"],
        "max_score": 100,
    },
    "code_quality": {
        "name": "Code Quality",
        "description": "Code cleanliness, readability, and best practices",
        "source_fields": ["aiResult.pylint_score", "aiResult.score.code_quality"],
        "max_score": 10,  # Pylint uses 0-10 scale
    },
    "logic": {
        "name": "Logic",
        "description": "Problem-solving approach and algorithmic thinking",
        "source_fields": ["aiResult.logic.score", "aiResult.rubric.logic"],
        "max_score": 100,
    },
    "documentation": {
        "name": "Documentation",
        "description": "Code comments, README quality, and project documentation",
        "source_fields": ["aiResult.structure", "aiResult.rubric.documentation"],
        "max_score": 100,
    },
    "testing": {
        "name": "Testing",
        "description": "Test coverage and quality assurance practices",
        "source_fields": ["aiResult.rubric.testing", "aiResult.has_tests"],
        "max_score": 100,
    },
    "architecture": {
        "name": "Architecture",
        "description": "System design, scalability, and code organization",
        "source_fields": ["aiResult.rubric.architecture", "aiResult.relevance"],
        "max_score": 100,
    },
}

# Weights for computing overall score from dimensions
SKILL_DIMENSION_WEIGHTS: Dict[str, float] = {
    "design": 0.15,
    "code_quality": 0.25,
    "logic": 0.20,
    "documentation": 0.10,
    "testing": 0.15,
    "architecture": 0.15,
}

# Default score when a dimension cannot be calculated
DEFAULT_DIMENSION_SCORE: float = 50.0


# =============================================================================
# CONSISTENCY SCORE CONFIGURATION
# =============================================================================

# Weights for consistency score calculation
CONSISTENCY_WEIGHTS: Dict[str, float] = {
    "score_variance": 0.4,      # Lower variance = more consistent
    "percentile_stability": 0.3, # Stable percentile rankings
    "participation_rate": 0.3,   # Regular participation
}

# Thresholds for consistency classification
CONSISTENCY_THRESHOLDS: Dict[str, float] = {
    "highly_consistent": 80.0,
    "consistent": 60.0,
    "variable": 40.0,
    # Below 40 is "inconsistent"
}


# =============================================================================
# EXPORT CONFIGURATION
# =============================================================================

# All available columns for CSV export
EXPORTABLE_COLUMNS: Dict[str, Dict[str, str]] = {
    # Submission metadata
    "submission_id": {"type": "string", "description": "Unique submission ID"},
    "event_id": {"type": "string", "description": "Event identifier"},
    "team_id": {"type": "string", "description": "Team identifier"},
    "team_name": {"type": "string", "description": "Team display name"},
    "round_id": {"type": "string", "description": "Evaluation round (ppt/repo/viva)"},
    "submitted_at": {"type": "datetime", "description": "Submission timestamp"},
    
    # Scores
    "overall_score": {"type": "float", "description": "Overall evaluation score"},
    "final_score": {"type": "float", "description": "Final computed score"},
    "risk_score": {"type": "float", "description": "Risk assessment score"},
    "pylint_score": {"type": "float", "description": "Code quality score (0-10)"},
    
    # PPT-specific
    "ppt_clarity_score": {"type": "float", "description": "PPT clarity score"},
    "ppt_design_score": {"type": "float", "description": "PPT design score"},
    "ppt_storytelling_score": {"type": "float", "description": "PPT storytelling score"},
    
    # Repo-specific
    "repo_url": {"type": "string", "description": "GitHub repository URL"},
    "files_analyzed": {"type": "int", "description": "Number of files analyzed"},
    "plagiarism_percentage": {"type": "float", "description": "Detected plagiarism %"},
    
    # AI Feedback
    "llm_feedback": {"type": "text", "description": "AI-generated feedback"},
    "mentor_summary": {"type": "text", "description": "Mentor summary markdown"},
    "rewrite_suggestions": {"type": "text", "description": "Code improvement suggestions"},
    
    # Computed
    "percentile_rank": {"type": "float", "description": "Percentile within event"},
    "grade": {"type": "string", "description": "Letter grade (A+, A, B, etc.)"},
    "theme": {"type": "string", "description": "Detected event theme"},
}

# Default columns for quick export
DEFAULT_EXPORT_COLUMNS: List[str] = [
    "team_name",
    "round_id",
    "overall_score",
    "final_score",
    "submitted_at",
    "grade",
]

# Maximum rows to export in a single request
MAX_EXPORT_ROWS: int = 10000

# Supported export formats
SUPPORTED_EXPORT_FORMATS: Set[str] = {"csv", "json"}


# =============================================================================
# API CONFIGURATION
# =============================================================================

# Rate limiting for analytics endpoints (requests per minute)
ANALYTICS_RATE_LIMIT: int = 30

# Cache TTL for analytics results (seconds)
ANALYTICS_CACHE_TTL: int = 300  # 5 minutes

# Maximum events to include in historical trends
MAX_EVENTS_FOR_TRENDS: int = 50

# Pagination defaults
DEFAULT_PAGE_SIZE: int = 20
MAX_PAGE_SIZE: int = 100


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_grade_for_score(score: float) -> str:
    """
    Get letter grade for a numerical score.
    
    Args:
        score: Numerical score (0-100).
    
    Returns:
        Letter grade string (A+, A, B, etc.).
    
    Example:
        >>> get_grade_for_score(87.5)
        'A-'
    """
    for grade, (min_score, max_score) in SCORE_GRADES.items():
        if min_score <= score <= max_score:
            return grade
    return "F"


def get_score_bucket(score: float) -> str:
    """
    Get the histogram bucket for a score.
    
    Args:
        score: Numerical score (0-100).
    
    Returns:
        Bucket label string (e.g., "61-80").
    
    Example:
        >>> get_score_bucket(75.5)
        '61-80'
    """
    for min_val, max_val, label in SCORE_BUCKETS:
        if min_val <= score <= max_val:
            return label
    return "0-20"


def classify_trend(slope: float) -> str:
    """
    Classify a trend based on its slope.
    
    Args:
        slope: Linear regression slope of scores over time.
    
    Returns:
        Trend classification ('improving', 'declining', or 'stable').
    
    Example:
        >>> classify_trend(1.2)
        'improving'
    """
    if slope > TREND_SLOPE_THRESHOLDS["improving"]:
        return "improving"
    elif slope < TREND_SLOPE_THRESHOLDS["declining"]:
        return "declining"
    else:
        return "stable"


def classify_consistency(score: float) -> str:
    """
    Classify consistency based on score.
    
    Args:
        score: Consistency score (0-100).
    
    Returns:
        Consistency classification string.
    
    Example:
        >>> classify_consistency(85.0)
        'highly_consistent'
    """
    if score >= CONSISTENCY_THRESHOLDS["highly_consistent"]:
        return "highly_consistent"
    elif score >= CONSISTENCY_THRESHOLDS["consistent"]:
        return "consistent"
    elif score >= CONSISTENCY_THRESHOLDS["variable"]:
        return "variable"
    else:
        return "inconsistent"
