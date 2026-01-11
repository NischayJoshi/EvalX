"""
EvalX Analytics Module
======================

This module provides comprehensive analytics capabilities for the EvalX
hackathon evaluation platform. It includes tools for both organizers
and participants to gain insights from evaluation data.

Modules:
    - models: Pydantic data models for analytics responses
    - exceptions: Custom exceptions for error handling
    - constants: Configuration constants and thresholds
    - organizer_analytics: Analytics functions for event organizers
    - participant_analytics: Analytics functions for participants/developers
    - aggregation_pipelines: MongoDB aggregation pipeline builders
    - export_service: CSV/data export functionality

Usage:
    from analytics.organizer_analytics import calculate_calibration_metrics
    from analytics.participant_analytics import generate_skill_radar
    from analytics.models import CalibrationMetrics, SkillRadarData

Author: EvalX Team
Version: 1.0.0
"""

import logging

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Module version
__version__ = "1.0.0"

# Public API exports
__all__ = [
    # Models
    "CalibrationMetrics",
    "ThemeAnalysis",
    "ThemePerformance",
    "SubmissionPattern",
    "HeatmapCell",
    "HistoricalTrend",
    "TrendDataPoint",
    "SkillRadarData",
    "SkillDimension",
    "PeerComparison",
    "ProgressTimeline",
    "ProgressEntry",
    "ExportConfig",
    "ScoringAnomaly",
    # Exceptions
    "AnalyticsError",
    "InsufficientDataError",
    "ExportError",
    "ThemeDetectionError",
    "InvalidEventError",
    # Core functions
    "calculate_calibration_metrics",
    "analyze_themes",
    "get_historical_trends",
    "analyze_submission_patterns",
    "detect_scoring_anomalies",
    "generate_skill_radar",
    "calculate_peer_comparison",
    "track_progress",
    # Export
    "ExportService",
    # Logger
    "logger",
]

# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    """Lazy import module components."""
    if name in [
        "CalibrationMetrics", "ThemeAnalysis", "ThemePerformance",
        "SubmissionPattern", "HeatmapCell", "HistoricalTrend",
        "TrendDataPoint", "SkillRadarData", "SkillDimension",
        "PeerComparison", "ProgressTimeline", "ProgressEntry",
        "ExportConfig", "ScoringAnomaly"
    ]:
        from analytics.models import (
            CalibrationMetrics, ThemeAnalysis, ThemePerformance,
            SubmissionPattern, HeatmapCell, HistoricalTrend,
            TrendDataPoint, SkillRadarData, SkillDimension,
            PeerComparison, ProgressTimeline, ProgressEntry,
            ExportConfig, ScoringAnomaly
        )
        return locals()[name]
    
    if name in [
        "AnalyticsError", "InsufficientDataError", "ExportError",
        "ThemeDetectionError", "InvalidEventError"
    ]:
        from analytics.exceptions import (
            AnalyticsError, InsufficientDataError, ExportError,
            ThemeDetectionError, InvalidEventError
        )
        return locals()[name]
    
    if name in [
        "calculate_calibration_metrics", "analyze_themes",
        "get_historical_trends", "analyze_submission_patterns",
        "detect_scoring_anomalies"
    ]:
        from analytics.organizer_analytics import (
            calculate_calibration_metrics, analyze_themes,
            get_historical_trends, analyze_submission_patterns,
            detect_scoring_anomalies
        )
        return locals()[name]
    
    if name in [
        "generate_skill_radar", "calculate_peer_comparison", "track_progress"
    ]:
        from analytics.participant_analytics import (
            generate_skill_radar, calculate_peer_comparison, track_progress
        )
        return locals()[name]
    
    if name == "ExportService":
        from analytics.export_service import ExportService
        return ExportService
    
    raise AttributeError(f"module 'analytics' has no attribute '{name}'")
