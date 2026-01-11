"""
Organizer Analytics Module
==========================

This module provides analytics functions specifically designed for
event organizers. These functions analyze evaluation data across
teams and submissions to provide insights into event performance,
AI calibration, and submission patterns.

Functions:
    - calculate_calibration_metrics: Assess AI evaluation consistency
    - analyze_themes: Group performance by hackathon theme
    - get_historical_trends: Compare across multiple events
    - analyze_submission_patterns: Study submission timing behavior
    - detect_scoring_anomalies: Find unusual scores

Dependencies:
    - numpy: Statistical calculations
    - Motor: Async MongoDB operations
    - Pydantic models from analytics.models
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import statistics

import numpy as np
from bson import ObjectId

from analytics.models import (
    CalibrationMetrics,
    ScoringAnomaly,
    ThemeAnalysis,
    ThemePerformance,
    SubmissionPattern,
    HeatmapCell,
    HistoricalTrend,
    TrendDataPoint,
)
from analytics.exceptions import (
    InsufficientDataError,
    InvalidEventError,
    ThemeDetectionError,
    CalculationError,
)
from analytics.constants import (
    ANOMALY_Z_SCORE_THRESHOLD,
    MIN_SUBMISSIONS_FOR_CALIBRATION,
    MIN_EVENTS_FOR_TRENDS,
    COMPILED_THEME_PATTERNS,
    DEFAULT_THEME,
    THEME_CONFIDENCE_THRESHOLD,
    SCORE_BUCKETS,
    DAYS_OF_WEEK,
    HOURS_OF_DAY,
    SUBMISSION_TIMING,
    TREND_SLOPE_THRESHOLDS,
    classify_trend,
    get_score_bucket,
    get_grade_for_score,
    MAX_EVENTS_FOR_TRENDS,
)

# Module logger
logger = logging.getLogger(__name__)


# =============================================================================
# CALIBRATION METRICS
# =============================================================================

async def calculate_calibration_metrics(
    db: Any,
    event_id: str,
    round_filter: Optional[str] = None
) -> CalibrationMetrics:
    """
    Calculate AI evaluation calibration metrics for an event.
    
    Analyzes the consistency and distribution of AI-generated scores
    to help organizers assess evaluation quality. Includes statistical
    measures like variance, standard deviation, and anomaly detection.
    
    Args:
        db: MongoDB database instance from Motor.
        event_id: The event identifier to analyze.
        round_filter: Optional filter for specific round ('ppt', 'repo', 'viva').
            If None, analyzes all rounds.
    
    Returns:
        CalibrationMetrics: Comprehensive calibration statistics.
    
    Raises:
        InvalidEventError: If event_id is invalid or event not found.
        InsufficientDataError: If fewer than MIN_SUBMISSIONS_FOR_CALIBRATION
            submissions exist.
    
    Example:
        >>> metrics = await calculate_calibration_metrics(db, "event123")
        >>> print(f"Mean score: {metrics.mean_score:.2f}")
        >>> print(f"Anomalies found: {len(metrics.anomalies)}")
    """
    logger.info(f"Calculating calibration metrics for event: {event_id}")
    
    # Validate event exists
    try:
        events_collection = db["events"]
        event = await events_collection.find_one({"_id": ObjectId(event_id)})
    except Exception as e:
        logger.error(f"Invalid event ID format: {event_id}, error: {e}")
        raise InvalidEventError(
            message="Invalid event ID format",
            event_id=event_id,
            reason="invalid_format"
        )
    
    if not event:
        logger.warning(f"Event not found: {event_id}")
        raise InvalidEventError(
            message="Event not found",
            event_id=event_id,
            reason="not_found"
        )
    
    # Build query for submissions
    submissions_collection = db["submissions"]
    query: Dict[str, Any] = {"eventId": event_id}
    
    if round_filter:
        query["roundId"] = round_filter
        logger.debug(f"Filtering by round: {round_filter}")
    
    # Fetch all submissions for the event
    cursor = submissions_collection.find(query)
    submissions = await cursor.to_list(None)
    
    logger.info(f"Found {len(submissions)} submissions for event {event_id}")
    
    if len(submissions) < MIN_SUBMISSIONS_FOR_CALIBRATION:
        raise InsufficientDataError(
            message=f"Need at least {MIN_SUBMISSIONS_FOR_CALIBRATION} submissions for calibration analysis",
            required_count=MIN_SUBMISSIONS_FOR_CALIBRATION,
            actual_count=len(submissions)
        )
    
    # Extract scores from submissions
    scores_with_meta = await _extract_scores_with_metadata(db, submissions)
    scores = [item["score"] for item in scores_with_meta if item["score"] is not None]
    
    if len(scores) < MIN_SUBMISSIONS_FOR_CALIBRATION:
        raise InsufficientDataError(
            message="Not enough valid scores for calibration",
            required_count=MIN_SUBMISSIONS_FOR_CALIBRATION,
            actual_count=len(scores)
        )
    
    # Calculate core statistics
    mean_score = statistics.mean(scores)
    median_score = statistics.median(scores)
    std_deviation = statistics.stdev(scores) if len(scores) > 1 else 0.0
    variance = statistics.variance(scores) if len(scores) > 1 else 0.0
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    
    # Coefficient of variation (handle zero mean)
    cv = (std_deviation / mean_score * 100) if mean_score > 0 else 0.0
    
    # Interquartile range
    sorted_scores = sorted(scores)
    q1_idx = len(sorted_scores) // 4
    q3_idx = (3 * len(sorted_scores)) // 4
    q1 = sorted_scores[q1_idx] if q1_idx < len(sorted_scores) else 0
    q3 = sorted_scores[q3_idx] if q3_idx < len(sorted_scores) else 0
    iqr = q3 - q1
    
    logger.debug(f"Stats - Mean: {mean_score:.2f}, Std: {std_deviation:.2f}, CV: {cv:.2f}%")
    
    # Detect anomalies
    anomalies = await _detect_anomalies(
        scores_with_meta, mean_score, std_deviation
    )
    
    anomaly_rate = (len(anomalies) / len(scores)) * 100 if scores else 0.0
    
    # Calculate per-round breakdown
    by_round = await _calculate_per_round_metrics(submissions)
    
    logger.info(
        f"Calibration complete for event {event_id}: "
        f"mean={mean_score:.2f}, std={std_deviation:.2f}, anomalies={len(anomalies)}"
    )
    
    return CalibrationMetrics(
        event_id=event_id,
        total_submissions=len(submissions),
        mean_score=round(mean_score, 2),
        median_score=round(median_score, 2),
        std_deviation=round(std_deviation, 2),
        variance=round(variance, 2),
        min_score=round(min_score, 2),
        max_score=round(max_score, 2),
        score_range=round(score_range, 2),
        coefficient_of_variation=round(cv, 2),
        interquartile_range=round(iqr, 2),
        anomalies=anomalies,
        anomaly_rate=round(anomaly_rate, 2),
        by_round=by_round,
        calculated_at=datetime.utcnow()
    )


async def _extract_scores_with_metadata(
    db: Any,
    submissions: List[Dict]
) -> List[Dict[str, Any]]:
    """
    Extract scores and metadata from submissions.
    
    Handles various score field locations in the submission schema
    and enriches with team information.
    
    Args:
        db: MongoDB database instance.
        submissions: List of submission documents.
    
    Returns:
        List of dicts with score, submission_id, team_id, team_name, round_id.
    """
    teams_collection = db["teams"]
    results = []
    
    for sub in submissions:
        # Try multiple possible score locations
        score = None
        
        # Check aiResult.final_score first
        if "aiResult" in sub and sub["aiResult"]:
            ai_result = sub["aiResult"]
            if isinstance(ai_result, dict):
                score = ai_result.get("final_score")
                if score is None:
                    # Try nested score object
                    score_obj = ai_result.get("score", {})
                    if isinstance(score_obj, dict):
                        score = score_obj.get("overall_score")
        
        # Fallback to top-level score fields
        if score is None:
            score = sub.get("score") or sub.get("finalScore") or sub.get("overall_score")
        
        # Get team information
        team_id = sub.get("teamId", "")
        team_name = "Unknown Team"
        
        if team_id:
            try:
                team = await teams_collection.find_one({"_id": ObjectId(team_id)})
                if team:
                    team_name = team.get("teamName", "Unknown Team")
            except Exception:
                # Handle string team ID
                team = await teams_collection.find_one({"teamId": team_id})
                if team:
                    team_name = team.get("teamName", "Unknown Team")
        
        results.append({
            "score": float(score) if score is not None else None,
            "submission_id": str(sub.get("_id", "")),
            "team_id": str(team_id),
            "team_name": team_name,
            "round_id": sub.get("roundId", "unknown"),
        })
    
    return results


async def _detect_anomalies(
    scores_with_meta: List[Dict[str, Any]],
    mean_score: float,
    std_deviation: float
) -> List[ScoringAnomaly]:
    """
    Detect scoring anomalies using z-score analysis.
    
    Identifies submissions with scores significantly above or below
    the mean, indicating potential evaluation inconsistencies.
    
    Args:
        scores_with_meta: List of score data with metadata.
        mean_score: Calculated mean score.
        std_deviation: Calculated standard deviation.
    
    Returns:
        List of ScoringAnomaly objects for detected anomalies.
    """
    anomalies = []
    
    # Cannot detect anomalies with zero std deviation
    if std_deviation == 0:
        logger.debug("Standard deviation is zero, skipping anomaly detection")
        return anomalies
    
    for item in scores_with_meta:
        score = item["score"]
        if score is None:
            continue
        
        # Calculate z-score
        z_score = (score - mean_score) / std_deviation
        
        if abs(z_score) > ANOMALY_Z_SCORE_THRESHOLD:
            # Classify anomaly type
            if z_score > 0:
                anomaly_type = "high"
            else:
                anomaly_type = "low"
            
            anomaly = ScoringAnomaly(
                submission_id=item["submission_id"],
                team_id=item["team_id"],
                team_name=item["team_name"],
                score=round(score, 2),
                z_score=round(z_score, 2),
                round_id=item["round_id"],
                anomaly_type=anomaly_type,
                detected_at=datetime.utcnow()
            )
            anomalies.append(anomaly)
            
            logger.debug(
                f"Anomaly detected: team={item['team_name']}, "
                f"score={score:.2f}, z-score={z_score:.2f}"
            )
    
    return anomalies


async def _calculate_per_round_metrics(
    submissions: List[Dict]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistics broken down by evaluation round.
    
    Args:
        submissions: List of submission documents.
    
    Returns:
        Dict mapping round_id to statistics dict.
    """
    rounds: Dict[str, List[float]] = defaultdict(list)
    
    for sub in submissions:
        round_id = sub.get("roundId", "unknown")
        
        # Extract score (same logic as main extraction)
        score = None
        if "aiResult" in sub and sub["aiResult"]:
            ai_result = sub["aiResult"]
            if isinstance(ai_result, dict):
                score = ai_result.get("final_score")
                if score is None:
                    score_obj = ai_result.get("score", {})
                    if isinstance(score_obj, dict):
                        score = score_obj.get("overall_score")
        
        if score is None:
            score = sub.get("score") or sub.get("finalScore")
        
        if score is not None:
            rounds[round_id].append(float(score))
    
    # Calculate stats for each round
    result = {}
    for round_id, scores in rounds.items():
        if scores:
            result[round_id] = {
                "count": len(scores),
                "mean": round(statistics.mean(scores), 2),
                "median": round(statistics.median(scores), 2),
                "std_dev": round(statistics.stdev(scores), 2) if len(scores) > 1 else 0.0,
                "min": round(min(scores), 2),
                "max": round(max(scores), 2),
            }
    
    return result


# =============================================================================
# THEME ANALYSIS
# =============================================================================

async def analyze_themes(
    db: Any,
    event_id: str
) -> ThemeAnalysis:
    """
    Analyze submission performance grouped by detected theme.
    
    Uses regex pattern matching on event descriptions to classify
    submissions into themes (AI/ML, Web3, IoT, etc.) and calculates
    performance metrics for each theme.
    
    Args:
        db: MongoDB database instance from Motor.
        event_id: The event identifier to analyze.
    
    Returns:
        ThemeAnalysis: Theme-wise performance breakdown.
    
    Raises:
        InvalidEventError: If event_id is invalid or not found.
        InsufficientDataError: If no submissions found.
    
    Example:
        >>> analysis = await analyze_themes(db, "event123")
        >>> print(f"Strongest theme: {analysis.strongest_theme}")
        >>> for theme in analysis.themes:
        ...     print(f"{theme.theme}: {theme.avg_score:.2f}")
    """
    logger.info(f"Analyzing themes for event: {event_id}")
    
    # Validate event and get description for theme detection
    events_collection = db["events"]
    try:
        event = await events_collection.find_one({"_id": ObjectId(event_id)})
    except Exception as e:
        logger.error(f"Invalid event ID: {event_id}, error: {e}")
        raise InvalidEventError(
            message="Invalid event ID format",
            event_id=event_id,
            reason="invalid_format"
        )
    
    if not event:
        raise InvalidEventError(
            message="Event not found",
            event_id=event_id,
            reason="not_found"
        )
    
    # Detect theme from event description
    event_text = f"{event.get('name', '')} {event.get('description', '')} {event.get('summary', '')}"
    event_theme = _detect_theme_from_text(event_text)
    
    logger.info(f"Detected event theme: {event_theme}")
    
    # Fetch submissions and teams
    submissions_collection = db["submissions"]
    teams_collection = db["teams"]
    
    submissions = await submissions_collection.find({"eventId": event_id}).to_list(None)
    
    if not submissions:
        raise InsufficientDataError(
            message="No submissions found for this event",
            required_count=1,
            actual_count=0
        )
    
    # Group submissions by team and detect per-team themes if applicable
    # For now, we'll use the event theme for all submissions
    theme_data: Dict[str, List[Dict]] = defaultdict(list)
    unclassified_count = 0
    
    for sub in submissions:
        # Extract score
        score = None
        if "aiResult" in sub and sub["aiResult"]:
            ai_result = sub["aiResult"]
            if isinstance(ai_result, dict):
                score = ai_result.get("final_score")
                if score is None:
                    score_obj = ai_result.get("score", {})
                    if isinstance(score_obj, dict):
                        score = score_obj.get("overall_score")
        
        if score is None:
            score = sub.get("score") or sub.get("finalScore")
        
        if score is None:
            unclassified_count += 1
            continue
        
        # Get team info for optional team-specific theme detection
        team_id = sub.get("teamId", "")
        team_name = "Unknown"
        team_theme = event_theme  # Default to event theme
        
        if team_id:
            try:
                team = await teams_collection.find_one({"_id": ObjectId(team_id)})
                if not team:
                    team = await teams_collection.find_one({"teamId": team_id})
                if team:
                    team_name = team.get("teamName", "Unknown")
                    # Could detect team-specific theme from project description if available
            except Exception:
                pass
        
        theme_data[team_theme].append({
            "score": float(score),
            "team_id": str(team_id),
            "team_name": team_name,
            "submission_id": str(sub.get("_id", "")),
        })
    
    # Calculate per-theme metrics
    themes_list: List[ThemePerformance] = []
    
    for theme, data_list in theme_data.items():
        scores = [d["score"] for d in data_list]
        
        if not scores:
            continue
        
        avg_score = statistics.mean(scores)
        median_score = statistics.median(scores)
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0
        
        # Find top team
        sorted_data = sorted(data_list, key=lambda x: x["score"], reverse=True)
        top_team = None
        if sorted_data:
            top = sorted_data[0]
            top_team = {
                "team_name": top["team_name"],
                "team_id": top["team_id"],
                "score": top["score"],
            }
        
        # Score distribution
        distribution = defaultdict(int)
        for score in scores:
            bucket = get_score_bucket(score)
            distribution[bucket] += 1
        
        theme_perf = ThemePerformance(
            theme=theme,
            submission_count=len(scores),
            avg_score=round(avg_score, 2),
            median_score=round(median_score, 2),
            std_deviation=round(std_dev, 2),
            min_score=round(min(scores), 2),
            max_score=round(max(scores), 2),
            top_team=top_team,
            score_distribution=dict(distribution),
        )
        themes_list.append(theme_perf)
    
    # Sort themes by average score
    themes_list.sort(key=lambda x: x.avg_score, reverse=True)
    
    # Determine strongest and weakest themes
    strongest_theme = themes_list[0].theme if themes_list else None
    weakest_theme = themes_list[-1].theme if themes_list else None
    
    logger.info(
        f"Theme analysis complete: {len(themes_list)} themes detected, "
        f"strongest={strongest_theme}, weakest={weakest_theme}"
    )
    
    return ThemeAnalysis(
        event_id=event_id,
        themes_detected=len(themes_list),
        themes=themes_list,
        strongest_theme=strongest_theme,
        weakest_theme=weakest_theme,
        unclassified_count=unclassified_count,
        analysis_timestamp=datetime.utcnow()
    )


def _detect_theme_from_text(text: str) -> str:
    """
    Detect hackathon theme from text using pattern matching.
    
    Analyzes text against predefined keyword patterns for each theme.
    Returns the theme with the highest match count above the threshold.
    
    Args:
        text: Text to analyze (event name, description, etc.).
    
    Returns:
        Detected theme name or DEFAULT_THEME if no match.
    """
    if not text:
        return DEFAULT_THEME
    
    text_lower = text.lower()
    theme_scores: Dict[str, int] = defaultdict(int)
    
    for theme, patterns in COMPILED_THEME_PATTERNS.items():
        for pattern in patterns:
            matches = pattern.findall(text_lower)
            theme_scores[theme] += len(matches)
    
    # Find theme with highest score
    if theme_scores:
        best_theme = max(theme_scores.items(), key=lambda x: x[1])
        if best_theme[1] > 0:
            logger.debug(f"Theme detection scores: {dict(theme_scores)}")
            return best_theme[0]
    
    return DEFAULT_THEME


# =============================================================================
# HISTORICAL TRENDS
# =============================================================================

async def get_historical_trends(
    db: Any,
    organizer_id: str,
    scope: str = "organizer",
    current_event_id: Optional[str] = None
) -> HistoricalTrend:
    """
    Analyze historical trends across multiple events.
    
    Provides trend analysis over time to help organizers understand
    how their events perform compared to historical averages.
    
    Args:
        db: MongoDB database instance from Motor.
        organizer_id: The organizer's user ID.
        scope: Analysis scope - 'organizer' for same organizer's events,
            'global' for all events in the system.
        current_event_id: Optional current event ID for comparison.
    
    Returns:
        HistoricalTrend: Trend analysis with data points and comparisons.
    
    Raises:
        InsufficientDataError: If fewer than MIN_EVENTS_FOR_TRENDS events exist.
    
    Example:
        >>> trends = await get_historical_trends(db, "user123", scope="organizer")
        >>> print(f"Score trend: {trends.score_trend}")
        >>> print(f"Trend slope: {trends.trend_slope:.2f}")
    """
    logger.info(f"Generating historical trends: scope={scope}, organizer={organizer_id}")
    
    events_collection = db["events"]
    submissions_collection = db["submissions"]
    teams_collection = db["teams"]
    
    # Build query based on scope
    if scope == "organizer":
        events_query = {"organizerId": ObjectId(organizer_id)}
    else:  # global
        events_query = {}
    
    # Fetch events, sorted by date
    events_cursor = events_collection.find(events_query).sort("date", 1)
    events = await events_cursor.to_list(MAX_EVENTS_FOR_TRENDS)
    
    if len(events) < MIN_EVENTS_FOR_TRENDS:
        raise InsufficientDataError(
            message=f"Need at least {MIN_EVENTS_FOR_TRENDS} events for trend analysis",
            required_count=MIN_EVENTS_FOR_TRENDS,
            actual_count=len(events)
        )
    
    logger.info(f"Analyzing trends across {len(events)} events")
    
    # Calculate metrics for each event
    data_points: List[TrendDataPoint] = []
    all_avg_scores: List[float] = []
    
    for event in events:
        event_id = str(event["_id"])
        event_name = event.get("name", "Unknown Event")
        
        # Parse event date
        event_date_str = event.get("date", "")
        try:
            event_date = datetime.fromisoformat(event_date_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            event_date = datetime.utcnow()
        
        # Get submissions for this event
        submissions = await submissions_collection.find(
            {"eventId": event_id}
        ).to_list(None)
        
        # Get team count
        team_count = await teams_collection.count_documents({"eventId": event_id})
        
        # Calculate average score
        scores = []
        for sub in submissions:
            score = None
            if "aiResult" in sub and sub["aiResult"]:
                ai_result = sub["aiResult"]
                if isinstance(ai_result, dict):
                    score = ai_result.get("final_score")
                    if score is None:
                        score_obj = ai_result.get("score", {})
                        if isinstance(score_obj, dict):
                            score = score_obj.get("overall_score")
            
            if score is None:
                score = sub.get("score") or sub.get("finalScore")
            
            if score is not None:
                scores.append(float(score))
        
        avg_score = statistics.mean(scores) if scores else 0.0
        all_avg_scores.append(avg_score)
        
        # Calculate completion rate
        submission_count = len(submissions)
        completion_rate = (submission_count / team_count * 100) if team_count > 0 else 0.0
        
        data_point = TrendDataPoint(
            event_id=event_id,
            event_name=event_name,
            event_date=event_date,
            avg_score=round(avg_score, 2),
            submission_count=submission_count,
            team_count=team_count,
            completion_rate=round(min(completion_rate, 100), 2),
        )
        data_points.append(data_point)
    
    # Calculate overall statistics
    overall_avg = statistics.mean(all_avg_scores) if all_avg_scores else 0.0
    
    # Calculate trend slope using linear regression
    if len(all_avg_scores) >= 2:
        x = np.arange(len(all_avg_scores))
        y = np.array(all_avg_scores)
        # Simple linear regression: y = mx + b
        slope = np.polyfit(x, y, 1)[0]
    else:
        slope = 0.0
    
    # Classify trend
    score_trend = classify_trend(slope)
    
    # Find best performing event
    best_event = max(data_points, key=lambda x: x.avg_score) if data_points else None
    
    # Compare current event to historical
    current_comparison = None
    if current_event_id:
        current_point = next(
            (p for p in data_points if p.event_id == current_event_id),
            None
        )
        if current_point:
            diff_from_avg = current_point.avg_score - overall_avg
            # Calculate percentile
            sorted_scores = sorted(all_avg_scores)
            percentile = (
                sum(1 for s in sorted_scores if s <= current_point.avg_score) 
                / len(sorted_scores) * 100
            ) if sorted_scores else 0
            
            current_comparison = {
                "current_score": current_point.avg_score,
                "historical_avg": round(overall_avg, 2),
                "difference": round(diff_from_avg, 2),
                "percentile": round(percentile, 2),
                "better_than_average": diff_from_avg > 0,
            }
    
    logger.info(
        f"Trend analysis complete: overall_avg={overall_avg:.2f}, "
        f"trend={score_trend}, slope={slope:.3f}"
    )
    
    return HistoricalTrend(
        scope=scope,
        organizer_id=organizer_id if scope == "organizer" else None,
        total_events=len(data_points),
        data_points=data_points,
        overall_avg_score=round(overall_avg, 2),
        score_trend=score_trend,
        trend_slope=round(slope, 3),
        best_performing_event=best_event,
        current_vs_historical=current_comparison,
        generated_at=datetime.utcnow()
    )


# =============================================================================
# SUBMISSION PATTERNS
# =============================================================================

async def analyze_submission_patterns(
    db: Any,
    event_id: str
) -> SubmissionPattern:
    """
    Analyze submission timing patterns for an event.
    
    Generates heatmap data showing when submissions occur and
    calculates metrics around early vs. last-minute submissions.
    
    Args:
        db: MongoDB database instance from Motor.
        event_id: The event identifier to analyze.
    
    Returns:
        SubmissionPattern: Timing analysis with heatmap and metrics.
    
    Raises:
        InvalidEventError: If event_id is invalid or not found.
        InsufficientDataError: If no submissions found.
    
    Example:
        >>> patterns = await analyze_submission_patterns(db, "event123")
        >>> print(f"Peak submission hour: {patterns.peak_hour}")
        >>> print(f"Last-minute submissions: {patterns.last_minute_ratio:.1f}%")
    """
    logger.info(f"Analyzing submission patterns for event: {event_id}")
    
    # Validate event and get deadline
    events_collection = db["events"]
    try:
        event = await events_collection.find_one({"_id": ObjectId(event_id)})
    except Exception as e:
        logger.error(f"Invalid event ID: {event_id}, error: {e}")
        raise InvalidEventError(
            message="Invalid event ID format",
            event_id=event_id,
            reason="invalid_format"
        )
    
    if not event:
        raise InvalidEventError(
            message="Event not found",
            event_id=event_id,
            reason="not_found"
        )
    
    # Get deadline (use registration deadline or event date)
    deadline_str = event.get("registrationDeadline") or event.get("date")
    deadline = None
    if deadline_str:
        try:
            deadline = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            deadline = None
    
    # Fetch submissions
    submissions_collection = db["submissions"]
    submissions = await submissions_collection.find(
        {"eventId": event_id}
    ).to_list(None)
    
    if not submissions:
        raise InsufficientDataError(
            message="No submissions found for this event",
            required_count=1,
            actual_count=0
        )
    
    # Extract submission timestamps
    timestamps: List[datetime] = []
    for sub in submissions:
        submitted_at = sub.get("submittedAt")
        if submitted_at:
            if isinstance(submitted_at, str):
                try:
                    submitted_at = datetime.fromisoformat(submitted_at.replace("Z", "+00:00"))
                except ValueError:
                    continue
            timestamps.append(submitted_at)
    
    if not timestamps:
        raise InsufficientDataError(
            message="No valid submission timestamps found",
            required_count=1,
            actual_count=0
        )
    
    logger.info(f"Analyzing {len(timestamps)} submission timestamps")
    
    # Build heatmap data (day of week × hour of day)
    heatmap_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    
    for ts in timestamps:
        day = ts.weekday()  # 0 = Monday
        hour = ts.hour
        heatmap_counts[(day, hour)] += 1
    
    # Convert to heatmap cells with percentages
    total_submissions = len(timestamps)
    heatmap: List[HeatmapCell] = []
    
    for day in range(7):
        for hour in range(24):
            count = heatmap_counts.get((day, hour), 0)
            percentage = (count / total_submissions * 100) if total_submissions > 0 else 0
            heatmap.append(HeatmapCell(
                day=day,
                hour=hour,
                count=count,
                percentage=round(percentage, 2),
            ))
    
    # Find peak hour and day
    hour_counts = defaultdict(int)
    day_counts = defaultdict(int)
    
    for ts in timestamps:
        hour_counts[ts.hour] += 1
        day_counts[ts.weekday()] += 1
    
    peak_hour = max(hour_counts.items(), key=lambda x: x[1])[0] if hour_counts else 12
    peak_day = max(day_counts.items(), key=lambda x: x[1])[0] if day_counts else 0
    
    # Calculate submission timing ratios
    early_count = 0
    on_time_count = 0
    last_minute_count = 0
    time_before_deadline_hours: List[float] = []
    
    if deadline:
        for ts in timestamps:
            hours_before = (deadline - ts).total_seconds() / 3600
            time_before_deadline_hours.append(hours_before)
            
            if hours_before > SUBMISSION_TIMING["early"]:
                early_count += 1
            elif hours_before >= SUBMISSION_TIMING["on_time"]:
                on_time_count += 1
                if hours_before <= SUBMISSION_TIMING["last_minute"]:
                    last_minute_count += 1
    else:
        # No deadline available, use 100% on-time
        on_time_count = len(timestamps)
    
    early_ratio = (early_count / total_submissions * 100) if total_submissions > 0 else 0
    on_time_ratio = (on_time_count / total_submissions * 100) if total_submissions > 0 else 100
    last_minute_ratio = (last_minute_count / total_submissions * 100) if total_submissions > 0 else 0
    
    avg_time_before = statistics.mean(time_before_deadline_hours) if time_before_deadline_hours else 0
    
    # Calculate submission velocity in final 24 hours
    final_24h_count = 0
    if deadline:
        cutoff = deadline - timedelta(hours=24)
        final_24h_count = sum(1 for ts in timestamps if ts >= cutoff)
    
    submission_velocity = final_24h_count / 24 if final_24h_count > 0 else 0
    
    # Get first and last submissions
    sorted_timestamps = sorted(timestamps)
    first_submission = sorted_timestamps[0] if sorted_timestamps else None
    last_submission = sorted_timestamps[-1] if sorted_timestamps else None
    
    logger.info(
        f"Pattern analysis complete: peak_hour={peak_hour}, "
        f"early={early_ratio:.1f}%, last_minute={last_minute_ratio:.1f}%"
    )
    
    return SubmissionPattern(
        event_id=event_id,
        total_submissions=total_submissions,
        heatmap=heatmap,
        peak_hour=peak_hour,
        peak_day=peak_day,
        early_submission_ratio=round(early_ratio, 2),
        on_time_ratio=round(on_time_ratio, 2),
        last_minute_ratio=round(last_minute_ratio, 2),
        avg_time_before_deadline_hours=round(avg_time_before, 2),
        submission_velocity=round(submission_velocity, 2),
        deadline=deadline,
        first_submission=first_submission,
        last_submission=last_submission,
    )


# =============================================================================
# SCORING ANOMALY DETECTION (Standalone)
# =============================================================================

async def detect_scoring_anomalies(
    db: Any,
    event_id: str,
    z_score_threshold: float = ANOMALY_Z_SCORE_THRESHOLD
) -> List[ScoringAnomaly]:
    """
    Detect scoring anomalies in event submissions.
    
    Standalone function to identify submissions with unusually high
    or low scores that may warrant review.
    
    Args:
        db: MongoDB database instance from Motor.
        event_id: The event identifier to analyze.
        z_score_threshold: Z-score threshold for flagging anomalies.
            Default is ANOMALY_Z_SCORE_THRESHOLD (2.0).
    
    Returns:
        List of ScoringAnomaly objects for detected anomalies.
    
    Raises:
        InvalidEventError: If event not found.
        InsufficientDataError: If insufficient data for analysis.
    
    Example:
        >>> anomalies = await detect_scoring_anomalies(db, "event123", z_score_threshold=2.5)
        >>> for a in anomalies:
        ...     print(f"{a.team_name}: {a.score} (z={a.z_score:.2f})")
    """
    logger.info(f"Detecting scoring anomalies for event: {event_id}")
    
    # Use calibration metrics calculation which includes anomaly detection
    try:
        metrics = await calculate_calibration_metrics(db, event_id)
        return metrics.anomalies
    except (InvalidEventError, InsufficientDataError):
        raise
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        raise CalculationError(
            message=f"Failed to detect anomalies: {str(e)}",
            operation="anomaly_detection",
            input_values={"event_id": event_id, "threshold": z_score_threshold}
        )
