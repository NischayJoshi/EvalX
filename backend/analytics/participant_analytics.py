"""
Participant Analytics Module
============================

This module provides analytics functions specifically designed for
hackathon participants and developers. These functions analyze individual
and team performance to provide personalized insights and comparisons.

Functions:
    - generate_skill_radar: Create skill radar chart data
    - calculate_peer_comparison: Compare to event peers (per-event percentile)
    - track_progress: Track improvement across multiple events

Dependencies:
    - numpy: Statistical calculations
    - Motor: Async MongoDB operations
    - Pydantic models from analytics.models
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict
import statistics

import numpy as np
from bson import ObjectId

from analytics.models import (
    SkillRadarData,
    SkillDimension,
    PeerComparison,
    ProgressTimeline,
    ProgressEntry,
)
from analytics.exceptions import (
    InsufficientDataError,
    InvalidEventError,
    InvalidTeamError,
    CalculationError,
)
from analytics.constants import (
    SKILL_DIMENSIONS,
    SKILL_DIMENSION_WEIGHTS,
    DEFAULT_DIMENSION_SCORE,
    MIN_SUBMISSIONS_FOR_PERCENTILE,
    MIN_EVENTS_FOR_TRENDS,
    CONSISTENCY_WEIGHTS,
    classify_consistency,
    classify_trend,
)

# Module logger
logger = logging.getLogger(__name__)


# =============================================================================
# SKILL RADAR GENERATION
# =============================================================================

async def generate_skill_radar(
    db: Any,
    event_id: str,
    user_id: str
) -> SkillRadarData:
    """
    Generate skill radar chart data for a participant's team.
    
    Maps AI evaluation results to six skill dimensions and normalizes
    scores to a 0-100 scale for radar chart visualization.
    
    Args:
        db: MongoDB database instance from Motor.
        event_id: The event identifier.
        user_id: The participant's user ID.
    
    Returns:
        SkillRadarData: Skill dimensions with scores and metadata.
    
    Raises:
        InvalidEventError: If event_id is invalid or not found.
        InvalidTeamError: If user is not part of any team in the event.
        InsufficientDataError: If no submissions found for the team.
    
    Example:
        >>> radar = await generate_skill_radar(db, "event123", "user456")
        >>> for dim in radar.dimensions:
        ...     print(f"{dim.name}: {dim.score:.0f}/100")
        >>> print(f"Strongest: {radar.strongest_skill}")
    """
    logger.info(f"Generating skill radar for user {user_id} in event {event_id}")
    
    # Validate event
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
    
    # Find user's team in this event
    teams_collection = db["teams"]
    team = await _find_user_team(teams_collection, event_id, user_id)
    
    if not team:
        raise InvalidTeamError(
            message="User is not part of any team in this event",
            team_id=None,
            event_id=event_id,
            reason="not_member"
        )
    
    team_id = str(team.get("_id", team.get("teamId", "")))
    team_name = team.get("teamName", "Unknown Team")
    
    logger.info(f"Found team {team_name} (ID: {team_id}) for user {user_id}")
    
    # Fetch team's submissions
    submissions_collection = db["submissions"]
    submissions = await submissions_collection.find({
        "eventId": event_id,
        "teamId": team_id
    }).to_list(None)
    
    # Also try with ObjectId team_id
    if not submissions:
        submissions = await submissions_collection.find({
            "eventId": event_id,
            "teamId": str(team.get("_id", ""))
        }).to_list(None)
    
    if not submissions:
        raise InsufficientDataError(
            message="No submissions found for your team",
            required_count=1,
            actual_count=0
        )
    
    logger.info(f"Found {len(submissions)} submissions for team {team_name}")
    
    # Extract skill scores from submissions
    skill_scores = await _extract_skill_scores(submissions)
    
    # Build skill dimensions
    dimensions: List[SkillDimension] = []
    
    for skill_key, skill_config in SKILL_DIMENSIONS.items():
        raw_score = skill_scores.get(skill_key, DEFAULT_DIMENSION_SCORE)
        max_score = float(skill_config.get("max_score", 100))
        
        # Normalize to 0-100 scale
        normalized_score = (raw_score / max_score * 100) if max_score > 0 else 0
        normalized_score = min(max(normalized_score, 0), 100)  # Clamp to 0-100
        
        dimension = SkillDimension(
            name=skill_config["name"],
            score=round(normalized_score, 2),
            raw_score=round(raw_score, 2),
            max_possible=max_score,
            description=skill_config["description"],
        )
        dimensions.append(dimension)
    
    # Calculate overall score using weights
    overall_score = 0.0
    for dim in dimensions:
        weight = SKILL_DIMENSION_WEIGHTS.get(dim.name.lower().replace(" ", "_"), 0.1)
        overall_score += dim.score * weight
    
    # Find strongest and weakest skills
    sorted_dims = sorted(dimensions, key=lambda x: x.score, reverse=True)
    strongest_skill = sorted_dims[0].name if sorted_dims else None
    weakest_skill = sorted_dims[-1].name if sorted_dims else None
    
    # Generate improvement suggestions based on weak areas
    improvement_suggestions = _generate_improvement_suggestions(dimensions)
    
    # Identify which rounds contributed
    based_on_rounds = list(set(sub.get("roundId", "unknown") for sub in submissions))
    
    logger.info(
        f"Skill radar generated: overall={overall_score:.2f}, "
        f"strongest={strongest_skill}, weakest={weakest_skill}"
    )
    
    return SkillRadarData(
        event_id=event_id,
        team_id=team_id,
        team_name=team_name,
        dimensions=dimensions,
        overall_score=round(overall_score, 2),
        strongest_skill=strongest_skill,
        weakest_skill=weakest_skill,
        improvement_suggestions=improvement_suggestions,
        based_on_rounds=based_on_rounds,
        generated_at=datetime.utcnow()
    )


async def _find_user_team(
    teams_collection: Any,
    event_id: str,
    user_id: str
) -> Optional[Dict]:
    """
    Find the team a user belongs to in an event.
    
    Args:
        teams_collection: MongoDB teams collection.
        event_id: Event identifier.
        user_id: User identifier.
    
    Returns:
        Team document if found, None otherwise.
    """
    # Search by member userId
    team = await teams_collection.find_one({
        "eventId": event_id,
        "members.userId": user_id
    })
    
    if team:
        return team
    
    # Also check if user is the leader
    team = await teams_collection.find_one({
        "eventId": event_id,
        "leaderId": user_id
    })
    
    return team


async def _extract_skill_scores(submissions: List[Dict]) -> Dict[str, float]:
    """
    Extract skill dimension scores from submission AI results.
    
    Maps various fields in aiResult to standardized skill dimensions.
    
    Args:
        submissions: List of submission documents.
    
    Returns:
        Dict mapping skill_key to extracted score.
    """
    scores: Dict[str, List[float]] = defaultdict(list)
    
    for sub in submissions:
        ai_result = sub.get("aiResult", {})
        if not ai_result or not isinstance(ai_result, dict):
            continue
        
        # Extract design score
        design_score = _safe_get_nested(ai_result, ["rubric", "design"])
        if design_score is None:
            design_score = _safe_get_nested(sub, ["pptAnalysis", "design_score"])
        if design_score is not None:
            scores["design"].append(float(design_score))
        
        # Extract code quality (Pylint score is 0-10)
        code_quality = ai_result.get("pylint_score")
        if code_quality is None:
            code_quality = _safe_get_nested(ai_result, ["score", "code_quality"])
        if code_quality is not None:
            scores["code_quality"].append(float(code_quality))
        
        # Extract logic score
        logic_score = _safe_get_nested(ai_result, ["logic", "score"])
        if logic_score is None:
            logic_score = _safe_get_nested(ai_result, ["rubric", "logic"])
        if logic_score is not None:
            scores["logic"].append(float(logic_score))
        
        # Extract documentation score (from structure or rubric)
        doc_score = _safe_get_nested(ai_result, ["rubric", "documentation"])
        if doc_score is None:
            # Estimate from structure analysis
            structure = ai_result.get("structure", "")
            if structure:
                # Simple heuristic: presence of README, comments, etc.
                doc_score = 60.0  # Base score
                if "readme" in str(structure).lower():
                    doc_score += 20.0
                if "comment" in str(structure).lower():
                    doc_score += 20.0
        if doc_score is not None:
            scores["documentation"].append(float(doc_score))
        
        # Extract testing score
        testing_score = _safe_get_nested(ai_result, ["rubric", "testing"])
        if testing_score is None:
            has_tests = ai_result.get("has_tests")
            if has_tests is True:
                testing_score = 80.0
            elif has_tests is False:
                testing_score = 20.0
        if testing_score is not None:
            scores["testing"].append(float(testing_score))
        
        # Extract architecture score
        arch_score = _safe_get_nested(ai_result, ["rubric", "architecture"])
        if arch_score is None:
            relevance = ai_result.get("relevance")
            if relevance is not None:
                arch_score = float(relevance)
        if arch_score is not None:
            scores["architecture"].append(float(arch_score))
    
    # Average scores for each dimension
    result = {}
    for skill_key, score_list in scores.items():
        if score_list:
            result[skill_key] = statistics.mean(score_list)
        else:
            result[skill_key] = DEFAULT_DIMENSION_SCORE
    
    return result


def _safe_get_nested(data: Dict, keys: List[str]) -> Optional[Any]:
    """
    Safely get nested dictionary value.
    
    Args:
        data: Dictionary to traverse.
        keys: List of keys to follow.
    
    Returns:
        Value at nested path or None if not found.
    """
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def _generate_improvement_suggestions(dimensions: List[SkillDimension]) -> List[str]:
    """
    Generate improvement suggestions based on skill scores.
    
    Args:
        dimensions: List of skill dimensions with scores.
    
    Returns:
        List of actionable improvement suggestions.
    """
    suggestions = []
    
    # Sort by score to find weak areas
    sorted_dims = sorted(dimensions, key=lambda x: x.score)
    
    for dim in sorted_dims[:2]:  # Focus on two weakest areas
        if dim.score < 50:
            severity = "significantly improve"
        elif dim.score < 70:
            severity = "improve"
        else:
            continue  # Skip if score is decent
        
        if dim.name == "Design":
            suggestions.append(
                f"Consider {severity}ing your UI/UX design. Focus on visual "
                "consistency, user experience, and modern design patterns."
            )
        elif dim.name == "Code Quality":
            suggestions.append(
                f"Work on {severity}ing code quality. Follow PEP8/linting rules, "
                "use meaningful variable names, and add proper error handling."
            )
        elif dim.name == "Logic":
            suggestions.append(
                f"Your problem-solving approach could be {severity}d. Consider "
                "breaking down problems, optimizing algorithms, and edge case handling."
            )
        elif dim.name == "Documentation":
            suggestions.append(
                f"Documentation needs {severity}ment. Add comprehensive README, "
                "inline comments, and API documentation."
            )
        elif dim.name == "Testing":
            suggestions.append(
                f"Testing coverage should be {severity}d. Add unit tests, "
                "integration tests, and ensure good test coverage."
            )
        elif dim.name == "Architecture":
            suggestions.append(
                f"System architecture could be {severity}d. Focus on scalability, "
                "separation of concerns, and clean code organization."
            )
    
    if not suggestions:
        suggestions.append(
            "Great work! Your skills are well-balanced. Keep refining all areas."
        )
    
    return suggestions


# =============================================================================
# PEER COMPARISON (Per-Event Percentile)
# =============================================================================

async def calculate_peer_comparison(
    db: Any,
    event_id: str,
    user_id: str
) -> PeerComparison:
    """
    Calculate per-event peer comparison metrics for a participant.
    
    Computes the team's percentile ranking within the event, comparing
    their performance against all other teams.
    
    Args:
        db: MongoDB database instance from Motor.
        event_id: The event identifier.
        user_id: The participant's user ID.
    
    Returns:
        PeerComparison: Percentile ranking and comparison metrics.
    
    Raises:
        InvalidEventError: If event_id is invalid or not found.
        InvalidTeamError: If user is not part of any team.
        InsufficientDataError: If not enough teams for comparison.
    
    Example:
        >>> comparison = await calculate_peer_comparison(db, "event123", "user456")
        >>> print(f"Percentile: {comparison.percentile_rank:.1f}th")
        >>> print(f"Rank: {comparison.rank}/{comparison.total_teams}")
    """
    logger.info(f"Calculating peer comparison for user {user_id} in event {event_id}")
    
    # Validate event
    events_collection = db["events"]
    try:
        event = await events_collection.find_one({"_id": ObjectId(event_id)})
    except Exception:
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
    
    # Find user's team
    teams_collection = db["teams"]
    user_team = await _find_user_team(teams_collection, event_id, user_id)
    
    if not user_team:
        raise InvalidTeamError(
            message="User is not part of any team in this event",
            team_id=None,
            event_id=event_id,
            reason="not_member"
        )
    
    user_team_id = str(user_team.get("_id", user_team.get("teamId", "")))
    user_team_name = user_team.get("teamName", "Unknown Team")
    
    # Get all teams in the event
    all_teams = await teams_collection.find({"eventId": event_id}).to_list(None)
    
    if len(all_teams) < MIN_SUBMISSIONS_FOR_PERCENTILE:
        raise InsufficientDataError(
            message=f"Need at least {MIN_SUBMISSIONS_FOR_PERCENTILE} teams for comparison",
            required_count=MIN_SUBMISSIONS_FOR_PERCENTILE,
            actual_count=len(all_teams)
        )
    
    # Calculate score for each team
    submissions_collection = db["submissions"]
    team_scores: List[Dict[str, Any]] = []
    
    for team in all_teams:
        team_id = str(team.get("_id", team.get("teamId", "")))
        
        # Get team's submissions
        submissions = await submissions_collection.find({
            "eventId": event_id,
            "$or": [
                {"teamId": team_id},
                {"teamId": str(team.get("_id", ""))}
            ]
        }).to_list(None)
        
        if not submissions:
            continue
        
        # Calculate average score across submissions
        scores = []
        for sub in submissions:
            score = _extract_score_from_submission(sub)
            if score is not None:
                scores.append(score)
        
        if scores:
            avg_score = statistics.mean(scores)
            team_scores.append({
                "team_id": team_id,
                "team_name": team.get("teamName", "Unknown"),
                "score": avg_score,
            })
    
    if not team_scores:
        raise InsufficientDataError(
            message="No scored submissions found in this event",
            required_count=1,
            actual_count=0
        )
    
    # Sort teams by score (descending)
    team_scores.sort(key=lambda x: x["score"], reverse=True)
    
    # Find user's team in the rankings
    user_team_data = None
    user_rank = 0
    
    for i, ts in enumerate(team_scores, 1):
        if ts["team_id"] == user_team_id or ts["team_id"] == str(user_team.get("_id", "")):
            user_team_data = ts
            user_rank = i
            break
    
    if not user_team_data:
        raise InsufficientDataError(
            message="Your team has no scored submissions",
            required_count=1,
            actual_count=0
        )
    
    # Calculate statistics
    all_scores = [ts["score"] for ts in team_scores]
    event_avg = statistics.mean(all_scores)
    event_median = statistics.median(all_scores)
    
    user_score = user_team_data["score"]
    score_difference = user_score - event_avg
    above_average = score_difference > 0
    
    # Calculate percentile (percentage of teams scoring below this team)
    teams_below = sum(1 for ts in team_scores if ts["score"] < user_score)
    percentile_rank = (teams_below / len(team_scores)) * 100
    
    # Calculate per-round breakdown
    by_round = await _calculate_per_round_comparison(
        submissions_collection, event_id, user_team_id, all_teams
    )
    
    logger.info(
        f"Peer comparison complete: rank={user_rank}/{len(team_scores)}, "
        f"percentile={percentile_rank:.1f}, score={user_score:.2f}"
    )
    
    return PeerComparison(
        event_id=event_id,
        team_id=user_team_id,
        team_name=user_team_name,
        team_score=round(user_score, 2),
        event_avg_score=round(event_avg, 2),
        event_median_score=round(event_median, 2),
        percentile_rank=round(percentile_rank, 2),
        rank=user_rank,
        total_teams=len(team_scores),
        above_average=above_average,
        score_difference=round(score_difference, 2),
        by_round=by_round,
        generated_at=datetime.utcnow()
    )


def _extract_score_from_submission(sub: Dict) -> Optional[float]:
    """
    Extract score from a submission document.
    
    Args:
        sub: Submission document.
    
    Returns:
        Extracted score or None.
    """
    # Try aiResult first
    if "aiResult" in sub and sub["aiResult"]:
        ai_result = sub["aiResult"]
        if isinstance(ai_result, dict):
            score = ai_result.get("final_score")
            if score is not None:
                return float(score)
            
            score_obj = ai_result.get("score", {})
            if isinstance(score_obj, dict):
                score = score_obj.get("overall_score")
                if score is not None:
                    return float(score)
    
    # Fallback to top-level fields
    for field in ["score", "finalScore", "overall_score"]:
        if sub.get(field) is not None:
            return float(sub[field])
    
    return None


async def _calculate_per_round_comparison(
    submissions_collection: Any,
    event_id: str,
    user_team_id: str,
    all_teams: List[Dict]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate comparison metrics for each round.
    
    Args:
        submissions_collection: MongoDB submissions collection.
        event_id: Event identifier.
        user_team_id: User's team ID.
        all_teams: List of all team documents.
    
    Returns:
        Dict mapping round_id to comparison metrics.
    """
    result: Dict[str, Dict[str, float]] = {}
    round_ids = ["ppt", "repo", "viva"]
    
    for round_id in round_ids:
        round_scores: List[float] = []
        user_round_score: Optional[float] = None
        
        for team in all_teams:
            team_id = str(team.get("_id", team.get("teamId", "")))
            
            # Get submission for this round
            sub = await submissions_collection.find_one({
                "eventId": event_id,
                "teamId": team_id,
                "roundId": round_id
            })
            
            if sub:
                score = _extract_score_from_submission(sub)
                if score is not None:
                    round_scores.append(score)
                    if team_id == user_team_id:
                        user_round_score = score
        
        if round_scores and user_round_score is not None:
            avg = statistics.mean(round_scores)
            result[round_id] = {
                "team_score": round(user_round_score, 2),
                "event_avg": round(avg, 2),
                "difference": round(user_round_score - avg, 2),
                "percentile": round(
                    sum(1 for s in round_scores if s < user_round_score) 
                    / len(round_scores) * 100, 2
                ),
            }
    
    return result


# =============================================================================
# PROGRESS TRACKING
# =============================================================================

async def track_progress(
    db: Any,
    user_id: str
) -> ProgressTimeline:
    """
    Track a participant's progress across multiple hackathons.
    
    Aggregates performance data from all events the user has participated
    in to show improvement over time.
    
    Args:
        db: MongoDB database instance from Motor.
        user_id: The participant's user ID.
    
    Returns:
        ProgressTimeline: Chronological progress with metrics.
    
    Raises:
        InsufficientDataError: If user hasn't participated in enough events.
    
    Example:
        >>> progress = await track_progress(db, "user456")
        >>> print(f"Events participated: {progress.total_events_participated}")
        >>> print(f"Overall improvement: {progress.overall_improvement:+.2f}")
        >>> print(f"Trend: {progress.trend}")
    """
    logger.info(f"Tracking progress for user: {user_id}")
    
    teams_collection = db["teams"]
    events_collection = db["events"]
    submissions_collection = db["submissions"]
    
    # Find all teams the user has been part of
    user_teams = await teams_collection.find({
        "$or": [
            {"members.userId": user_id},
            {"leaderId": user_id}
        ]
    }).to_list(None)
    
    if not user_teams:
        raise InsufficientDataError(
            message="User has not participated in any events",
            required_count=1,
            actual_count=0
        )
    
    logger.info(f"Found {len(user_teams)} team memberships for user {user_id}")
    
    # Build progress entries for each event
    entries: List[ProgressEntry] = []
    
    for team in user_teams:
        event_id = team.get("eventId", "")
        team_id = str(team.get("_id", team.get("teamId", "")))
        team_name = team.get("teamName", "Unknown Team")
        
        # Get event details
        try:
            event = await events_collection.find_one({"_id": ObjectId(event_id)})
        except Exception:
            event = None
        
        if not event:
            continue
        
        event_name = event.get("name", "Unknown Event")
        event_date_str = event.get("date", "")
        
        try:
            event_date = datetime.fromisoformat(event_date_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            event_date = datetime.utcnow()
        
        # Get team's submissions
        submissions = await submissions_collection.find({
            "eventId": event_id,
            "teamId": team_id
        }).to_list(None)
        
        if not submissions:
            continue
        
        # Calculate team's score
        scores = []
        for sub in submissions:
            score = _extract_score_from_submission(sub)
            if score is not None:
                scores.append(score)
        
        if not scores:
            continue
        
        overall_score = statistics.mean(scores)
        
        # Get all teams' scores for percentile/rank calculation
        all_event_teams = await teams_collection.find({"eventId": event_id}).to_list(None)
        all_team_scores = []
        
        for t in all_event_teams:
            t_id = str(t.get("_id", t.get("teamId", "")))
            t_subs = await submissions_collection.find({
                "eventId": event_id,
                "teamId": t_id
            }).to_list(None)
            
            t_scores = [_extract_score_from_submission(s) for s in t_subs]
            t_scores = [s for s in t_scores if s is not None]
            
            if t_scores:
                all_team_scores.append({
                    "team_id": t_id,
                    "score": statistics.mean(t_scores)
                })
        
        # Calculate rank and percentile
        all_team_scores.sort(key=lambda x: x["score"], reverse=True)
        rank = next(
            (i + 1 for i, ts in enumerate(all_team_scores) if ts["team_id"] == team_id),
            len(all_team_scores)
        )
        
        teams_below = sum(1 for ts in all_team_scores if ts["score"] < overall_score)
        percentile = (teams_below / len(all_team_scores) * 100) if all_team_scores else 0
        
        # Get skill snapshot
        skill_scores = await _extract_skill_scores(submissions)
        
        entry = ProgressEntry(
            event_id=event_id,
            event_name=event_name,
            event_date=event_date,
            team_name=team_name,
            overall_score=round(overall_score, 2),
            percentile_rank=round(percentile, 2),
            rank=rank,
            total_participants=len(all_team_scores),
            skills_snapshot={k: round(v, 2) for k, v in skill_scores.items()},
            improvement_from_previous=None,  # Will be filled in next step
        )
        entries.append(entry)
    
    if not entries:
        raise InsufficientDataError(
            message="No scored submissions found for this user",
            required_count=1,
            actual_count=0
        )
    
    # Sort entries by date
    entries.sort(key=lambda x: x.event_date)
    
    # Calculate improvement from previous event
    for i in range(1, len(entries)):
        entries[i].improvement_from_previous = round(
            entries[i].overall_score - entries[i - 1].overall_score, 2
        )
    
    # Calculate overall statistics
    all_scores = [e.overall_score for e in entries]
    all_percentiles = [e.percentile_rank for e in entries]
    
    overall_improvement = all_scores[-1] - all_scores[0] if len(all_scores) >= 2 else 0
    avg_percentile = statistics.mean(all_percentiles) if all_percentiles else 0
    
    # Find best performance
    best_performance = max(entries, key=lambda x: x.overall_score) if entries else None
    
    # Calculate consistency score
    if len(all_scores) > 1:
        score_std = statistics.stdev(all_scores)
        # Consistency = 100 - normalized std dev (lower variance = more consistent)
        consistency_score = max(0, 100 - (score_std * 2))
    else:
        consistency_score = 100.0
    
    # Determine trend
    if len(all_scores) >= 2:
        x = np.arange(len(all_scores))
        y = np.array(all_scores)
        slope = np.polyfit(x, y, 1)[0]
        trend = classify_trend(slope)
    else:
        trend = "stable"
    
    logger.info(
        f"Progress tracking complete: {len(entries)} events, "
        f"improvement={overall_improvement:+.2f}, trend={trend}"
    )
    
    return ProgressTimeline(
        user_id=user_id,
        total_events_participated=len(entries),
        entries=entries,
        overall_improvement=round(overall_improvement, 2),
        avg_percentile=round(avg_percentile, 2),
        best_performance=best_performance,
        consistency_score=round(consistency_score, 2),
        trend=trend,
        generated_at=datetime.utcnow()
    )
