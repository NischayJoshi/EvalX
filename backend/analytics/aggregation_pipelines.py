"""
MongoDB Aggregation Pipelines
=============================

This module contains reusable MongoDB aggregation pipeline builders
for efficient analytics queries. These pipelines are optimized for
performance and handle complex data transformations server-side.

Pipeline Categories:
    - Score aggregations: Distribution, statistics, rankings
    - Time-based aggregations: Heatmaps, trends, patterns
    - Theme aggregations: Group by detected theme
    - User history: Cross-event participant data

Usage:
    pipeline = build_score_distribution_pipeline(event_id)
    result = await collection.aggregate(pipeline).to_list(None)
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Module logger
logger = logging.getLogger(__name__)


# =============================================================================
# SCORE AGGREGATION PIPELINES
# =============================================================================

def build_score_distribution_pipeline(
    event_id: str,
    round_filter: Optional[str] = None,
    bucket_size: int = 20
) -> List[Dict[str, Any]]:
    """
    Build aggregation pipeline for score distribution histogram.
    
    Creates buckets of scores (e.g., 0-20, 21-40) and counts submissions
    in each bucket.
    
    Args:
        event_id: Event identifier to filter by.
        round_filter: Optional round filter ('ppt', 'repo', 'viva').
        bucket_size: Size of each score bucket (default 20).
    
    Returns:
        MongoDB aggregation pipeline as list of stages.
    
    Example:
        >>> pipeline = build_score_distribution_pipeline("event123", round_filter="repo")
        >>> result = await submissions.aggregate(pipeline).to_list(None)
        >>> # Returns: [{"_id": "0-20", "count": 5}, {"_id": "21-40", "count": 12}, ...]
    """
    logger.debug(f"Building score distribution pipeline for event {event_id}")
    
    # Match stage
    match_stage: Dict[str, Any] = {"eventId": event_id}
    if round_filter:
        match_stage["roundId"] = round_filter
    
    # Build bucket boundaries
    boundaries = list(range(0, 101, bucket_size)) + [101]
    
    pipeline = [
        # Filter by event and optionally round
        {"$match": match_stage},
        
        # Project to extract score from aiResult or fallback fields
        {"$project": {
            "score": {
                "$ifNull": [
                    "$aiResult.final_score",
                    {"$ifNull": [
                        "$aiResult.score.overall_score",
                        {"$ifNull": ["$score", "$finalScore"]}
                    ]}
                ]
            }
        }},
        
        # Filter out null scores
        {"$match": {"score": {"$ne": None}}},
        
        # Bucket by score ranges
        {"$bucket": {
            "groupBy": "$score",
            "boundaries": boundaries,
            "default": "Other",
            "output": {
                "count": {"$sum": 1},
                "scores": {"$push": "$score"}
            }
        }},
        
        # Format output
        {"$project": {
            "bucket": {
                "$concat": [
                    {"$toString": "$_id"},
                    "-",
                    {"$toString": {"$add": ["$_id", bucket_size - 1]}}
                ]
            },
            "count": 1,
            "min_score": {"$min": "$scores"},
            "max_score": {"$max": "$scores"},
            "avg_score": {"$avg": "$scores"}
        }},
        
        # Sort by bucket
        {"$sort": {"_id": 1}}
    ]
    
    return pipeline


def build_score_statistics_pipeline(
    event_id: str,
    round_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Build aggregation pipeline for score statistics.
    
    Calculates mean, median, standard deviation, min, max for an event.
    
    Args:
        event_id: Event identifier to filter by.
        round_filter: Optional round filter.
    
    Returns:
        MongoDB aggregation pipeline.
    
    Example:
        >>> pipeline = build_score_statistics_pipeline("event123")
        >>> result = await submissions.aggregate(pipeline).to_list(None)
        >>> # Returns: [{"mean": 72.5, "stdDev": 15.3, "min": 35, "max": 98, ...}]
    """
    logger.debug(f"Building score statistics pipeline for event {event_id}")
    
    match_stage: Dict[str, Any] = {"eventId": event_id}
    if round_filter:
        match_stage["roundId"] = round_filter
    
    pipeline = [
        {"$match": match_stage},
        
        # Extract score
        {"$project": {
            "score": {
                "$ifNull": [
                    "$aiResult.final_score",
                    {"$ifNull": [
                        "$aiResult.score.overall_score",
                        {"$ifNull": ["$score", "$finalScore"]}
                    ]}
                ]
            },
            "roundId": 1
        }},
        
        {"$match": {"score": {"$ne": None}}},
        
        # Group and calculate statistics
        {"$group": {
            "_id": None,
            "count": {"$sum": 1},
            "mean": {"$avg": "$score"},
            "stdDev": {"$stdDevPop": "$score"},
            "min": {"$min": "$score"},
            "max": {"$max": "$score"},
            "scores": {"$push": "$score"}
        }},
        
        # Calculate additional metrics
        {"$project": {
            "_id": 0,
            "count": 1,
            "mean": {"$round": ["$mean", 2]},
            "stdDev": {"$round": [{"$ifNull": ["$stdDev", 0]}, 2]},
            "min": {"$round": ["$min", 2]},
            "max": {"$round": ["$max", 2]},
            "range": {"$round": [{"$subtract": ["$max", "$min"]}, 2]},
            "variance": {
                "$round": [
                    {"$pow": [{"$ifNull": ["$stdDev", 0]}, 2]},
                    2
                ]
            }
        }}
    ]
    
    return pipeline


def build_percentile_pipeline(
    event_id: str,
    target_score: float,
    round_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Build aggregation pipeline to calculate percentile for a score.
    
    Counts how many submissions scored below the target score
    to calculate percentile ranking.
    
    Args:
        event_id: Event identifier.
        target_score: Score to calculate percentile for.
        round_filter: Optional round filter.
    
    Returns:
        MongoDB aggregation pipeline.
    
    Example:
        >>> pipeline = build_percentile_pipeline("event123", 85.0)
        >>> result = await submissions.aggregate(pipeline).to_list(None)
        >>> # Returns: [{"percentile": 78.5, "total": 50, "below": 39}]
    """
    logger.debug(f"Building percentile pipeline for score {target_score}")
    
    match_stage: Dict[str, Any] = {"eventId": event_id}
    if round_filter:
        match_stage["roundId"] = round_filter
    
    pipeline = [
        {"$match": match_stage},
        
        {"$project": {
            "score": {
                "$ifNull": [
                    "$aiResult.final_score",
                    {"$ifNull": [
                        "$aiResult.score.overall_score",
                        {"$ifNull": ["$score", "$finalScore"]}
                    ]}
                ]
            }
        }},
        
        {"$match": {"score": {"$ne": None}}},
        
        # Count total and below target
        {"$group": {
            "_id": None,
            "total": {"$sum": 1},
            "below": {
                "$sum": {
                    "$cond": [{"$lt": ["$score", target_score]}, 1, 0]
                }
            },
            "equal": {
                "$sum": {
                    "$cond": [{"$eq": ["$score", target_score]}, 1, 0]
                }
            }
        }},
        
        {"$project": {
            "_id": 0,
            "total": 1,
            "below": 1,
            "equal": 1,
            "percentile": {
                "$round": [
                    {"$multiply": [
                        {"$divide": ["$below", "$total"]},
                        100
                    ]},
                    2
                ]
            }
        }}
    ]
    
    return pipeline


# =============================================================================
# TIME-BASED AGGREGATION PIPELINES
# =============================================================================

def build_submission_heatmap_pipeline(
    event_id: str
) -> List[Dict[str, Any]]:
    """
    Build aggregation pipeline for submission timing heatmap.
    
    Groups submissions by day of week and hour of day for
    heatmap visualization.
    
    Args:
        event_id: Event identifier.
    
    Returns:
        MongoDB aggregation pipeline.
    
    Example:
        >>> pipeline = build_submission_heatmap_pipeline("event123")
        >>> result = await submissions.aggregate(pipeline).to_list(None)
        >>> # Returns: [{"day": 0, "hour": 14, "count": 5}, ...]
    """
    logger.debug(f"Building heatmap pipeline for event {event_id}")
    
    pipeline = [
        {"$match": {
            "eventId": event_id,
            "submittedAt": {"$exists": True, "$ne": None}
        }},
        
        # Extract day and hour from submittedAt
        {"$project": {
            "dayOfWeek": {"$dayOfWeek": "$submittedAt"},  # 1=Sunday, 2=Monday, ...
            "hour": {"$hour": "$submittedAt"}
        }},
        
        # Adjust day of week to 0=Monday format
        {"$project": {
            "day": {
                "$mod": [{"$add": ["$dayOfWeek", 5]}, 7]  # Convert to 0=Monday
            },
            "hour": 1
        }},
        
        # Group by day and hour
        {"$group": {
            "_id": {
                "day": "$day",
                "hour": "$hour"
            },
            "count": {"$sum": 1}
        }},
        
        # Flatten structure
        {"$project": {
            "_id": 0,
            "day": "$_id.day",
            "hour": "$_id.hour",
            "count": 1
        }},
        
        # Sort for consistent output
        {"$sort": {"day": 1, "hour": 1}}
    ]
    
    return pipeline


def build_submission_timeline_pipeline(
    event_id: str,
    interval_hours: int = 6
) -> List[Dict[str, Any]]:
    """
    Build aggregation pipeline for submission timeline.
    
    Groups submissions into time intervals for trend analysis.
    
    Args:
        event_id: Event identifier.
        interval_hours: Hours per interval (default 6).
    
    Returns:
        MongoDB aggregation pipeline.
    
    Example:
        >>> pipeline = build_submission_timeline_pipeline("event123", interval_hours=4)
        >>> # Groups submissions into 4-hour intervals
    """
    logger.debug(f"Building timeline pipeline for event {event_id}")
    
    # Calculate interval in milliseconds
    interval_ms = interval_hours * 60 * 60 * 1000
    
    pipeline = [
        {"$match": {
            "eventId": event_id,
            "submittedAt": {"$exists": True}
        }},
        
        # Group by time interval
        {"$group": {
            "_id": {
                "$toDate": {
                    "$subtract": [
                        {"$toLong": "$submittedAt"},
                        {"$mod": [{"$toLong": "$submittedAt"}, interval_ms]}
                    ]
                }
            },
            "count": {"$sum": 1},
            "avg_score": {
                "$avg": {
                    "$ifNull": [
                        "$aiResult.final_score",
                        "$score"
                    ]
                }
            }
        }},
        
        {"$project": {
            "_id": 0,
            "timestamp": "$_id",
            "count": 1,
            "avg_score": {"$round": ["$avg_score", 2]}
        }},
        
        {"$sort": {"timestamp": 1}}
    ]
    
    return pipeline


def build_deadline_analysis_pipeline(
    event_id: str,
    deadline: datetime
) -> List[Dict[str, Any]]:
    """
    Build aggregation pipeline for deadline proximity analysis.
    
    Categorizes submissions by how close to deadline they were submitted.
    
    Args:
        event_id: Event identifier.
        deadline: Event deadline datetime.
    
    Returns:
        MongoDB aggregation pipeline.
    
    Example:
        >>> pipeline = build_deadline_analysis_pipeline("event123", deadline_dt)
        >>> # Returns: [{"category": "early", "count": 20}, {"category": "last_minute", "count": 5}]
    """
    logger.debug(f"Building deadline analysis pipeline for event {event_id}")
    
    pipeline = [
        {"$match": {
            "eventId": event_id,
            "submittedAt": {"$exists": True}
        }},
        
        # Calculate hours before deadline
        {"$project": {
            "hoursBefore": {
                "$divide": [
                    {"$subtract": [deadline, "$submittedAt"]},
                    3600000  # ms to hours
                ]
            }
        }},
        
        # Categorize by timing
        {"$project": {
            "category": {
                "$switch": {
                    "branches": [
                        {"case": {"$gt": ["$hoursBefore", 24]}, "then": "early"},
                        {"case": {"$lt": ["$hoursBefore", 1]}, "then": "last_minute"},
                        {"case": {"$gte": ["$hoursBefore", 0]}, "then": "on_time"},
                    ],
                    "default": "late"
                }
            },
            "hoursBefore": 1
        }},
        
        # Group by category
        {"$group": {
            "_id": "$category",
            "count": {"$sum": 1},
            "avg_hours_before": {"$avg": "$hoursBefore"}
        }},
        
        {"$project": {
            "_id": 0,
            "category": "$_id",
            "count": 1,
            "avg_hours_before": {"$round": ["$avg_hours_before", 2]}
        }},
        
        {"$sort": {"count": -1}}
    ]
    
    return pipeline


# =============================================================================
# TEAM AGGREGATION PIPELINES
# =============================================================================

def build_team_leaderboard_pipeline(
    event_id: str,
    round_filter: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Build aggregation pipeline for team leaderboard.
    
    Aggregates scores by team and ranks them.
    
    Args:
        event_id: Event identifier.
        round_filter: Optional round filter.
        limit: Maximum teams to return (default 10).
    
    Returns:
        MongoDB aggregation pipeline.
    
    Example:
        >>> pipeline = build_team_leaderboard_pipeline("event123", limit=20)
        >>> result = await submissions.aggregate(pipeline).to_list(None)
    """
    logger.debug(f"Building leaderboard pipeline for event {event_id}")
    
    match_stage: Dict[str, Any] = {"eventId": event_id}
    if round_filter:
        match_stage["roundId"] = round_filter
    
    pipeline = [
        {"$match": match_stage},
        
        # Extract score
        {"$project": {
            "teamId": 1,
            "roundId": 1,
            "score": {
                "$ifNull": [
                    "$aiResult.final_score",
                    {"$ifNull": [
                        "$aiResult.score.overall_score",
                        {"$ifNull": ["$score", "$finalScore"]}
                    ]}
                ]
            }
        }},
        
        {"$match": {"score": {"$ne": None}}},
        
        # Group by team
        {"$group": {
            "_id": "$teamId",
            "avg_score": {"$avg": "$score"},
            "max_score": {"$max": "$score"},
            "submission_count": {"$sum": 1},
            "rounds": {"$addToSet": "$roundId"}
        }},
        
        # Sort by average score descending
        {"$sort": {"avg_score": -1}},
        
        # Limit results
        {"$limit": limit},
        
        # Add rank
        {"$group": {
            "_id": None,
            "teams": {"$push": {
                "team_id": "$_id",
                "avg_score": {"$round": ["$avg_score", 2]},
                "max_score": {"$round": ["$max_score", 2]},
                "submission_count": "$submission_count",
                "rounds": "$rounds"
            }}
        }},
        
        {"$unwind": {"path": "$teams", "includeArrayIndex": "rank"}},
        
        {"$project": {
            "_id": 0,
            "team_id": "$teams.team_id",
            "rank": {"$add": ["$rank", 1]},
            "avg_score": "$teams.avg_score",
            "max_score": "$teams.max_score",
            "submission_count": "$teams.submission_count",
            "rounds": "$teams.rounds"
        }}
    ]
    
    return pipeline


def build_team_comparison_pipeline(
    event_id: str,
    team_ids: List[str]
) -> List[Dict[str, Any]]:
    """
    Build aggregation pipeline to compare specific teams.
    
    Retrieves detailed metrics for a set of teams for comparison.
    
    Args:
        event_id: Event identifier.
        team_ids: List of team IDs to compare.
    
    Returns:
        MongoDB aggregation pipeline.
    """
    logger.debug(f"Building team comparison pipeline for {len(team_ids)} teams")
    
    pipeline = [
        {"$match": {
            "eventId": event_id,
            "teamId": {"$in": team_ids}
        }},
        
        {"$project": {
            "teamId": 1,
            "roundId": 1,
            "score": {
                "$ifNull": [
                    "$aiResult.final_score",
                    "$score"
                ]
            },
            "submittedAt": 1
        }},
        
        {"$match": {"score": {"$ne": None}}},
        
        # Group by team and round
        {"$group": {
            "_id": {
                "teamId": "$teamId",
                "roundId": "$roundId"
            },
            "score": {"$first": "$score"},
            "submittedAt": {"$first": "$submittedAt"}
        }},
        
        # Reshape by team
        {"$group": {
            "_id": "$_id.teamId",
            "rounds": {
                "$push": {
                    "round": "$_id.roundId",
                    "score": {"$round": ["$score", 2]},
                    "submittedAt": "$submittedAt"
                }
            },
            "total_score": {"$sum": "$score"},
            "avg_score": {"$avg": "$score"}
        }},
        
        {"$project": {
            "_id": 0,
            "team_id": "$_id",
            "rounds": 1,
            "total_score": {"$round": ["$total_score", 2]},
            "avg_score": {"$round": ["$avg_score", 2]}
        }},
        
        {"$sort": {"avg_score": -1}}
    ]
    
    return pipeline


# =============================================================================
# USER HISTORY PIPELINES
# =============================================================================

def build_user_event_history_pipeline(
    user_id: str
) -> List[Dict[str, Any]]:
    """
    Build aggregation pipeline for user's event participation history.
    
    Retrieves all events a user has participated in through their teams.
    Note: This pipeline is for the teams collection.
    
    Args:
        user_id: User identifier.
    
    Returns:
        MongoDB aggregation pipeline (for teams collection).
    
    Example:
        >>> pipeline = build_user_event_history_pipeline("user123")
        >>> result = await teams.aggregate(pipeline).to_list(None)
    """
    logger.debug(f"Building user history pipeline for user {user_id}")
    
    pipeline = [
        # Match teams where user is a member or leader
        {"$match": {
            "$or": [
                {"members.userId": user_id},
                {"leaderId": user_id}
            ]
        }},
        
        # Project relevant fields
        {"$project": {
            "eventId": 1,
            "teamId": {"$toString": "$_id"},
            "teamName": 1,
            "createdAt": 1,
            "isLeader": {"$eq": ["$leaderId", user_id]},
            "memberCount": {"$size": {"$ifNull": ["$members", []]}}
        }},
        
        # Sort by creation date
        {"$sort": {"createdAt": -1}}
    ]
    
    return pipeline


def build_user_performance_trend_pipeline(
    event_ids: List[str],
    team_mapping: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Build aggregation pipeline for user performance across events.
    
    Retrieves scores for a user's teams across multiple events.
    Note: This pipeline is for the submissions collection.
    
    Args:
        event_ids: List of event IDs the user participated in.
        team_mapping: Dict mapping event_id to team_id.
    
    Returns:
        MongoDB aggregation pipeline.
    """
    logger.debug(f"Building performance trend pipeline for {len(event_ids)} events")
    
    # Build OR conditions for each event-team pair
    or_conditions = [
        {"eventId": event_id, "teamId": team_id}
        for event_id, team_id in team_mapping.items()
    ]
    
    if not or_conditions:
        return [{"$match": {"_id": None}}]  # Return empty
    
    pipeline = [
        {"$match": {"$or": or_conditions}},
        
        {"$project": {
            "eventId": 1,
            "teamId": 1,
            "roundId": 1,
            "score": {
                "$ifNull": [
                    "$aiResult.final_score",
                    "$score"
                ]
            },
            "submittedAt": 1
        }},
        
        {"$match": {"score": {"$ne": None}}},
        
        # Group by event
        {"$group": {
            "_id": "$eventId",
            "avg_score": {"$avg": "$score"},
            "max_score": {"$max": "$score"},
            "rounds_submitted": {"$addToSet": "$roundId"},
            "earliest_submission": {"$min": "$submittedAt"},
            "latest_submission": {"$max": "$submittedAt"}
        }},
        
        {"$project": {
            "_id": 0,
            "event_id": "$_id",
            "avg_score": {"$round": ["$avg_score", 2]},
            "max_score": {"$round": ["$max_score", 2]},
            "rounds_submitted": 1,
            "earliest_submission": 1,
            "latest_submission": 1
        }},
        
        {"$sort": {"earliest_submission": 1}}
    ]
    
    return pipeline


# =============================================================================
# THEME AGGREGATION PIPELINES
# =============================================================================

def build_event_summary_pipeline(
    event_id: str
) -> List[Dict[str, Any]]:
    """
    Build aggregation pipeline for comprehensive event summary.
    
    Provides overview statistics for an event including submission
    counts, score distributions, and timing metrics.
    
    Args:
        event_id: Event identifier.
    
    Returns:
        MongoDB aggregation pipeline.
    """
    logger.debug(f"Building event summary pipeline for event {event_id}")
    
    pipeline = [
        {"$match": {"eventId": event_id}},
        
        # Extract score and timing
        {"$project": {
            "roundId": 1,
            "teamId": 1,
            "submittedAt": 1,
            "score": {
                "$ifNull": [
                    "$aiResult.final_score",
                    {"$ifNull": ["$score", "$finalScore"]}
                ]
            }
        }},
        
        # Group to get overall summary
        {"$group": {
            "_id": None,
            "total_submissions": {"$sum": 1},
            "unique_teams": {"$addToSet": "$teamId"},
            "rounds": {"$addToSet": "$roundId"},
            "avg_score": {"$avg": "$score"},
            "min_score": {"$min": "$score"},
            "max_score": {"$max": "$score"},
            "std_dev": {"$stdDevPop": "$score"},
            "first_submission": {"$min": "$submittedAt"},
            "last_submission": {"$max": "$submittedAt"},
            "scores": {"$push": "$score"}
        }},
        
        # Calculate additional metrics
        {"$project": {
            "_id": 0,
            "total_submissions": 1,
            "unique_teams": {"$size": "$unique_teams"},
            "rounds_count": {"$size": "$rounds"},
            "rounds": 1,
            "avg_score": {"$round": ["$avg_score", 2]},
            "min_score": {"$round": [{"$ifNull": ["$min_score", 0]}, 2]},
            "max_score": {"$round": [{"$ifNull": ["$max_score", 0]}, 2]},
            "std_dev": {"$round": [{"$ifNull": ["$std_dev", 0]}, 2]},
            "score_range": {
                "$round": [
                    {"$subtract": [
                        {"$ifNull": ["$max_score", 0]},
                        {"$ifNull": ["$min_score", 0]}
                    ]},
                    2
                ]
            },
            "first_submission": 1,
            "last_submission": 1,
            "submission_window_hours": {
                "$round": [
                    {"$divide": [
                        {"$subtract": ["$last_submission", "$first_submission"]},
                        3600000
                    ]},
                    2
                ]
            },
            "scores_with_value": {
                "$size": {
                    "$filter": {
                        "input": "$scores",
                        "as": "s",
                        "cond": {"$ne": ["$$s", None]}
                    }
                }
            }
        }}
    ]
    
    return pipeline


def build_round_comparison_pipeline(
    event_id: str
) -> List[Dict[str, Any]]:
    """
    Build aggregation pipeline to compare rounds within an event.
    
    Shows performance metrics broken down by round type.
    
    Args:
        event_id: Event identifier.
    
    Returns:
        MongoDB aggregation pipeline.
    """
    logger.debug(f"Building round comparison pipeline for event {event_id}")
    
    pipeline = [
        {"$match": {"eventId": event_id}},
        
        {"$project": {
            "roundId": 1,
            "teamId": 1,
            "score": {
                "$ifNull": [
                    "$aiResult.final_score",
                    "$score"
                ]
            },
            "submittedAt": 1
        }},
        
        {"$match": {"score": {"$ne": None}}},
        
        # Group by round
        {"$group": {
            "_id": "$roundId",
            "submission_count": {"$sum": 1},
            "unique_teams": {"$addToSet": "$teamId"},
            "avg_score": {"$avg": "$score"},
            "min_score": {"$min": "$score"},
            "max_score": {"$max": "$score"},
            "std_dev": {"$stdDevPop": "$score"}
        }},
        
        {"$project": {
            "_id": 0,
            "round_id": "$_id",
            "submission_count": 1,
            "team_count": {"$size": "$unique_teams"},
            "avg_score": {"$round": ["$avg_score", 2]},
            "min_score": {"$round": ["$min_score", 2]},
            "max_score": {"$round": ["$max_score", 2]},
            "std_dev": {"$round": [{"$ifNull": ["$std_dev", 0]}, 2]},
            "score_range": {
                "$round": [{"$subtract": ["$max_score", "$min_score"]}, 2]
            }
        }},
        
        {"$sort": {"round_id": 1}}
    ]
    
    return pipeline
