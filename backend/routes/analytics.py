"""
Analytics API Routes
====================

FastAPI router for analytics endpoints. Provides API access to
analytics functions for both organizers and participants.

Endpoints:
    Organizer Routes (require organizer role):
        - GET /org/analytics/{event_id}/calibration - Calibration metrics
        - GET /org/analytics/{event_id}/themes - Theme-wise analysis
        - GET /org/analytics/trends - Historical trends
        - GET /org/analytics/{event_id}/patterns - Submission patterns
        - GET /org/analytics/{event_id}/export - CSV export
    
    Participant Routes (require authentication):
        - GET /dev/analytics/{event_id}/radar - Skill radar data
        - GET /dev/analytics/{event_id}/comparison - Peer comparison
        - GET /dev/analytics/progress - Progress timeline

All routes follow existing EvalX patterns for authentication,
error handling, and response format.
"""

import logging
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from config.db import db
from middlewares.auth_required import (
    get_user as get_current_user,
    organizer_required,
)
from utils.serializers import serialize_doc

from analytics.organizer_analytics import (
    calculate_calibration_metrics,
    analyze_themes,
    get_historical_trends,
    analyze_submission_patterns,
    detect_scoring_anomalies,
)
from analytics.participant_analytics import (
    generate_skill_radar,
    calculate_peer_comparison,
    track_progress,
)
from analytics.export_service import ExportService
from analytics.exceptions import (
    AnalyticsError,
    InsufficientDataError,
    InvalidEventError,
    InvalidTeamError,
    ExportError,
    ThemeDetectionError,
)

# Module logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["analytics"])


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def handle_analytics_exception(e: Exception) -> None:
    """
    Convert analytics exceptions to HTTPException.
    
    Args:
        e: Exception to handle.
    
    Raises:
        HTTPException: With appropriate status code and detail.
    """
    if isinstance(e, InvalidEventError):
        raise HTTPException(status_code=404, detail=str(e))
    elif isinstance(e, InvalidTeamError):
        raise HTTPException(status_code=404, detail=str(e))
    elif isinstance(e, InsufficientDataError):
        raise HTTPException(status_code=400, detail=str(e))
    elif isinstance(e, ExportError):
        raise HTTPException(status_code=400, detail=str(e))
    elif isinstance(e, ThemeDetectionError):
        raise HTTPException(status_code=400, detail=str(e))
    elif isinstance(e, AnalyticsError):
        raise HTTPException(status_code=500, detail=str(e))
    else:
        logger.error(f"Unexpected error in analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics error: {str(e)}")


# =============================================================================
# ORGANIZER ANALYTICS ROUTES
# =============================================================================

@router.get("/org/analytics/{event_id}/calibration")
async def get_calibration_metrics(
    event_id: str,
    round_filter: Optional[str] = Query(
        None,
        description="Filter by round (ppt, repo, viva)"
    ),
    user: dict = Depends(organizer_required)
) -> dict:
    """
    Get AI evaluation calibration metrics for an event.
    
    Analyzes the consistency and distribution of AI-generated scores
    to help organizers assess evaluation quality.
    
    Args:
        event_id: Event identifier.
        round_filter: Optional filter for specific round.
        user: Authenticated organizer (injected).
    
    Returns:
        Calibration metrics including mean, variance, anomalies.
    
    Raises:
        404: Event not found.
        400: Insufficient data for analysis.
        403: Not authorized (not organizer).
    """
    logger.info(
        f"Calibration metrics requested: event={event_id}, "
        f"round={round_filter}, user={user.get('id')}"
    )
    
    try:
        metrics = await calculate_calibration_metrics(
            db=db,
            event_id=event_id,
            round_filter=round_filter
        )
        
        return {
            "success": True,
            "data": metrics.model_dump(),
            "message": "Calibration metrics calculated successfully"
        }
    
    except (AnalyticsError, Exception) as e:
        handle_analytics_exception(e)


@router.get("/org/analytics/{event_id}/themes")
async def get_theme_analysis(
    event_id: str,
    user: dict = Depends(organizer_required)
) -> dict:
    """
    Get theme-wise performance analysis for an event.
    
    Groups submissions by detected theme (AI/ML, Web3, IoT, etc.)
    and provides performance metrics for each.
    
    Args:
        event_id: Event identifier.
        user: Authenticated organizer (injected).
    
    Returns:
        Theme analysis with per-theme metrics.
    
    Raises:
        404: Event not found.
        400: No submissions found.
        403: Not authorized.
    """
    logger.info(
        f"Theme analysis requested: event={event_id}, user={user.get('id')}"
    )
    
    try:
        analysis = await analyze_themes(db=db, event_id=event_id)
        
        return {
            "success": True,
            "data": analysis.model_dump(),
            "message": f"Theme analysis complete: {analysis.themes_detected} themes detected"
        }
    
    except (AnalyticsError, Exception) as e:
        handle_analytics_exception(e)


@router.get("/org/analytics/trends")
async def get_trends(
    scope: str = Query(
        "organizer",
        description="Analysis scope: 'organizer' (your events) or 'global' (all events)"
    ),
    current_event_id: Optional[str] = Query(
        None,
        description="Optional current event ID for comparison"
    ),
    user: dict = Depends(organizer_required)
) -> dict:
    """
    Get historical trends across multiple events.
    
    Analyzes performance patterns over time to help organizers
    understand how their events perform compared to history.
    
    Args:
        scope: 'organizer' for same organizer's events, 'global' for all.
        current_event_id: Optional event ID for comparison.
        user: Authenticated organizer (injected).
    
    Returns:
        Historical trend data with comparison metrics.
    
    Raises:
        400: Not enough events for analysis.
        403: Not authorized.
    """
    organizer_id = user.get("id")
    logger.info(
        f"Trends requested: scope={scope}, organizer={organizer_id}, "
        f"current_event={current_event_id}"
    )
    
    # Validate scope
    if scope not in ["organizer", "global"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid scope. Must be 'organizer' or 'global'"
        )
    
    try:
        trends = await get_historical_trends(
            db=db,
            organizer_id=organizer_id,
            scope=scope,
            current_event_id=current_event_id
        )
        
        return {
            "success": True,
            "data": trends.model_dump(),
            "message": f"Trends analyzed for {trends.total_events} events"
        }
    
    except (AnalyticsError, Exception) as e:
        handle_analytics_exception(e)


@router.get("/org/analytics/{event_id}/patterns")
async def get_submission_patterns(
    event_id: str,
    user: dict = Depends(organizer_required)
) -> dict:
    """
    Get submission timing patterns for an event.
    
    Provides heatmap data and timing metrics to understand
    when participants submit their work.
    
    Args:
        event_id: Event identifier.
        user: Authenticated organizer (injected).
    
    Returns:
        Submission patterns with heatmap and timing ratios.
    
    Raises:
        404: Event not found.
        400: No submissions found.
        403: Not authorized.
    """
    logger.info(
        f"Submission patterns requested: event={event_id}, user={user.get('id')}"
    )
    
    try:
        patterns = await analyze_submission_patterns(db=db, event_id=event_id)
        
        return {
            "success": True,
            "data": patterns.model_dump(),
            "message": f"Pattern analysis complete for {patterns.total_submissions} submissions"
        }
    
    except (AnalyticsError, Exception) as e:
        handle_analytics_exception(e)


@router.get("/org/analytics/{event_id}/anomalies")
async def get_scoring_anomalies(
    event_id: str,
    z_score_threshold: float = Query(
        2.0,
        description="Z-score threshold for anomaly detection (default 2.0)"
    ),
    user: dict = Depends(organizer_required)
) -> dict:
    """
    Detect scoring anomalies in event evaluations.
    
    Identifies submissions with unusually high or low scores
    that may warrant review.
    
    Args:
        event_id: Event identifier.
        z_score_threshold: Z-score threshold for flagging.
        user: Authenticated organizer (injected).
    
    Returns:
        List of detected scoring anomalies.
    
    Raises:
        404: Event not found.
        400: Insufficient data.
        403: Not authorized.
    """
    logger.info(
        f"Anomaly detection requested: event={event_id}, "
        f"threshold={z_score_threshold}, user={user.get('id')}"
    )
    
    try:
        anomalies = await detect_scoring_anomalies(
            db=db,
            event_id=event_id,
            z_score_threshold=z_score_threshold
        )
        
        return {
            "success": True,
            "data": {
                "anomalies": [a.model_dump() for a in anomalies],
                "count": len(anomalies),
                "z_score_threshold": z_score_threshold
            },
            "message": f"Found {len(anomalies)} scoring anomalies"
        }
    
    except (AnalyticsError, Exception) as e:
        handle_analytics_exception(e)


@router.get("/org/analytics/{event_id}/export")
async def export_evaluation_data(
    event_id: str,
    columns: Optional[str] = Query(
        None,
        description="Comma-separated list of columns to export"
    ),
    format: str = Query(
        "csv",
        description="Export format: 'csv' or 'json'"
    ),
    round_filter: Optional[str] = Query(
        None,
        description="Filter by round (ppt, repo, viva)"
    ),
    min_score: Optional[float] = Query(
        None,
        ge=0,
        le=100,
        description="Minimum score filter"
    ),
    max_score: Optional[float] = Query(
        None,
        ge=0,
        le=100,
        description="Maximum score filter"
    ),
    user: dict = Depends(organizer_required)
) -> dict:
    """
    Export evaluation data for offline analysis.
    
    Exports submission data with dynamic column selection.
    Available columns can be retrieved via /export/columns endpoint.
    
    Args:
        event_id: Event identifier.
        columns: Comma-separated column names (empty = defaults).
        format: Export format ('csv' or 'json').
        round_filter: Optional round filter.
        min_score: Optional minimum score filter.
        max_score: Optional maximum score filter.
        user: Authenticated organizer (injected).
    
    Returns:
        Export data and metadata.
    
    Raises:
        404: Event not found.
        400: Invalid columns or no data.
        403: Not authorized.
    """
    logger.info(
        f"Export requested: event={event_id}, format={format}, "
        f"columns={columns}, user={user.get('id')}"
    )
    
    # Parse columns
    column_list = None
    if columns:
        column_list = [c.strip() for c in columns.split(",") if c.strip()]
    
    try:
        export_service = ExportService(db)
        
        export_data, result = await export_service.export_evaluations(
            event_id=event_id,
            columns=column_list,
            export_format=format,
            round_filter=round_filter,
            min_score=min_score,
            max_score=max_score
        )
        
        return {
            "success": True,
            "data": {
                "content": export_data,
                "metadata": result.model_dump()
            },
            "message": f"Exported {result.row_count} rows"
        }
    
    except (AnalyticsError, Exception) as e:
        handle_analytics_exception(e)


@router.get("/org/analytics/{event_id}/export/download")
async def download_export(
    event_id: str,
    columns: Optional[str] = Query(None),
    format: str = Query("csv"),
    user: dict = Depends(organizer_required)
):
    """
    Download export as a file (streaming response).
    
    Returns the export as a downloadable file with appropriate
    content-type headers.
    
    Args:
        event_id: Event identifier.
        columns: Comma-separated column names.
        format: Export format.
        user: Authenticated organizer.
    
    Returns:
        StreamingResponse with file download.
    """
    logger.info(f"Export download requested: event={event_id}, format={format}")
    
    column_list = None
    if columns:
        column_list = [c.strip() for c in columns.split(",") if c.strip()]
    
    try:
        export_service = ExportService(db)
        
        # Generate filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"evalx_export_{event_id}_{timestamp}.{format}"
        
        if format == "csv":
            content_type = "text/csv"
        else:
            content_type = "application/json"
        
        # Stream response
        async def generate():
            async for chunk in export_service.generate_export_stream(
                event_id=event_id,
                columns=column_list
            ):
                yield chunk
        
        return StreamingResponse(
            generate(),
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    
    except (AnalyticsError, Exception) as e:
        handle_analytics_exception(e)


@router.get("/org/analytics/export/columns")
async def get_export_columns(
    user: dict = Depends(organizer_required)
) -> dict:
    """
    Get available columns for export.
    
    Returns list of all exportable columns with their types
    and descriptions.
    
    Args:
        user: Authenticated organizer (injected).
    
    Returns:
        Dictionary of available columns.
    """
    columns = ExportService.get_available_columns()
    defaults = ExportService.get_default_columns()
    
    return {
        "success": True,
        "data": {
            "available_columns": columns,
            "default_columns": defaults,
            "total_columns": len(columns)
        },
        "message": f"{len(columns)} columns available for export"
    }


# =============================================================================
# PARTICIPANT ANALYTICS ROUTES
# =============================================================================

@router.get("/dev/analytics/{event_id}/radar")
async def get_skill_radar(
    event_id: str,
    user: dict = Depends(get_current_user)
) -> dict:
    """
    Get skill radar chart data for participant's team.
    
    Maps AI evaluation results to skill dimensions for
    radar chart visualization.
    
    Args:
        event_id: Event identifier.
        user: Authenticated user (injected).
    
    Returns:
        Skill radar data with dimension scores.
    
    Raises:
        404: Event not found or user not in team.
        400: No submissions found.
    """
    user_id = user.get("id")
    logger.info(f"Skill radar requested: event={event_id}, user={user_id}")
    
    try:
        radar = await generate_skill_radar(
            db=db,
            event_id=event_id,
            user_id=user_id
        )
        
        return {
            "success": True,
            "data": radar.model_dump(),
            "message": "Skill radar generated successfully"
        }
    
    except (AnalyticsError, Exception) as e:
        handle_analytics_exception(e)


@router.get("/dev/analytics/{event_id}/comparison")
async def get_peer_comparison(
    event_id: str,
    user: dict = Depends(get_current_user)
) -> dict:
    """
    Get peer comparison metrics for participant's team.
    
    Calculates per-event percentile ranking and comparison
    to event averages.
    
    Args:
        event_id: Event identifier.
        user: Authenticated user (injected).
    
    Returns:
        Peer comparison with percentile and rank.
    
    Raises:
        404: Event not found or user not in team.
        400: Not enough data for comparison.
    """
    user_id = user.get("id")
    logger.info(f"Peer comparison requested: event={event_id}, user={user_id}")
    
    try:
        comparison = await calculate_peer_comparison(
            db=db,
            event_id=event_id,
            user_id=user_id
        )
        
        return {
            "success": True,
            "data": comparison.model_dump(),
            "message": f"Ranked {comparison.rank} of {comparison.total_teams} teams"
        }
    
    except (AnalyticsError, Exception) as e:
        handle_analytics_exception(e)


@router.get("/dev/analytics/progress")
async def get_progress_timeline(
    user: dict = Depends(get_current_user)
) -> dict:
    """
    Get progress timeline across all participated events.
    
    Tracks improvement and performance trends across
    multiple hackathons.
    
    Args:
        user: Authenticated user (injected).
    
    Returns:
        Progress timeline with trend analysis.
    
    Raises:
        400: Not enough events for analysis.
    """
    user_id = user.get("id")
    logger.info(f"Progress timeline requested: user={user_id}")
    
    try:
        progress = await track_progress(db=db, user_id=user_id)
        
        return {
            "success": True,
            "data": progress.model_dump(),
            "message": f"Progress tracked across {progress.total_events_participated} events"
        }
    
    except (AnalyticsError, Exception) as e:
        handle_analytics_exception(e)


# =============================================================================
# UTILITY ROUTES
# =============================================================================

@router.get("/analytics/health")
async def analytics_health_check() -> dict:
    """
    Health check endpoint for analytics module.
    
    Returns:
        Health status of analytics service.
    """
    return {
        "success": True,
        "data": {
            "status": "healthy",
            "module": "analytics",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat()
        },
        "message": "Analytics module is operational"
    }
