"""
Analytics Data Models
=====================

Pydantic models for all analytics data structures used in the EvalX
analytics module. These models ensure type safety and provide automatic
validation for API responses.

All models follow Google-style docstrings and include comprehensive
field descriptions for API documentation.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# CALIBRATION METRICS MODELS
# =============================================================================

class ScoringAnomaly(BaseModel):
    """
    Represents a detected scoring anomaly in evaluation data.
    
    Anomalies are submissions with scores that deviate significantly
    from the expected distribution, potentially indicating evaluation
    inconsistencies or exceptional submissions.
    
    Attributes:
        submission_id: Unique identifier for the submission.
        team_id: ID of the team that made the submission.
        team_name: Display name of the team.
        score: The anomalous score value.
        z_score: Standard deviations from mean (positive = above, negative = below).
        round_id: Which evaluation round (ppt, repo, viva).
        anomaly_type: Classification of anomaly (high, low, or outlier).
        detected_at: Timestamp when anomaly was detected.
    """
    model_config = ConfigDict(populate_by_name=True)
    
    submission_id: str = Field(..., description="Unique submission identifier")
    team_id: str = Field(..., description="Team identifier")
    team_name: str = Field(..., description="Team display name")
    score: float = Field(..., ge=0, le=100, description="Anomalous score value")
    z_score: float = Field(..., description="Z-score (std deviations from mean)")
    round_id: str = Field(..., description="Evaluation round (ppt/repo/viva)")
    anomaly_type: str = Field(..., description="Type: 'high', 'low', or 'outlier'")
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class CalibrationMetrics(BaseModel):
    """
    Judge/AI calibration metrics for an event's evaluations.
    
    Provides statistical measures to assess the consistency and
    reliability of AI-powered evaluations across submissions.
    
    Attributes:
        event_id: The event being analyzed.
        total_submissions: Count of submissions analyzed.
        mean_score: Average score across all submissions.
        median_score: Median score value.
        std_deviation: Standard deviation of scores.
        variance: Score variance.
        min_score: Lowest score recorded.
        max_score: Highest score recorded.
        score_range: Difference between max and min scores.
        coefficient_of_variation: CV as percentage (std_dev / mean * 100).
        interquartile_range: IQR for robust spread measure.
        anomalies: List of detected scoring anomalies.
        anomaly_rate: Percentage of submissions flagged as anomalies.
        by_round: Breakdown of metrics per evaluation round.
        calculated_at: Timestamp of calculation.
    """
    model_config = ConfigDict(populate_by_name=True)
    
    event_id: str = Field(..., description="Event identifier")
    total_submissions: int = Field(..., ge=0, description="Total submissions analyzed")
    mean_score: float = Field(..., ge=0, le=100, description="Average score")
    median_score: float = Field(..., ge=0, le=100, description="Median score")
    std_deviation: float = Field(..., ge=0, description="Standard deviation")
    variance: float = Field(..., ge=0, description="Score variance")
    min_score: float = Field(..., ge=0, le=100, description="Minimum score")
    max_score: float = Field(..., ge=0, le=100, description="Maximum score")
    score_range: float = Field(..., ge=0, description="Score range (max - min)")
    coefficient_of_variation: float = Field(..., description="CV percentage")
    interquartile_range: float = Field(..., ge=0, description="IQR")
    anomalies: List[ScoringAnomaly] = Field(default_factory=list)
    anomaly_rate: float = Field(..., ge=0, le=100, description="Anomaly percentage")
    by_round: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Metrics breakdown by round (ppt/repo/viva)"
    )
    calculated_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# THEME ANALYSIS MODELS
# =============================================================================

class ThemePerformance(BaseModel):
    """
    Performance metrics for a specific hackathon theme.
    
    Attributes:
        theme: Theme name (e.g., 'AI/ML', 'Web3', 'IoT').
        submission_count: Number of submissions in this theme.
        avg_score: Average score for the theme.
        median_score: Median score for the theme.
        std_deviation: Score spread within theme.
        min_score: Lowest score in theme.
        max_score: Highest score in theme.
        top_team: Best performing team in this theme.
        score_distribution: Score buckets for histogram.
    """
    model_config = ConfigDict(populate_by_name=True)
    
    theme: str = Field(..., description="Theme name")
    submission_count: int = Field(..., ge=0, description="Submissions in theme")
    avg_score: float = Field(..., ge=0, le=100, description="Average score")
    median_score: float = Field(..., ge=0, le=100, description="Median score")
    std_deviation: float = Field(..., ge=0, description="Standard deviation")
    min_score: float = Field(0, ge=0, le=100, description="Minimum score")
    max_score: float = Field(0, ge=0, le=100, description="Maximum score")
    top_team: Optional[Dict[str, Any]] = Field(None, description="Top team info")
    score_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Score buckets (0-20, 21-40, etc.)"
    )


class ThemeAnalysis(BaseModel):
    """
    Complete theme-wise analysis for an event.
    
    Attributes:
        event_id: Event being analyzed.
        themes_detected: Number of unique themes found.
        themes: List of per-theme performance metrics.
        strongest_theme: Theme with highest average score.
        weakest_theme: Theme with lowest average score.
        unclassified_count: Submissions that couldn't be classified.
        analysis_timestamp: When analysis was performed.
    """
    model_config = ConfigDict(populate_by_name=True)
    
    event_id: str = Field(..., description="Event identifier")
    themes_detected: int = Field(..., ge=0, description="Unique themes count")
    themes: List[ThemePerformance] = Field(default_factory=list)
    strongest_theme: Optional[str] = Field(None, description="Best performing theme")
    weakest_theme: Optional[str] = Field(None, description="Worst performing theme")
    unclassified_count: int = Field(0, ge=0, description="Unclassified submissions")
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# SUBMISSION PATTERNS MODELS
# =============================================================================

class HeatmapCell(BaseModel):
    """
    Single cell in a submission heatmap.
    
    Represents submission activity for a specific hour/day combination.
    
    Attributes:
        day: Day of week (0=Monday, 6=Sunday).
        hour: Hour of day (0-23).
        count: Number of submissions in this time slot.
        percentage: Percentage of total submissions.
    """
    model_config = ConfigDict(populate_by_name=True)
    
    day: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    count: int = Field(..., ge=0, description="Submission count")
    percentage: float = Field(..., ge=0, le=100, description="Percentage of total")


class SubmissionPattern(BaseModel):
    """
    Complete submission pattern analysis for an event.
    
    Includes heatmap data and timing metrics to understand
    when participants submit their work.
    
    Attributes:
        event_id: Event being analyzed.
        total_submissions: Total submission count.
        heatmap: List of heatmap cells for visualization.
        peak_hour: Most active hour (0-23).
        peak_day: Most active day (0-6).
        early_submission_ratio: Percentage submitted >24h before deadline.
        on_time_ratio: Percentage submitted within deadline.
        last_minute_ratio: Percentage submitted <1h before deadline.
        avg_time_before_deadline_hours: Average hours before deadline.
        submission_velocity: Submissions per hour in final 24 hours.
        deadline: Event submission deadline.
        first_submission: Timestamp of first submission.
        last_submission: Timestamp of last submission.
    """
    model_config = ConfigDict(populate_by_name=True)
    
    event_id: str = Field(..., description="Event identifier")
    total_submissions: int = Field(..., ge=0, description="Total submissions")
    heatmap: List[HeatmapCell] = Field(default_factory=list)
    peak_hour: int = Field(..., ge=0, le=23, description="Busiest hour")
    peak_day: int = Field(..., ge=0, le=6, description="Busiest day")
    early_submission_ratio: float = Field(..., ge=0, le=100)
    on_time_ratio: float = Field(..., ge=0, le=100)
    last_minute_ratio: float = Field(..., ge=0, le=100)
    avg_time_before_deadline_hours: float = Field(..., description="Avg hours early")
    submission_velocity: float = Field(..., description="Submissions/hour in final 24h")
    deadline: Optional[datetime] = Field(None, description="Submission deadline")
    first_submission: Optional[datetime] = Field(None)
    last_submission: Optional[datetime] = Field(None)


# =============================================================================
# HISTORICAL TRENDS MODELS
# =============================================================================

class TrendDataPoint(BaseModel):
    """
    Single data point in a historical trend.
    
    Attributes:
        event_id: Event identifier.
        event_name: Event display name.
        event_date: When the event occurred.
        avg_score: Average score for the event.
        submission_count: Number of submissions.
        team_count: Number of teams.
        completion_rate: Percentage of teams that submitted.
    """
    model_config = ConfigDict(populate_by_name=True)
    
    event_id: str = Field(..., description="Event identifier")
    event_name: str = Field(..., description="Event name")
    event_date: datetime = Field(..., description="Event date")
    avg_score: float = Field(..., ge=0, le=100, description="Average score")
    submission_count: int = Field(..., ge=0, description="Submissions")
    team_count: int = Field(..., ge=0, description="Teams registered")
    completion_rate: float = Field(..., ge=0, le=100, description="Completion %")


class HistoricalTrend(BaseModel):
    """
    Historical trends analysis across multiple events.
    
    Attributes:
        scope: Analysis scope ('organizer' or 'global').
        organizer_id: Organizer ID if scope is 'organizer'.
        total_events: Number of events analyzed.
        data_points: List of per-event trend data.
        overall_avg_score: Average score across all events.
        score_trend: 'improving', 'declining', or 'stable'.
        trend_slope: Linear regression slope of scores over time.
        best_performing_event: Event with highest avg score.
        current_vs_historical: Comparison of current event to history.
        generated_at: Timestamp of analysis.
    """
    model_config = ConfigDict(populate_by_name=True)
    
    scope: str = Field(..., description="'organizer' or 'global'")
    organizer_id: Optional[str] = Field(None, description="Organizer ID if scoped")
    total_events: int = Field(..., ge=0, description="Events analyzed")
    data_points: List[TrendDataPoint] = Field(default_factory=list)
    overall_avg_score: float = Field(..., ge=0, le=100)
    score_trend: str = Field(..., description="'improving', 'declining', 'stable'")
    trend_slope: float = Field(..., description="Score trend slope")
    best_performing_event: Optional[TrendDataPoint] = Field(None)
    current_vs_historical: Optional[Dict[str, float]] = Field(
        None,
        description="Comparison metrics (diff_from_avg, percentile, etc.)"
    )
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# SKILL RADAR MODELS (Participant)
# =============================================================================

class SkillDimension(BaseModel):
    """
    Single dimension in a skill radar chart.
    
    Attributes:
        name: Skill dimension name.
        score: Normalized score (0-100).
        raw_score: Original score before normalization.
        max_possible: Maximum possible raw score.
        description: Brief description of what this measures.
    """
    model_config = ConfigDict(populate_by_name=True)
    
    name: str = Field(..., description="Skill name")
    score: float = Field(..., ge=0, le=100, description="Normalized score")
    raw_score: float = Field(..., description="Raw score")
    max_possible: float = Field(..., description="Max possible score")
    description: str = Field("", description="Skill description")


class SkillRadarData(BaseModel):
    """
    Complete skill radar chart data for a team/participant.
    
    Attributes:
        event_id: Event context.
        team_id: Team being analyzed.
        team_name: Team display name.
        dimensions: List of skill dimensions with scores.
        overall_score: Computed overall score.
        strongest_skill: Highest scoring dimension.
        weakest_skill: Lowest scoring dimension.
        improvement_suggestions: AI-generated suggestions.
        based_on_rounds: Which rounds contributed to analysis.
        generated_at: Timestamp of generation.
    """
    model_config = ConfigDict(populate_by_name=True)
    
    event_id: str = Field(..., description="Event identifier")
    team_id: str = Field(..., description="Team identifier")
    team_name: str = Field(..., description="Team name")
    dimensions: List[SkillDimension] = Field(default_factory=list)
    overall_score: float = Field(..., ge=0, le=100, description="Overall score")
    strongest_skill: Optional[str] = Field(None, description="Best skill")
    weakest_skill: Optional[str] = Field(None, description="Weakest skill")
    improvement_suggestions: List[str] = Field(default_factory=list)
    based_on_rounds: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# PEER COMPARISON MODELS (Participant)
# =============================================================================

class PeerComparison(BaseModel):
    """
    Peer comparison metrics for a team within an event.
    
    Attributes:
        event_id: Event context.
        team_id: Team being compared.
        team_name: Team display name.
        team_score: Team's overall score.
        event_avg_score: Event average score.
        event_median_score: Event median score.
        percentile_rank: Team's percentile (0-100).
        rank: Absolute rank (1 = best).
        total_teams: Total teams in event.
        above_average: Whether team is above event average.
        score_difference: Difference from average (positive = above).
        by_round: Per-round comparison data.
        generated_at: Timestamp of analysis.
    """
    model_config = ConfigDict(populate_by_name=True)
    
    event_id: str = Field(..., description="Event identifier")
    team_id: str = Field(..., description="Team identifier")
    team_name: str = Field(..., description="Team name")
    team_score: float = Field(..., ge=0, le=100, description="Team's score")
    event_avg_score: float = Field(..., ge=0, le=100, description="Event average")
    event_median_score: float = Field(..., ge=0, le=100, description="Event median")
    percentile_rank: float = Field(..., ge=0, le=100, description="Percentile")
    rank: int = Field(..., ge=1, description="Absolute rank")
    total_teams: int = Field(..., ge=1, description="Total teams")
    above_average: bool = Field(..., description="Is above average")
    score_difference: float = Field(..., description="Diff from average")
    by_round: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Per-round comparison"
    )
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# PROGRESS TRACKING MODELS (Participant)
# =============================================================================

class ProgressEntry(BaseModel):
    """
    Single entry in a participant's progress timeline.
    
    Attributes:
        event_id: Event identifier.
        event_name: Event display name.
        event_date: When the event occurred.
        team_name: Team the user was part of.
        overall_score: User's/team's overall score.
        percentile_rank: Percentile in that event.
        rank: Absolute rank in that event.
        total_participants: Total teams in event.
        skills_snapshot: Skill scores at that time.
        improvement_from_previous: Score change from last event.
    """
    model_config = ConfigDict(populate_by_name=True)
    
    event_id: str = Field(..., description="Event identifier")
    event_name: str = Field(..., description="Event name")
    event_date: datetime = Field(..., description="Event date")
    team_name: str = Field(..., description="Team name")
    overall_score: float = Field(..., ge=0, le=100, description="Score")
    percentile_rank: float = Field(..., ge=0, le=100, description="Percentile")
    rank: int = Field(..., ge=1, description="Rank")
    total_participants: int = Field(..., ge=1, description="Total teams")
    skills_snapshot: Dict[str, float] = Field(
        default_factory=dict,
        description="Skill scores at this event"
    )
    improvement_from_previous: Optional[float] = Field(
        None,
        description="Score change from previous event"
    )


class ProgressTimeline(BaseModel):
    """
    Complete progress timeline for a participant.
    
    Attributes:
        user_id: User being analyzed.
        total_events_participated: Number of events.
        entries: Chronological list of event entries.
        overall_improvement: Total score improvement (first to last).
        avg_percentile: Average percentile across events.
        best_performance: Highest scoring event.
        consistency_score: How consistent the participant is (0-100).
        trend: Overall performance trend.
        generated_at: Timestamp of analysis.
    """
    model_config = ConfigDict(populate_by_name=True)
    
    user_id: str = Field(..., description="User identifier")
    total_events_participated: int = Field(..., ge=0, description="Events count")
    entries: List[ProgressEntry] = Field(default_factory=list)
    overall_improvement: float = Field(..., description="Score change first→last")
    avg_percentile: float = Field(..., ge=0, le=100, description="Avg percentile")
    best_performance: Optional[ProgressEntry] = Field(None)
    consistency_score: float = Field(..., ge=0, le=100, description="Consistency")
    trend: str = Field(..., description="'improving', 'declining', 'stable'")
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# EXPORT MODELS
# =============================================================================

class ExportConfig(BaseModel):
    """
    Configuration for data export operations.
    
    Attributes:
        event_id: Event to export data from.
        columns: List of column names to include (empty = all).
        format: Export format ('csv', 'json').
        include_ai_feedback: Whether to include AI feedback text.
        include_metadata: Whether to include submission metadata.
        filter_round: Optional round filter (ppt/repo/viva).
        filter_min_score: Optional minimum score filter.
        filter_max_score: Optional maximum score filter.
    """
    model_config = ConfigDict(populate_by_name=True)
    
    event_id: str = Field(..., description="Event identifier")
    columns: List[str] = Field(default_factory=list, description="Columns to export")
    format: str = Field("csv", description="Export format")
    include_ai_feedback: bool = Field(True, description="Include AI text")
    include_metadata: bool = Field(True, description="Include metadata")
    filter_round: Optional[str] = Field(None, description="Round filter")
    filter_min_score: Optional[float] = Field(None, ge=0, le=100)
    filter_max_score: Optional[float] = Field(None, ge=0, le=100)


class ExportResult(BaseModel):
    """
    Result of an export operation.
    
    Attributes:
        event_id: Event exported.
        row_count: Number of rows exported.
        columns_included: List of column names in export.
        file_size_bytes: Size of generated file.
        format: Export format used.
        generated_at: Timestamp of export.
    """
    model_config = ConfigDict(populate_by_name=True)
    
    event_id: str = Field(..., description="Event identifier")
    row_count: int = Field(..., ge=0, description="Rows exported")
    columns_included: List[str] = Field(default_factory=list)
    file_size_bytes: int = Field(..., ge=0, description="File size")
    format: str = Field(..., description="Export format")
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# API RESPONSE WRAPPER MODELS
# =============================================================================

class AnalyticsResponse(BaseModel):
    """
    Standard wrapper for analytics API responses.
    
    Attributes:
        success: Whether the operation succeeded.
        data: The analytics data payload.
        message: Optional message.
        errors: List of any errors encountered.
    """
    model_config = ConfigDict(populate_by_name=True)
    
    success: bool = Field(True, description="Operation success")
    data: Optional[Any] = Field(None, description="Response data")
    message: Optional[str] = Field(None, description="Status message")
    errors: List[str] = Field(default_factory=list, description="Error messages")
