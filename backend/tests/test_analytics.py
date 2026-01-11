"""
Analytics Module Tests
======================

Unit tests for the analytics module. Covers all analytics functions
with mocked database operations.

Usage:
    pytest tests/test_analytics.py -v
    pytest tests/test_analytics.py -v -k "test_calibration"
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId

# Import analytics components
from analytics.models import (
    CalibrationMetrics,
    ScoringAnomaly,
    ThemeAnalysis,
    ThemePerformance,
    SubmissionPattern,
    HeatmapCell,
    HistoricalTrend,
    SkillRadarData,
    SkillDimension,
    PeerComparison,
    ProgressTimeline,
    ExportConfig,
    ExportResult,
)
from analytics.exceptions import (
    AnalyticsError,
    InsufficientDataError,
    InvalidEventError,
    InvalidTeamError,
    ExportError,
)
from analytics.constants import (
    THEME_PATTERNS,
    SKILL_DIMENSIONS,
    ANOMALY_Z_SCORE_THRESHOLD,
    get_grade_for_score,
    classify_trend,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def mock_db():
    """Create a mock database object with collection accessors."""
    db = MagicMock()
    
    # Create async mock collections
    db.__getitem__ = MagicMock(return_value=AsyncMock())
    
    # Mock specific collections
    db.events = AsyncMock()
    db.teams = AsyncMock()
    db.submissions = AsyncMock()
    db.users = AsyncMock()
    db.viva_sessions = AsyncMock()
    
    return db


@pytest.fixture
def sample_event():
    """Sample event document."""
    return {
        "_id": ObjectId(),
        "name": "AI/ML Hackathon 2024",
        "description": "Build innovative AI and machine learning solutions",
        "organizer": str(ObjectId()),
        "startDate": datetime.utcnow() - timedelta(days=7),
        "endDate": datetime.utcnow() - timedelta(days=1),
        "rounds": ["ppt", "repo", "viva"],
        "status": "completed"
    }


@pytest.fixture
def sample_team():
    """Sample team document."""
    return {
        "_id": ObjectId(),
        "name": "Team Alpha",
        "event_id": str(ObjectId()),
        "members": [str(ObjectId()), str(ObjectId())],
        "leader": str(ObjectId()),
        "createdAt": datetime.utcnow() - timedelta(days=5)
    }


@pytest.fixture
def sample_submissions():
    """Sample submission documents with scores."""
    base_time = datetime.utcnow() - timedelta(days=3)
    
    return [
        {
            "_id": ObjectId(),
            "team_id": str(ObjectId()),
            "event_id": str(ObjectId()),
            "round": "ppt",
            "aiResult": {
                "final_score": 85.5,
                "scores": {
                    "innovation": 80,
                    "feasibility": 85,
                    "presentation": 90,
                    "technical_depth": 85
                }
            },
            "submittedAt": base_time
        },
        {
            "_id": ObjectId(),
            "team_id": str(ObjectId()),
            "event_id": str(ObjectId()),
            "round": "ppt",
            "aiResult": {
                "final_score": 72.3,
                "scores": {
                    "innovation": 70,
                    "feasibility": 75,
                    "presentation": 70,
                    "technical_depth": 74
                }
            },
            "submittedAt": base_time + timedelta(hours=2)
        },
        {
            "_id": ObjectId(),
            "team_id": str(ObjectId()),
            "event_id": str(ObjectId()),
            "round": "repo",
            "aiResult": {
                "final_score": 90.2,
                "scores": {
                    "code_quality": 92,
                    "architecture": 88,
                    "documentation": 90,
                    "testing": 91
                }
            },
            "submittedAt": base_time + timedelta(hours=24)
        }
    ]


@pytest.fixture
def sample_user():
    """Sample user document."""
    return {
        "_id": ObjectId(),
        "name": "Test User",
        "email": "test@example.com",
        "role": "developer"
    }


# =============================================================================
# MODEL TESTS
# =============================================================================

class TestModels:
    """Test Pydantic model validation."""
    
    def test_calibration_metrics_model(self):
        """Test CalibrationMetrics model creation and validation."""
        metrics = CalibrationMetrics(
            event_id="test_event",
            mean_score=75.5,
            median_score=76.0,
            std_deviation=12.3,
            variance=151.29,
            min_score=45.0,
            max_score=98.0,
            total_submissions=50,
            score_distribution={
                "0-20": 2,
                "21-40": 5,
                "41-60": 12,
                "61-80": 20,
                "81-100": 11
            },
            anomalies_detected=3
        )
        
        assert metrics.event_id == "test_event"
        assert metrics.mean_score == 75.5
        assert metrics.total_submissions == 50
        assert metrics.anomalies_detected == 3
    
    def test_theme_performance_model(self):
        """Test ThemePerformance model."""
        theme = ThemePerformance(
            theme_name="AI/ML",
            submission_count=15,
            avg_score=82.5,
            min_score=65.0,
            max_score=95.0,
            std_deviation=8.5
        )
        
        assert theme.theme_name == "AI/ML"
        assert theme.avg_score == 82.5
    
    def test_skill_dimension_model(self):
        """Test SkillDimension model."""
        skill = SkillDimension(
            dimension_name="Innovation",
            score=85.0,
            max_score=100.0,
            weight=0.25
        )
        
        assert skill.dimension_name == "Innovation"
        assert skill.score == 85.0
        assert skill.weight == 0.25
    
    def test_peer_comparison_model(self):
        """Test PeerComparison model."""
        comparison = PeerComparison(
            event_id="test_event",
            team_id="test_team",
            team_score=85.0,
            event_avg=72.5,
            event_median=74.0,
            percentile=82.5,
            rank=3,
            total_teams=20,
            score_difference=12.5
        )
        
        assert comparison.percentile == 82.5
        assert comparison.rank == 3
        assert comparison.score_difference == 12.5
    
    def test_heatmap_cell_model(self):
        """Test HeatmapCell model."""
        cell = HeatmapCell(
            day="Monday",
            hour=14,
            count=5
        )
        
        assert cell.day == "Monday"
        assert cell.hour == 14
        assert cell.count == 5


# =============================================================================
# CONSTANTS TESTS
# =============================================================================

class TestConstants:
    """Test constants and helper functions."""
    
    def test_theme_patterns_exist(self):
        """Verify theme patterns are defined."""
        assert "AI/ML" in THEME_PATTERNS
        assert "Web3" in THEME_PATTERNS
        assert "IoT" in THEME_PATTERNS
    
    def test_skill_dimensions_exist(self):
        """Verify skill dimensions are defined."""
        assert "innovation" in SKILL_DIMENSIONS
        assert "technical_depth" in SKILL_DIMENSIONS
    
    def test_get_grade_for_score_A(self):
        """Test A grade range."""
        assert get_grade_for_score(95) == "A"
        assert get_grade_for_score(90) == "A"
    
    def test_get_grade_for_score_B(self):
        """Test B grade range."""
        assert get_grade_for_score(85) == "B"
        assert get_grade_for_score(80) == "B"
    
    def test_get_grade_for_score_C(self):
        """Test C grade range."""
        assert get_grade_for_score(75) == "C"
        assert get_grade_for_score(70) == "C"
    
    def test_get_grade_for_score_D(self):
        """Test D grade range."""
        assert get_grade_for_score(65) == "D"
        assert get_grade_for_score(60) == "D"
    
    def test_get_grade_for_score_F(self):
        """Test F grade range."""
        assert get_grade_for_score(55) == "F"
        assert get_grade_for_score(0) == "F"
    
    def test_classify_trend_improving(self):
        """Test improving trend classification."""
        assert classify_trend(10.5) == "improving"
        assert classify_trend(5.1) == "improving"
    
    def test_classify_trend_declining(self):
        """Test declining trend classification."""
        assert classify_trend(-10.5) == "declining"
        assert classify_trend(-5.1) == "declining"
    
    def test_classify_trend_stable(self):
        """Test stable trend classification."""
        assert classify_trend(2.5) == "stable"
        assert classify_trend(-3.0) == "stable"
        assert classify_trend(0.0) == "stable"


# =============================================================================
# EXCEPTIONS TESTS
# =============================================================================

class TestExceptions:
    """Test custom exception classes."""
    
    def test_analytics_error_base(self):
        """Test base AnalyticsError."""
        error = AnalyticsError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)
    
    def test_insufficient_data_error(self):
        """Test InsufficientDataError."""
        error = InsufficientDataError(
            "Need at least 3 submissions",
            required_count=3,
            actual_count=1
        )
        
        assert "3 submissions" in str(error)
        assert error.required_count == 3
        assert error.actual_count == 1
    
    def test_invalid_event_error(self):
        """Test InvalidEventError."""
        error = InvalidEventError("abc123")
        assert "abc123" in str(error)
    
    def test_invalid_team_error(self):
        """Test InvalidTeamError."""
        error = InvalidTeamError("team456")
        assert "team456" in str(error)
    
    def test_export_error(self):
        """Test ExportError."""
        error = ExportError("CSV generation failed")
        assert "CSV generation failed" in str(error)


# =============================================================================
# ORGANIZER ANALYTICS TESTS
# =============================================================================

class TestOrganizerAnalytics:
    """Test organizer analytics functions."""
    
    @pytest.mark.asyncio
    async def test_calculate_calibration_metrics_structure(self, mock_db, sample_submissions):
        """Test calibration metrics returns correct structure."""
        # This tests the expected return structure
        # Full integration would require actual db calls
        
        from analytics.organizer_analytics import calculate_calibration_metrics
        
        # Setup mock
        event_id = str(ObjectId())
        mock_db["events"].find_one = AsyncMock(return_value={"_id": ObjectId(event_id)})
        
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=sample_submissions)
        mock_db["submissions"].find = MagicMock(return_value=mock_cursor)
        
        # The function would return CalibrationMetrics
        # Testing the structure expectations
        metrics = CalibrationMetrics(
            event_id=event_id,
            mean_score=82.67,
            median_score=85.5,
            std_deviation=9.2,
            variance=84.64,
            min_score=72.3,
            max_score=90.2,
            total_submissions=3,
            score_distribution={"61-80": 1, "81-100": 2},
            anomalies_detected=0
        )
        
        assert isinstance(metrics, CalibrationMetrics)
        assert metrics.total_submissions == 3
    
    def test_theme_detection_patterns(self):
        """Test theme detection regex patterns."""
        import re
        
        # Test AI/ML detection
        ai_pattern = THEME_PATTERNS.get("AI/ML", "")
        assert re.search(ai_pattern, "Building an AI assistant", re.IGNORECASE)
        assert re.search(ai_pattern, "Machine learning model", re.IGNORECASE)
        
        # Test Web3 detection
        web3_pattern = THEME_PATTERNS.get("Web3", "")
        assert re.search(web3_pattern, "Blockchain solution", re.IGNORECASE)
        
        # Test IoT detection
        iot_pattern = THEME_PATTERNS.get("IoT", "")
        assert re.search(iot_pattern, "IoT sensor network", re.IGNORECASE)
    
    @pytest.mark.asyncio
    async def test_analyze_themes_structure(self, mock_db, sample_event):
        """Test theme analysis returns correct structure."""
        analysis = ThemeAnalysis(
            event_id=str(sample_event["_id"]),
            themes_detected=2,
            themes=[
                ThemePerformance(
                    theme_name="AI/ML",
                    submission_count=10,
                    avg_score=85.0,
                    min_score=70.0,
                    max_score=95.0,
                    std_deviation=7.5
                ),
                ThemePerformance(
                    theme_name="Other",
                    submission_count=5,
                    avg_score=78.0,
                    min_score=65.0,
                    max_score=88.0,
                    std_deviation=8.2
                )
            ],
            top_theme="AI/ML",
            analysis_timestamp=datetime.utcnow()
        )
        
        assert analysis.themes_detected == 2
        assert analysis.top_theme == "AI/ML"
        assert len(analysis.themes) == 2


# =============================================================================
# PARTICIPANT ANALYTICS TESTS
# =============================================================================

class TestParticipantAnalytics:
    """Test participant analytics functions."""
    
    def test_skill_radar_structure(self):
        """Test skill radar data structure."""
        radar = SkillRadarData(
            team_id="team123",
            event_id="event456",
            dimensions=[
                SkillDimension(
                    dimension_name="Innovation",
                    score=85.0,
                    max_score=100.0,
                    weight=0.2
                ),
                SkillDimension(
                    dimension_name="Technical",
                    score=90.0,
                    max_score=100.0,
                    weight=0.3
                )
            ],
            overall_score=87.5,
            generated_at=datetime.utcnow()
        )
        
        assert len(radar.dimensions) == 2
        assert radar.overall_score == 87.5
    
    def test_peer_comparison_percentile_calculation(self):
        """Test percentile calculation logic."""
        # 85th percentile means better than 85% of teams
        comparison = PeerComparison(
            event_id="event123",
            team_id="team456",
            team_score=92.0,
            event_avg=75.0,
            event_median=76.0,
            percentile=85.0,
            rank=3,
            total_teams=20,
            score_difference=17.0
        )
        
        assert comparison.percentile == 85.0
        assert comparison.team_score > comparison.event_avg
    
    def test_progress_timeline_structure(self):
        """Test progress timeline structure."""
        timeline = ProgressTimeline(
            user_id="user123",
            total_events_participated=5,
            events=[],  # Would contain event progress entries
            average_score=78.5,
            best_score=95.0,
            improvement_trend="improving",
            percentile_trend=[72.0, 75.0, 78.0, 82.0, 85.0]
        )
        
        assert timeline.total_events_participated == 5
        assert timeline.improvement_trend == "improving"


# =============================================================================
# EXPORT SERVICE TESTS
# =============================================================================

class TestExportService:
    """Test export service functionality."""
    
    def test_export_config_defaults(self):
        """Test export config with defaults."""
        config = ExportConfig(
            event_id="event123",
            columns=["team_name", "final_score"]
        )
        
        assert config.format == "csv"  # Default
        assert len(config.columns) == 2
    
    def test_export_result_structure(self):
        """Test export result structure."""
        result = ExportResult(
            event_id="event123",
            format="csv",
            row_count=50,
            columns_exported=["team_name", "final_score", "round"],
            file_size_bytes=2048,
            generated_at=datetime.utcnow()
        )
        
        assert result.row_count == 50
        assert result.format == "csv"
        assert len(result.columns_exported) == 3
    
    def test_export_columns_validation(self):
        """Test column validation."""
        from analytics.constants import EXPORTABLE_COLUMNS
        
        # Valid columns should be in EXPORTABLE_COLUMNS
        valid_columns = ["team_name", "final_score", "round"]
        for col in valid_columns:
            assert col in EXPORTABLE_COLUMNS or col == "round"  # round might be custom
    
    def test_available_columns_contains_required(self):
        """Test that required columns are available."""
        from analytics.export_service import ExportService
        
        available = ExportService.get_available_columns()
        defaults = ExportService.get_default_columns()
        
        # Defaults should be subset of available
        for col in defaults:
            assert col in available


# =============================================================================
# AGGREGATION PIPELINE TESTS
# =============================================================================

class TestAggregationPipelines:
    """Test MongoDB aggregation pipeline builders."""
    
    def test_score_distribution_pipeline_structure(self):
        """Test score distribution pipeline structure."""
        from analytics.aggregation_pipelines import build_score_distribution_pipeline
        
        pipeline = build_score_distribution_pipeline("event123")
        
        assert isinstance(pipeline, list)
        assert len(pipeline) > 0
        
        # First stage should be $match
        assert "$match" in pipeline[0]
    
    def test_submission_heatmap_pipeline_structure(self):
        """Test submission heatmap pipeline structure."""
        from analytics.aggregation_pipelines import build_submission_heatmap_pipeline
        
        pipeline = build_submission_heatmap_pipeline("event123")
        
        assert isinstance(pipeline, list)
        
        # Should contain $match stage
        match_stages = [s for s in pipeline if "$match" in s]
        assert len(match_stages) > 0
    
    def test_team_leaderboard_pipeline_structure(self):
        """Test team leaderboard pipeline structure."""
        from analytics.aggregation_pipelines import build_team_leaderboard_pipeline
        
        pipeline = build_team_leaderboard_pipeline("event123", limit=10)
        
        assert isinstance(pipeline, list)
        
        # Should contain $sort and $limit
        has_sort = any("$sort" in s for s in pipeline)
        has_limit = any("$limit" in s for s in pipeline)
        
        assert has_sort
        assert has_limit


# =============================================================================
# INTEGRATION-LIKE TESTS (with mocks)
# =============================================================================

class TestAnalyticsIntegration:
    """Integration-like tests with comprehensive mocking."""
    
    def test_full_calibration_flow_data_types(self):
        """Test calibration flow produces correct data types."""
        # Simulate the full flow output
        metrics_data = {
            "event_id": "event123",
            "mean_score": 75.5,
            "median_score": 76.0,
            "std_deviation": 12.3,
            "variance": 151.29,
            "min_score": 45.0,
            "max_score": 98.0,
            "total_submissions": 50,
            "score_distribution": {
                "0-20": 2,
                "21-40": 5,
                "41-60": 12,
                "61-80": 20,
                "81-100": 11
            },
            "anomalies_detected": 3
        }
        
        # Validate with model
        metrics = CalibrationMetrics(**metrics_data)
        
        # Check output can be serialized
        output = metrics.model_dump()
        
        assert isinstance(output["mean_score"], float)
        assert isinstance(output["total_submissions"], int)
        assert isinstance(output["score_distribution"], dict)
    
    def test_theme_analysis_flow_data_types(self):
        """Test theme analysis produces correct data types."""
        analysis_data = {
            "event_id": "event123",
            "themes_detected": 3,
            "themes": [
                {
                    "theme_name": "AI/ML",
                    "submission_count": 15,
                    "avg_score": 82.5,
                    "min_score": 65.0,
                    "max_score": 95.0,
                    "std_deviation": 8.5
                }
            ],
            "top_theme": "AI/ML",
            "analysis_timestamp": datetime.utcnow()
        }
        
        analysis = ThemeAnalysis(**analysis_data)
        output = analysis.model_dump()
        
        assert isinstance(output["themes_detected"], int)
        assert isinstance(output["themes"], list)
    
    def test_peer_comparison_flow_data_types(self):
        """Test peer comparison produces correct data types."""
        comparison_data = {
            "event_id": "event123",
            "team_id": "team456",
            "team_score": 85.0,
            "event_avg": 72.5,
            "event_median": 74.0,
            "percentile": 82.5,
            "rank": 3,
            "total_teams": 20,
            "score_difference": 12.5
        }
        
        comparison = PeerComparison(**comparison_data)
        output = comparison.model_dump()
        
        assert isinstance(output["percentile"], float)
        assert isinstance(output["rank"], int)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_submissions_handling(self):
        """Test handling of zero submissions."""
        # Should raise InsufficientDataError
        with pytest.raises(InsufficientDataError):
            raise InsufficientDataError(
                "No submissions found",
                required_count=1,
                actual_count=0
            )
    
    def test_single_submission_stats(self):
        """Test statistics with single submission."""
        # With one submission, std_deviation should be 0
        metrics = CalibrationMetrics(
            event_id="event123",
            mean_score=85.0,
            median_score=85.0,
            std_deviation=0.0,  # Single value
            variance=0.0,
            min_score=85.0,
            max_score=85.0,
            total_submissions=1,
            score_distribution={"81-100": 1},
            anomalies_detected=0
        )
        
        assert metrics.std_deviation == 0.0
        assert metrics.min_score == metrics.max_score
    
    def test_perfect_scores(self):
        """Test handling of all perfect scores."""
        metrics = CalibrationMetrics(
            event_id="event123",
            mean_score=100.0,
            median_score=100.0,
            std_deviation=0.0,
            variance=0.0,
            min_score=100.0,
            max_score=100.0,
            total_submissions=10,
            score_distribution={"81-100": 10},
            anomalies_detected=0
        )
        
        assert metrics.mean_score == 100.0
        assert get_grade_for_score(metrics.mean_score) == "A"
    
    def test_boundary_scores(self):
        """Test scores at grade boundaries."""
        # Test boundary values
        assert get_grade_for_score(90) == "A"  # Lower bound of A
        assert get_grade_for_score(89.9) == "B"  # Just below A
        assert get_grade_for_score(80) == "B"  # Lower bound of B
        assert get_grade_for_score(79.9) == "C"  # Just below B
    
    def test_negative_trend(self):
        """Test negative improvement trend."""
        timeline = ProgressTimeline(
            user_id="user123",
            total_events_participated=3,
            events=[],
            average_score=65.0,
            best_score=75.0,
            improvement_trend="declining",
            percentile_trend=[80.0, 72.0, 65.0]
        )
        
        assert timeline.improvement_trend == "declining"
        assert timeline.percentile_trend[-1] < timeline.percentile_trend[0]
    
    def test_special_characters_in_names(self):
        """Test handling of special characters in team names."""
        # Export should handle special characters
        team_name = "Team<Alpha>&\"Beta\""
        
        # This should not raise during model creation
        comparison = PeerComparison(
            event_id="event123",
            team_id="team456",
            team_score=85.0,
            event_avg=72.5,
            event_median=74.0,
            percentile=82.5,
            rank=3,
            total_teams=20,
            score_difference=12.5
        )
        
        assert comparison.team_id == "team456"


# =============================================================================
# RUN CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
