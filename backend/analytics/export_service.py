"""
Export Service Module
=====================

This module provides data export functionality for the EvalX analytics
system. It supports exporting evaluation data to CSV format with
dynamic column selection.

Features:
    - Dynamic column selection
    - Multiple export formats (CSV, JSON)
    - Streaming export for large datasets
    - Data filtering and transformation

Dependencies:
    - pandas: Data manipulation and CSV generation
    - Motor: Async MongoDB operations
"""

import logging
import io
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
import json

import pandas as pd
from bson import ObjectId

from analytics.models import ExportConfig, ExportResult
from analytics.exceptions import (
    ExportError,
    InvalidEventError,
    InsufficientDataError,
)
from analytics.constants import (
    EXPORTABLE_COLUMNS,
    DEFAULT_EXPORT_COLUMNS,
    MAX_EXPORT_ROWS,
    SUPPORTED_EXPORT_FORMATS,
    get_grade_for_score,
)

# Module logger
logger = logging.getLogger(__name__)


class ExportService:
    """
    Service class for exporting analytics data.
    
    Provides methods to export evaluation data from MongoDB to various
    formats with support for dynamic column selection and filtering.
    
    Attributes:
        db: MongoDB database instance.
    
    Example:
        >>> service = ExportService(db)
        >>> csv_data, result = await service.export_evaluations(
        ...     event_id="event123",
        ...     columns=["team_name", "overall_score", "grade"],
        ...     export_format="csv"
        ... )
        >>> print(f"Exported {result.row_count} rows")
    """
    
    def __init__(self, db: Any) -> None:
        """
        Initialize the ExportService.
        
        Args:
            db: MongoDB database instance from Motor.
        """
        self.db = db
        logger.info("ExportService initialized")
    
    @staticmethod
    def get_available_columns() -> Dict[str, Dict[str, str]]:
        """
        Get list of all available columns for export.
        
        Returns:
            Dictionary mapping column names to their metadata
            (type and description).
        
        Example:
            >>> columns = ExportService.get_available_columns()
            >>> for name, info in columns.items():
            ...     print(f"{name}: {info['description']}")
        """
        return EXPORTABLE_COLUMNS.copy()
    
    @staticmethod
    def get_default_columns() -> List[str]:
        """
        Get default columns for quick export.
        
        Returns:
            List of default column names.
        """
        return DEFAULT_EXPORT_COLUMNS.copy()
    
    def validate_columns(
        self,
        requested_columns: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Validate requested columns against available columns.
        
        Args:
            requested_columns: List of column names requested for export.
        
        Returns:
            Tuple of (valid_columns, invalid_columns).
        
        Example:
            >>> valid, invalid = service.validate_columns(["team_name", "xyz"])
            >>> # valid = ["team_name"], invalid = ["xyz"]
        """
        available = set(EXPORTABLE_COLUMNS.keys())
        requested = set(requested_columns)
        
        valid_columns = list(requested & available)
        invalid_columns = list(requested - available)
        
        # Preserve original order for valid columns
        valid_columns = [c for c in requested_columns if c in available]
        
        return valid_columns, invalid_columns
    
    async def export_evaluations(
        self,
        event_id: str,
        columns: Optional[List[str]] = None,
        export_format: str = "csv",
        round_filter: Optional[str] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        include_ai_feedback: bool = True
    ) -> Tuple[str, ExportResult]:
        """
        Export evaluation data for an event.
        
        Retrieves submission data and exports it in the requested format
        with the specified columns.
        
        Args:
            event_id: Event identifier to export data from.
            columns: List of columns to include. If None or empty,
                uses default columns.
            export_format: Export format ('csv' or 'json').
            round_filter: Optional filter for specific round.
            min_score: Optional minimum score filter.
            max_score: Optional maximum score filter.
            include_ai_feedback: Whether to include AI feedback text fields.
        
        Returns:
            Tuple of (export_data_string, ExportResult metadata).
        
        Raises:
            InvalidEventError: If event not found.
            ExportError: If export fails or invalid columns specified.
            InsufficientDataError: If no data to export.
        
        Example:
            >>> csv_data, result = await service.export_evaluations(
            ...     event_id="event123",
            ...     columns=["team_name", "overall_score"],
            ...     export_format="csv"
            ... )
        """
        logger.info(
            f"Starting export for event {event_id}: format={export_format}, "
            f"columns={columns}, round={round_filter}"
        )
        
        # Validate format
        if export_format not in SUPPORTED_EXPORT_FORMATS:
            raise ExportError(
                message=f"Unsupported export format: {export_format}",
                export_format=export_format,
                context={"supported_formats": list(SUPPORTED_EXPORT_FORMATS)}
            )
        
        # Validate event exists
        events_collection = self.db["events"]
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
        
        # Handle columns
        if not columns:
            columns = DEFAULT_EXPORT_COLUMNS.copy()
            logger.debug(f"Using default columns: {columns}")
        
        # Filter out AI feedback columns if not requested
        if not include_ai_feedback:
            ai_columns = {"llm_feedback", "mentor_summary", "rewrite_suggestions"}
            columns = [c for c in columns if c not in ai_columns]
        
        # Validate columns
        valid_columns, invalid_columns = self.validate_columns(columns)
        
        if invalid_columns:
            logger.warning(f"Invalid columns requested: {invalid_columns}")
            raise ExportError(
                message=f"Invalid columns: {', '.join(invalid_columns)}",
                export_format=export_format,
                invalid_columns=invalid_columns
            )
        
        if not valid_columns:
            valid_columns = DEFAULT_EXPORT_COLUMNS.copy()
        
        # Fetch data
        raw_data = await self._fetch_export_data(
            event_id=event_id,
            round_filter=round_filter,
            min_score=min_score,
            max_score=max_score
        )
        
        if not raw_data:
            raise InsufficientDataError(
                message="No data available for export",
                required_count=1,
                actual_count=0
            )
        
        # Limit rows
        if len(raw_data) > MAX_EXPORT_ROWS:
            logger.warning(
                f"Truncating export from {len(raw_data)} to {MAX_EXPORT_ROWS} rows"
            )
            raw_data = raw_data[:MAX_EXPORT_ROWS]
        
        # Transform data
        transformed_data = await self._transform_data(raw_data, valid_columns)
        
        # Generate export
        if export_format == "csv":
            export_data = self._generate_csv(transformed_data, valid_columns)
        else:  # json
            export_data = self._generate_json(transformed_data)
        
        # Create result metadata
        result = ExportResult(
            event_id=event_id,
            row_count=len(transformed_data),
            columns_included=valid_columns,
            file_size_bytes=len(export_data.encode('utf-8')),
            format=export_format,
            generated_at=datetime.utcnow()
        )
        
        logger.info(
            f"Export complete: {result.row_count} rows, "
            f"{result.file_size_bytes} bytes, format={export_format}"
        )
        
        return export_data, result
    
    async def _fetch_export_data(
        self,
        event_id: str,
        round_filter: Optional[str] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch raw submission data for export.
        
        Args:
            event_id: Event identifier.
            round_filter: Optional round filter.
            min_score: Optional minimum score filter.
            max_score: Optional maximum score filter.
        
        Returns:
            List of submission documents with team info.
        """
        submissions_collection = self.db["submissions"]
        teams_collection = self.db["teams"]
        
        # Build query
        query: Dict[str, Any] = {"eventId": event_id}
        
        if round_filter:
            query["roundId"] = round_filter
        
        # Fetch submissions
        cursor = submissions_collection.find(query)
        submissions = await cursor.to_list(MAX_EXPORT_ROWS + 100)  # Buffer for filtering
        
        logger.debug(f"Fetched {len(submissions)} submissions for export")
        
        # Enrich with team data and apply score filters
        enriched_data = []
        
        for sub in submissions:
            # Extract score for filtering
            score = self._extract_score(sub)
            
            # Apply score filters
            if min_score is not None and (score is None or score < min_score):
                continue
            if max_score is not None and (score is None or score > max_score):
                continue
            
            # Get team info
            team_id = sub.get("teamId", "")
            team_name = "Unknown Team"
            
            if team_id:
                try:
                    team = await teams_collection.find_one({"_id": ObjectId(team_id)})
                    if not team:
                        team = await teams_collection.find_one({"teamId": team_id})
                    if team:
                        team_name = team.get("teamName", "Unknown Team")
                except Exception:
                    pass
            
            # Add team name to submission
            sub["_team_name"] = team_name
            sub["_extracted_score"] = score
            
            enriched_data.append(sub)
        
        logger.debug(f"After filtering: {len(enriched_data)} submissions")
        
        return enriched_data
    
    def _extract_score(self, sub: Dict) -> Optional[float]:
        """
        Extract score from submission document.
        
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
    
    async def _transform_data(
        self,
        raw_data: List[Dict],
        columns: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Transform raw submission data to export format.
        
        Maps raw MongoDB fields to standardized export columns.
        
        Args:
            raw_data: List of raw submission documents.
            columns: List of columns to include.
        
        Returns:
            List of transformed row dictionaries.
        """
        transformed = []
        
        for sub in raw_data:
            row = {}
            ai_result = sub.get("aiResult", {}) or {}
            
            for col in columns:
                value = self._extract_column_value(sub, ai_result, col)
                row[col] = value
            
            transformed.append(row)
        
        return transformed
    
    def _extract_column_value(
        self,
        sub: Dict,
        ai_result: Dict,
        column: str
    ) -> Any:
        """
        Extract value for a specific column from submission data.
        
        Args:
            sub: Submission document.
            ai_result: AI result sub-document.
            column: Column name to extract.
        
        Returns:
            Extracted value or None.
        """
        # Handle each column type
        column_extractors = {
            "submission_id": lambda: str(sub.get("_id", "")),
            "event_id": lambda: sub.get("eventId", ""),
            "team_id": lambda: sub.get("teamId", ""),
            "team_name": lambda: sub.get("_team_name", "Unknown"),
            "round_id": lambda: sub.get("roundId", ""),
            "submitted_at": lambda: self._format_datetime(sub.get("submittedAt")),
            
            # Scores
            "overall_score": lambda: sub.get("_extracted_score"),
            "final_score": lambda: ai_result.get("final_score"),
            "risk_score": lambda: ai_result.get("risk_score"),
            "pylint_score": lambda: ai_result.get("pylint_score"),
            
            # PPT-specific
            "ppt_clarity_score": lambda: self._safe_get_nested(
                sub, ["pptAnalysis", "clarity_score"]
            ),
            "ppt_design_score": lambda: self._safe_get_nested(
                sub, ["pptAnalysis", "design_score"]
            ),
            "ppt_storytelling_score": lambda: self._safe_get_nested(
                sub, ["pptAnalysis", "storytelling_score"]
            ),
            
            # Repo-specific
            "repo_url": lambda: sub.get("repo", ""),
            "files_analyzed": lambda: ai_result.get("files_analyzed"),
            "plagiarism_percentage": lambda: self._safe_get_nested(
                ai_result, ["plagiarism", "percentage"]
            ) or self._safe_get_nested(ai_result, ["plagiarism", "score"]),
            
            # AI Feedback
            "llm_feedback": lambda: ai_result.get("llm_feedback", ""),
            "mentor_summary": lambda: ai_result.get("mentor_summary_markdown", ""),
            "rewrite_suggestions": lambda: ai_result.get(
                "rewrite_suggestions_markdown", ""
            ),
            
            # Computed
            "percentile_rank": lambda: sub.get("_percentile"),  # Would need computation
            "grade": lambda: get_grade_for_score(sub.get("_extracted_score", 0) or 0),
            "theme": lambda: sub.get("_theme", ""),  # Would need computation
        }
        
        extractor = column_extractors.get(column)
        if extractor:
            try:
                return extractor()
            except Exception as e:
                logger.debug(f"Error extracting column {column}: {e}")
                return None
        
        return None
    
    def _safe_get_nested(self, data: Dict, keys: List[str]) -> Any:
        """
        Safely get nested dictionary value.
        
        Args:
            data: Dictionary to traverse.
            keys: List of keys to follow.
        
        Returns:
            Value at path or None.
        """
        current = data
        for key in keys:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
            if current is None:
                return None
        return current
    
    def _format_datetime(self, dt: Any) -> str:
        """
        Format datetime value for export.
        
        Args:
            dt: Datetime value (datetime object or string).
        
        Returns:
            Formatted datetime string.
        """
        if dt is None:
            return ""
        
        if isinstance(dt, str):
            return dt
        
        if isinstance(dt, datetime):
            return dt.isoformat()
        
        return str(dt)
    
    def _generate_csv(
        self,
        data: List[Dict],
        columns: List[str]
    ) -> str:
        """
        Generate CSV string from data.
        
        Args:
            data: List of row dictionaries.
            columns: List of column names (determines order).
        
        Returns:
            CSV formatted string.
        """
        if not data:
            # Return header only
            return ",".join(columns) + "\n"
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Ensure columns are in correct order
        df = df.reindex(columns=columns)
        
        # Generate CSV
        output = io.StringIO()
        df.to_csv(output, index=False, encoding='utf-8')
        
        return output.getvalue()
    
    def _generate_json(self, data: List[Dict]) -> str:
        """
        Generate JSON string from data.
        
        Args:
            data: List of row dictionaries.
        
        Returns:
            JSON formatted string.
        """
        # Handle datetime serialization
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, ObjectId):
                return str(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        return json.dumps(data, default=json_serializer, indent=2)
    
    async def generate_export_stream(
        self,
        event_id: str,
        columns: Optional[List[str]] = None,
        chunk_size: int = 1000
    ):
        """
        Generate CSV export as an async generator for streaming.
        
        Yields chunks of CSV data for memory-efficient large exports.
        
        Args:
            event_id: Event identifier.
            columns: Columns to include.
            chunk_size: Number of rows per chunk.
        
        Yields:
            CSV data chunks as strings.
        
        Example:
            >>> async for chunk in service.generate_export_stream("event123"):
            ...     await response.write(chunk.encode())
        """
        logger.info(f"Starting streaming export for event {event_id}")
        
        # Validate columns
        if not columns:
            columns = DEFAULT_EXPORT_COLUMNS.copy()
        
        valid_columns, _ = self.validate_columns(columns)
        if not valid_columns:
            valid_columns = DEFAULT_EXPORT_COLUMNS.copy()
        
        # Yield header first
        yield ",".join(valid_columns) + "\n"
        
        # Fetch data in chunks
        submissions_collection = self.db["submissions"]
        teams_collection = self.db["teams"]
        
        cursor = submissions_collection.find({"eventId": event_id})
        
        chunk_data = []
        row_count = 0
        
        async for sub in cursor:
            # Enrich with team name
            team_id = sub.get("teamId", "")
            if team_id:
                try:
                    team = await teams_collection.find_one({"_id": ObjectId(team_id)})
                    if team:
                        sub["_team_name"] = team.get("teamName", "Unknown")
                except Exception:
                    sub["_team_name"] = "Unknown"
            else:
                sub["_team_name"] = "Unknown"
            
            sub["_extracted_score"] = self._extract_score(sub)
            chunk_data.append(sub)
            row_count += 1
            
            # Yield chunk when size reached
            if len(chunk_data) >= chunk_size:
                transformed = await self._transform_data(chunk_data, valid_columns)
                chunk_csv = self._generate_csv_rows(transformed, valid_columns)
                yield chunk_csv
                chunk_data = []
            
            # Safety limit
            if row_count >= MAX_EXPORT_ROWS:
                logger.warning(f"Export limit reached: {MAX_EXPORT_ROWS} rows")
                break
        
        # Yield remaining data
        if chunk_data:
            transformed = await self._transform_data(chunk_data, valid_columns)
            chunk_csv = self._generate_csv_rows(transformed, valid_columns)
            yield chunk_csv
        
        logger.info(f"Streaming export complete: {row_count} rows")
    
    def _generate_csv_rows(
        self,
        data: List[Dict],
        columns: List[str]
    ) -> str:
        """
        Generate CSV rows (without header).
        
        Args:
            data: List of row dictionaries.
            columns: Column order.
        
        Returns:
            CSV rows as string.
        """
        if not data:
            return ""
        
        df = pd.DataFrame(data)
        df = df.reindex(columns=columns)
        
        output = io.StringIO()
        df.to_csv(output, index=False, header=False, encoding='utf-8')
        
        return output.getvalue()
