"""
AR/VR Domain Evaluator
======================

Evaluator for Augmented Reality and Virtual Reality projects including:
- Game engine detection (Unity, Unreal)
- WebXR and web-based XR
- 3D rendering patterns
- Input tracking (hand, head, eye)
- Performance optimization patterns

Scoring Weights:
    - Architecture: 35% (3D, spatial patterns)
    - Best Practices: 25% (performance, LOD)
    - Innovation: 20% (advanced tracking, spatial)
    - Security: 10% (input validation)
    - Completeness: 10% (assets, configs)

Author: EvalX Team
"""

from typing import List, Dict, Any
import logging

from evaluators.base import BaseDomainEvaluator
from evaluators.models import DomainType, DomainScore, PatternMatch
from evaluators.constants import AR_VR_PATTERNS
from evaluators.detector_utils import normalize_score, count_pattern_categories


logger = logging.getLogger(__name__)


class ARVREvaluator(BaseDomainEvaluator):
    """
    Domain evaluator for Augmented/Virtual Reality projects.

    Analyzes AR/VR codebases for proper engine usage, rendering patterns,
    tracking implementations, and performance optimization.

    Features:
        - Engine detection (Unity, Unreal, WebXR)
        - 3D rendering pattern analysis
        - Input tracking detection (hand, head, eye)
        - Performance optimization patterns
        - Spatial computing features
    """

    SCORING_WEIGHTS = {
        "architecture": 0.35,
        "best_practices": 0.25,
        "innovation": 0.20,
        "security": 0.10,
        "completeness": 0.10,
    }

    @property
    def domain_type(self) -> DomainType:
        return DomainType.AR_VR

    def get_file_extensions(self) -> List[str]:
        return [
            ".cs",
            ".cpp",
            ".js",
            ".ts",
            ".shader",
            ".hlsl",
            ".glsl",
            ".json",
            ".yaml",
        ]

    def get_patterns(self) -> Dict[str, Dict[str, Any]]:
        return AR_VR_PATTERNS

    async def calculate_domain_score(
        self, patterns: List[PatternMatch], file_analysis: Dict[str, Any]
    ) -> DomainScore:
        """Calculate AR/VR domain score with focus on immersive experience quality."""
        logger.info(f"[{self._evaluation_id}] Calculating AR/VR domain score")

        pattern_names = {p.pattern_name for p in patterns}
        category_counts = count_pattern_categories(patterns)

        # Architecture Score (3D rendering, spatial)
        architecture_score = 45.0

        # Engine detection bonus
        if "unity_engine" in pattern_names:
            architecture_score += 15
        if "unreal_engine" in pattern_names:
            architecture_score += 15
        if "webxr" in pattern_names:
            architecture_score += 12

        if "3d_rendering" in pattern_names:
            architecture_score += 10
        if "spatial_mapping" in pattern_names:
            architecture_score += 12

        # Best Practices Score (performance)
        best_practices_score = 50.0
        if "frame_optimization" in pattern_names:
            best_practices_score += 18
        if "lod_system" in pattern_names:
            best_practices_score += 15
        if "object_occlusion" in pattern_names:
            best_practices_score += 10

        # Innovation Score (advanced tracking)
        innovation_score = 40.0
        tracking_patterns = ["hand_tracking", "head_tracking", "eye_tracking"]
        for tracking in tracking_patterns:
            if tracking in pattern_names:
                innovation_score += 12

        if "spatial_mapping" in pattern_names:
            innovation_score += 10
        if "image_tracking" in pattern_names:
            innovation_score += 8

        # Security Score (baseline for XR)
        security_score = 60.0  # Higher baseline for XR apps

        # Completeness Score
        completeness_score = 50.0
        cs_files = file_analysis.get("files_by_extension", {}).get(".cs", 0)
        shader_files = (
            file_analysis.get("files_by_extension", {}).get(".shader", 0)
            + file_analysis.get("files_by_extension", {}).get(".hlsl", 0)
            + file_analysis.get("files_by_extension", {}).get(".glsl", 0)
        )

        if cs_files >= 5:
            completeness_score += 15
        if shader_files >= 1:
            completeness_score += 12

        # Calculate overall
        overall_score = (
            architecture_score * 0.35
            + best_practices_score * 0.25
            + innovation_score * 0.20
            + security_score * 0.10
            + completeness_score * 0.10
        )

        # Detect primary engine
        engine = "Unknown"
        if "unity_engine" in pattern_names:
            engine = "Unity"
        elif "unreal_engine" in pattern_names:
            engine = "Unreal Engine"
        elif "webxr" in pattern_names:
            engine = "WebXR"

        breakdown = {
            "primary_engine": engine,
            "has_hand_tracking": "hand_tracking" in pattern_names,
            "has_eye_tracking": "eye_tracking" in pattern_names,
            "has_spatial_mapping": "spatial_mapping" in pattern_names,
            "has_performance_optimization": "frame_optimization" in pattern_names,
            "framework_patterns_count": category_counts.get("framework", 0),
        }

        return DomainScore(
            domain=self.domain_type,
            overall_score=normalize_score(overall_score),
            architecture_score=normalize_score(architecture_score),
            security_score=normalize_score(security_score),
            best_practices_score=normalize_score(best_practices_score),
            innovation_score=normalize_score(innovation_score),
            completeness_score=normalize_score(completeness_score),
            breakdown=breakdown,
        )

    async def _generate_recommendations(
        self, patterns: List[PatternMatch], score: DomainScore
    ) -> List[str]:
        """Generate AR/VR-specific recommendations."""
        recommendations = []
        pattern_names = {p.pattern_name for p in patterns}

        if "frame_optimization" not in pattern_names:
            recommendations.append(
                "🎮 Implement frame rate optimization to maintain 90+ FPS "
                "and prevent motion sickness"
            )

        if "lod_system" not in pattern_names:
            recommendations.append(
                "📐 Add Level of Detail (LOD) system for performance scaling"
            )

        if "hand_tracking" not in pattern_names:
            recommendations.append(
                "🖐️ Consider implementing hand tracking for natural interaction"
            )

        if "spatial_mapping" not in pattern_names:
            recommendations.append(
                "🗺️ Add spatial mapping for environment-aware experiences"
            )

        if "object_occlusion" not in pattern_names:
            recommendations.append(
                "👁️ Implement occlusion handling for realistic AR compositing"
            )

        if score.best_practices_score < 70:
            recommendations.append(
                "⚡ Focus on performance optimization - VR requires consistent high FPS"
            )

        return recommendations[:6]

    async def _identify_insights(
        self, patterns: List[PatternMatch], file_analysis: Dict[str, Any]
    ) -> tuple[List[str], List[str]]:
        """Identify AR/VR strengths and weaknesses."""
        strengths = []
        weaknesses = []

        pattern_names = {p.pattern_name for p in patterns}

        if "unity_engine" in pattern_names or "unreal_engine" in pattern_names:
            strengths.append("Uses industry-standard game engine")

        if "webxr" in pattern_names:
            strengths.append("Web-based XR for broad accessibility")

        if "hand_tracking" in pattern_names:
            strengths.append("Natural hand tracking interaction")

        if "frame_optimization" in pattern_names:
            strengths.append("Performance optimization for smooth VR experience")

        if "spatial_mapping" in pattern_names:
            strengths.append("Environment understanding for immersive AR")

        # Weaknesses
        if "frame_optimization" not in pattern_names:
            weaknesses.append("Missing frame rate optimization - motion sickness risk")

        if "lod_system" not in pattern_names:
            weaknesses.append("No LOD system - potential performance issues")

        return strengths, weaknesses
