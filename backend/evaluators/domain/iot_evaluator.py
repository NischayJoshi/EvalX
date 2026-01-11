"""
IoT Domain Evaluator
====================

Evaluator for Internet of Things and embedded systems projects including:
- Device communication protocols (MQTT, CoAP)
- Sensor and actuator integration
- Device management and provisioning
- Edge computing patterns
- Security for constrained devices

Scoring Weights:
    - Security: 30% (device auth, secure boot)
    - Architecture: 30% (protocols, edge patterns)
    - Best Practices: 20% (telemetry, OTA updates)
    - Innovation: 10% (edge AI, advanced protocols)
    - Completeness: 10% (documentation, configs)

Author: EvalX Team
"""

from typing import List, Dict, Any
import logging

from evaluators.base import BaseDomainEvaluator
from evaluators.models import DomainType, DomainScore, PatternMatch
from evaluators.constants import IOT_PATTERNS
from evaluators.detector_utils import normalize_score, count_pattern_categories


logger = logging.getLogger(__name__)


class IoTEvaluator(BaseDomainEvaluator):
    """
    Domain evaluator for Internet of Things projects.

    Analyzes IoT codebases for proper protocol usage, device management,
    security patterns, and embedded systems best practices.

    Features:
        - Protocol detection (MQTT, CoAP, WebSocket)
        - Device management pattern recognition
        - Sensor/actuator integration analysis
        - Security assessment for constrained devices
        - Edge computing pattern detection
    """

    SCORING_WEIGHTS = {
        "security": 0.30,
        "architecture": 0.30,
        "best_practices": 0.20,
        "innovation": 0.10,
        "completeness": 0.10,
    }

    @property
    def domain_type(self) -> DomainType:
        return DomainType.IOT

    def get_file_extensions(self) -> List[str]:
        return [
            ".py",
            ".c",
            ".cpp",
            ".h",
            ".ino",
            ".rs",
            ".go",
            ".js",
            ".ts",
            ".yaml",
            ".yml",
        ]

    def get_patterns(self) -> Dict[str, Dict[str, Any]]:
        return IOT_PATTERNS

    async def calculate_domain_score(
        self, patterns: List[PatternMatch], file_analysis: Dict[str, Any]
    ) -> DomainScore:
        """Calculate IoT domain score with focus on protocols and security."""
        logger.info(f"[{self._evaluation_id}] Calculating IoT domain score")

        pattern_names = {p.pattern_name for p in patterns}
        category_counts = count_pattern_categories(patterns)

        # Security Score
        security_score = 50.0
        if "device_authentication" in pattern_names:
            security_score += 20
        if "secure_boot" in pattern_names:
            security_score += 15
        security_score += category_counts.get("security", 0) * 5

        # Architecture Score
        architecture_score = 45.0
        protocol_patterns = ["mqtt_protocol", "coap_protocol", "websocket_iot"]
        for proto in protocol_patterns:
            if proto in pattern_names:
                architecture_score += 12

        if "device_provisioning" in pattern_names:
            architecture_score += 10
        if "device_telemetry" in pattern_names:
            architecture_score += 10
        if "edge_processing" in pattern_names:
            architecture_score += 10

        # Best Practices Score
        best_practices_score = 50.0
        if "ota_update" in pattern_names:
            best_practices_score += 15
        if "sensor_reading" in pattern_names:
            best_practices_score += 10
        if "actuator_control" in pattern_names:
            best_practices_score += 10

        # Innovation Score
        innovation_score = 40.0
        if "edge_processing" in pattern_names:
            innovation_score += 20
        if "coap_protocol" in pattern_names:
            innovation_score += 10

        # Completeness Score
        completeness_score = 50.0
        c_files = file_analysis.get("files_by_extension", {}).get(".c", 0)
        cpp_files = file_analysis.get("files_by_extension", {}).get(".cpp", 0)
        py_files = file_analysis.get("files_by_extension", {}).get(".py", 0)

        if c_files + cpp_files >= 3:
            completeness_score += 15
        if py_files >= 3:
            completeness_score += 10

        # Calculate overall
        overall_score = (
            security_score * 0.30
            + architecture_score * 0.30
            + best_practices_score * 0.20
            + innovation_score * 0.10
            + completeness_score * 0.10
        )

        breakdown = {
            "has_mqtt": "mqtt_protocol" in pattern_names,
            "has_coap": "coap_protocol" in pattern_names,
            "has_ota": "ota_update" in pattern_names,
            "has_edge_computing": "edge_processing" in pattern_names,
            "security_patterns_count": category_counts.get("security", 0),
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
        """Generate IoT-specific recommendations."""
        recommendations = []
        pattern_names = {p.pattern_name for p in patterns}

        if "device_authentication" not in pattern_names:
            recommendations.append(
                "🔐 Implement X.509 certificate-based device authentication"
            )

        if "secure_boot" not in pattern_names:
            recommendations.append(
                "🛡️ Add secure boot verification for firmware integrity"
            )

        if (
            "mqtt_protocol" not in pattern_names
            and "coap_protocol" not in pattern_names
        ):
            recommendations.append(
                "📡 Implement MQTT or CoAP for efficient IoT communication"
            )

        if "ota_update" not in pattern_names:
            recommendations.append(
                "📲 Add OTA update capability for remote firmware updates"
            )

        if "device_telemetry" not in pattern_names:
            recommendations.append(
                "📊 Implement device telemetry for monitoring and diagnostics"
            )

        if "edge_processing" not in pattern_names:
            recommendations.append(
                "💡 Consider edge computing for local data processing"
            )

        return recommendations[:6]

    async def _identify_insights(
        self, patterns: List[PatternMatch], file_analysis: Dict[str, Any]
    ) -> tuple[List[str], List[str]]:
        """Identify IoT strengths and weaknesses."""
        strengths = []
        weaknesses = []

        pattern_names = {p.pattern_name for p in patterns}

        if "mqtt_protocol" in pattern_names:
            strengths.append("MQTT protocol implementation for reliable messaging")

        if "device_authentication" in pattern_names:
            strengths.append("Certificate-based device authentication")

        if "edge_processing" in pattern_names:
            strengths.append("Edge computing for reduced latency")

        if "ota_update" in pattern_names:
            strengths.append("OTA update capability for maintenance")

        # Weaknesses
        if "device_authentication" not in pattern_names:
            weaknesses.append("Missing device authentication mechanisms")

        if "secure_boot" not in pattern_names:
            weaknesses.append("No secure boot implementation")

        return strengths, weaknesses
