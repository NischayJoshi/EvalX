"""
ML/AI Domain Evaluator
======================

Comprehensive evaluator for machine learning and AI projects including:
- Framework detection (TensorFlow, PyTorch, scikit-learn, Transformers)
- Model architecture analysis (CNN, RNN, Transformer, etc.)
- Training pipeline best practices
- MLOps patterns (experiment tracking, model versioning)
- Inference optimization patterns

Scoring Weights:
    - Architecture: 30% (model design, modularity)
    - Best Practices: 30% (training pipeline, reproducibility)
    - Innovation: 20% (novel approaches, SOTA techniques)
    - Security: 10% (model security, data privacy)
    - Completeness: 10% (documentation, tests, configs)

Author: EvalX Team
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import re

from evaluators.base import BaseDomainEvaluator
from evaluators.models import (
    DomainType,
    DomainScore,
    PatternMatch,
)
from evaluators.constants import ML_AI_PATTERNS
from evaluators.detector_utils import (
    calculate_weighted_score,
    normalize_score,
    count_pattern_categories,
)


logger = logging.getLogger(__name__)


class MLEvaluator(BaseDomainEvaluator):
    """
    Domain evaluator specialized for Machine Learning and AI projects.

    This evaluator analyzes ML/AI codebases for model architecture quality,
    training pipeline best practices, MLOps patterns, and adherence to
    industry standards for reproducible machine learning.

    Features:
        - Multi-framework support (TensorFlow, PyTorch, JAX, scikit-learn)
        - Model architecture detection (CNN, RNN, Transformer, etc.)
        - Training pipeline analysis (data loading, augmentation, scheduling)
        - MLOps pattern recognition (tracking, versioning, deployment)
        - Inference optimization detection (quantization, ONNX export)

    Example:
        >>> evaluator = MLEvaluator("/path/to/ml-project")
        >>> result = await evaluator.evaluate()
        >>> print(f"Architecture Score: {result.score.architecture_score}")
    """

    # Scoring weights for ML/AI domain
    SCORING_WEIGHTS = {
        "architecture": 0.30,
        "best_practices": 0.30,
        "innovation": 0.20,
        "security": 0.10,
        "completeness": 0.10,
    }

    # Framework indicators for detection
    FRAMEWORK_PATTERNS = [
        "tensorflow_usage",
        "pytorch_usage",
        "sklearn_usage",
        "huggingface_transformers",
    ]

    # Advanced ML patterns indicating sophisticated implementations
    ADVANCED_ML_PATTERNS = [
        "attention_mechanism",
        "transformer_architecture",
        "hyperparameter_tuning",
        "model_quantization",
        "experiment_tracking",
    ]

    # MLOps maturity indicators
    MLOPS_PATTERNS = [
        "experiment_tracking",
        "model_versioning",
        "model_checkpointing",
        "onnx_export",
    ]

    @property
    def domain_type(self) -> DomainType:
        """Return the ML/AI domain type."""
        return DomainType.ML_AI

    def get_file_extensions(self) -> List[str]:
        """
        Return file extensions relevant to ML/AI development.

        Includes:
            - .py: Python source files (primary ML language)
            - .ipynb: Jupyter notebooks (experimentation)
            - .yaml/.yml: Configuration files (configs, hyperparameters)
            - .json: Model configs, metrics
            - .onnx: Exported models
            - .pt/.pth: PyTorch checkpoints
            - .h5: Keras/TensorFlow models
        """
        return [
            ".py",  # Python source
            ".ipynb",  # Jupyter notebooks
            ".yaml",  # Configs
            ".yml",  # Configs
            ".json",  # Configs, metrics
            ".toml",  # Poetry/configs
        ]

    def get_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Return ML/AI-specific detection patterns."""
        return ML_AI_PATTERNS

    async def calculate_domain_score(
        self, patterns: List[PatternMatch], file_analysis: Dict[str, Any]
    ) -> DomainScore:
        """
        Calculate comprehensive ML/AI domain score.

        Scoring methodology:
            1. Architecture Score (30%): Model design quality, modularity,
               proper layer organization, framework usage
            2. Best Practices Score (30%): Training pipeline quality,
               data handling, reproducibility, logging
            3. Innovation Score (20%): Advanced techniques, SOTA models,
               novel approaches
            4. Security Score (10%): Model security, data privacy,
               input validation
            5. Completeness Score (10%): Documentation, tests, configs,
               deployment readiness

        Args:
            patterns: Detected patterns from codebase analysis
            file_analysis: File-level statistics and analysis

        Returns:
            DomainScore with detailed breakdown
        """
        logger.info(f"[{self._evaluation_id}] Calculating ML/AI domain score")

        # Categorize patterns
        category_counts = count_pattern_categories(patterns)

        # 1. Architecture Score
        architecture_score = await self._calculate_architecture_score(
            patterns, file_analysis
        )

        # 2. Best Practices Score
        best_practices_score = await self._calculate_best_practices_score(patterns)

        # 3. Innovation Score
        innovation_score = await self._calculate_innovation_score(patterns)

        # 4. Security Score
        security_score = await self._calculate_security_score(patterns)

        # 5. Completeness Score
        completeness_score = await self._calculate_completeness_score(
            patterns, file_analysis
        )

        # Calculate weighted overall score
        metrics = {
            "architecture": architecture_score,
            "best_practices": best_practices_score,
            "innovation": innovation_score,
            "security": security_score,
            "completeness": completeness_score,
        }

        overall_score = calculate_weighted_score(metrics, self.SCORING_WEIGHTS)

        # Detect primary framework
        pattern_names = {p.pattern_name for p in patterns}
        primary_framework = self._detect_primary_framework(pattern_names)

        # Build detailed breakdown
        breakdown = {
            "primary_framework": primary_framework,
            "framework_patterns_found": category_counts.get("framework", 0),
            "architecture_patterns_found": category_counts.get("architecture", 0),
            "best_practice_patterns_found": category_counts.get("best_practice", 0),
            "uses_pytorch": "pytorch_usage" in pattern_names,
            "uses_tensorflow": "tensorflow_usage" in pattern_names,
            "uses_transformers": "huggingface_transformers" in pattern_names,
            "has_experiment_tracking": "experiment_tracking" in pattern_names,
            "has_model_versioning": "model_versioning" in pattern_names,
            "mlops_maturity_score": self._calculate_mlops_maturity(pattern_names),
            "total_py_files": file_analysis.get("files_by_extension", {}).get(".py", 0),
            "has_notebooks": file_analysis.get("files_by_extension", {}).get(
                ".ipynb", 0
            )
            > 0,
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

    def _detect_primary_framework(self, pattern_names: set) -> str:
        """Detect the primary ML framework used."""
        if "pytorch_usage" in pattern_names:
            if "huggingface_transformers" in pattern_names:
                return "PyTorch + Transformers"
            return "PyTorch"
        if "tensorflow_usage" in pattern_names:
            return "TensorFlow/Keras"
        if "sklearn_usage" in pattern_names:
            return "scikit-learn"
        if "huggingface_transformers" in pattern_names:
            return "Transformers"
        return "Unknown"

    def _calculate_mlops_maturity(self, pattern_names: set) -> float:
        """Calculate MLOps maturity level (0-100)."""
        mlops_score = 0
        total_patterns = len(self.MLOPS_PATTERNS)

        for pattern in self.MLOPS_PATTERNS:
            if pattern in pattern_names:
                mlops_score += 25

        return min(100, mlops_score)

    async def _calculate_architecture_score(
        self, patterns: List[PatternMatch], file_analysis: Dict[str, Any]
    ) -> float:
        """
        Calculate model architecture quality score.

        Factors:
            - Framework usage (PyTorch, TensorFlow, etc.)
            - Model architecture patterns (CNN, Transformer, etc.)
            - Modular design indicators
            - Proper layer organization
        """
        base_score = 40.0

        arch_patterns = [p for p in patterns if p.category == "architecture"]
        framework_patterns = [p for p in patterns if p.category == "framework"]
        pattern_names = {p.pattern_name for p in patterns}

        # Framework bonus
        base_score += len(framework_patterns) * 8

        # Architecture patterns
        if "neural_network_definition" in pattern_names:
            base_score += 12

        if "transformer_architecture" in pattern_names:
            base_score += 15

        if "attention_mechanism" in pattern_names:
            base_score += 10

        if "cnn_architecture" in pattern_names:
            base_score += 10

        if "rnn_architecture" in pattern_names:
            base_score += 8

        # Inference patterns
        if "model_inference" in pattern_names:
            base_score += 8

        # File structure bonus (indicates modularity)
        py_files = file_analysis.get("files_by_extension", {}).get(".py", 0)
        if py_files >= 5:
            base_score += 8
        if py_files >= 10:
            base_score += 5

        return normalize_score(base_score)

    async def _calculate_best_practices_score(
        self, patterns: List[PatternMatch]
    ) -> float:
        """
        Calculate ML best practices adherence score.

        Factors:
            - Training loop quality
            - Data augmentation usage
            - Learning rate scheduling
            - Early stopping
            - Gradient clipping
            - Checkpointing
        """
        base_score = 40.0

        pattern_names = {p.pattern_name for p in patterns}
        bp_patterns = [p for p in patterns if p.category == "best_practice"]

        base_score += len(bp_patterns) * 5

        # Training pipeline patterns
        if "training_loop" in pattern_names:
            base_score += 12

        if "data_augmentation" in pattern_names:
            base_score += 10

        if "learning_rate_scheduler" in pattern_names:
            base_score += 10

        if "early_stopping" in pattern_names:
            base_score += 8

        if "gradient_clipping" in pattern_names:
            base_score += 8

        if "model_checkpointing" in pattern_names:
            base_score += 10

        # MLOps patterns
        if "experiment_tracking" in pattern_names:
            base_score += 12

        if "model_versioning" in pattern_names:
            base_score += 8

        if "hyperparameter_tuning" in pattern_names:
            base_score += 10

        return normalize_score(base_score)

    async def _calculate_innovation_score(self, patterns: List[PatternMatch]) -> float:
        """
        Calculate innovation score based on advanced ML techniques.

        Factors:
            - Transformer/Attention mechanisms
            - SOTA model architectures
            - Advanced training techniques
            - Model optimization methods
        """
        base_score = 35.0

        pattern_names = {p.pattern_name for p in patterns}

        # Advanced architectures
        if "transformer_architecture" in pattern_names:
            base_score += 20

        if "attention_mechanism" in pattern_names:
            base_score += 15

        if "huggingface_transformers" in pattern_names:
            base_score += 12

        # Optimization techniques
        if "model_quantization" in pattern_names:
            base_score += 12

        if "onnx_export" in pattern_names:
            base_score += 8

        # Advanced training
        if "hyperparameter_tuning" in pattern_names:
            base_score += 10

        # Multiple advanced patterns bonus
        advanced_count = sum(1 for p in self.ADVANCED_ML_PATTERNS if p in pattern_names)
        if advanced_count >= 3:
            base_score += 10

        return normalize_score(base_score)

    async def _calculate_security_score(self, patterns: List[PatternMatch]) -> float:
        """
        Calculate ML security score.

        For ML projects, security focuses on:
            - Model robustness
            - Data privacy considerations
            - Input validation for inference

        Note: ML security patterns are less common in traditional
        pattern matching, so baseline is higher.
        """
        base_score = 60.0  # Higher baseline for ML

        # Adjust based on presence of any validation patterns
        pattern_names = {p.pattern_name for p in patterns}

        if "model_inference" in pattern_names:
            base_score += 10  # Proper inference suggests input handling

        if "model_checkpointing" in pattern_names:
            base_score += 5  # Recovery capability

        return normalize_score(base_score)

    async def _calculate_completeness_score(
        self, patterns: List[PatternMatch], file_analysis: Dict[str, Any]
    ) -> float:
        """
        Calculate project completeness score.

        Factors:
            - Configuration files
            - Jupyter notebooks for experimentation
            - Multiple Python modules
            - Model export capabilities
        """
        base_score = 45.0

        files_by_ext = file_analysis.get("files_by_extension", {})
        pattern_names = {p.pattern_name for p in patterns}

        # Python files
        py_files = files_by_ext.get(".py", 0)
        if py_files >= 3:
            base_score += 10
        if py_files >= 8:
            base_score += 5

        # Notebooks
        ipynb_files = files_by_ext.get(".ipynb", 0)
        if ipynb_files >= 1:
            base_score += 10
        if ipynb_files >= 3:
            base_score += 5

        # Config files
        yaml_files = files_by_ext.get(".yaml", 0) + files_by_ext.get(".yml", 0)
        if yaml_files >= 1:
            base_score += 8

        # Export capability
        if "onnx_export" in pattern_names:
            base_score += 10

        # Total lines (substantial codebase)
        total_lines = file_analysis.get("total_lines", 0)
        if total_lines > 500:
            base_score += 5
        if total_lines > 2000:
            base_score += 5

        return normalize_score(base_score)

    async def _generate_recommendations(
        self, patterns: List[PatternMatch], score: DomainScore
    ) -> List[str]:
        """Generate ML/AI-specific recommendations."""
        recommendations = []
        pattern_names = {p.pattern_name for p in patterns}

        # Architecture recommendations
        if score.architecture_score < 70:
            recommendations.append(
                "📐 Improve model architecture with better modularity and "
                "clear separation of model, data, and training logic"
            )

        # Training pipeline recommendations
        if "training_loop" not in pattern_names:
            recommendations.append(
                "🔄 Implement a structured training loop with proper "
                "epoch handling, batch processing, and loss tracking"
            )

        if "learning_rate_scheduler" not in pattern_names:
            recommendations.append(
                "📉 Add learning rate scheduling (e.g., CosineAnnealing, "
                "ReduceLROnPlateau) for better convergence"
            )

        if "data_augmentation" not in pattern_names:
            recommendations.append(
                "🔀 Implement data augmentation to improve model "
                "generalization and reduce overfitting"
            )

        if "early_stopping" not in pattern_names:
            recommendations.append(
                "⏹️ Add early stopping to prevent overfitting and "
                "reduce unnecessary training time"
            )

        # MLOps recommendations
        if "experiment_tracking" not in pattern_names:
            recommendations.append(
                "📊 Integrate experiment tracking (MLflow, W&B, TensorBoard) "
                "for reproducibility and comparison"
            )

        if "model_checkpointing" not in pattern_names:
            recommendations.append(
                "💾 Implement model checkpointing to save best models "
                "and enable training recovery"
            )

        if "model_versioning" not in pattern_names:
            recommendations.append(
                "🏷️ Add model versioning (DVC, MLflow) for tracking "
                "model artifacts and experiments"
            )

        # Deployment recommendations
        if (
            "onnx_export" not in pattern_names
            and "model_quantization" not in pattern_names
        ):
            recommendations.append(
                "🚀 Consider ONNX export or quantization for efficient "
                "model deployment and inference"
            )

        # Innovation recommendations
        if score.innovation_score < 60:
            recommendations.append(
                "💡 Explore advanced techniques like attention mechanisms, "
                "transformers, or automated hyperparameter tuning"
            )

        return recommendations[:8]

    async def _identify_insights(
        self, patterns: List[PatternMatch], file_analysis: Dict[str, Any]
    ) -> tuple[List[str], List[str]]:
        """Identify ML/AI-specific strengths and weaknesses."""
        strengths = []
        weaknesses = []

        pattern_names = {p.pattern_name for p in patterns}
        bp_patterns = [p for p in patterns if p.category == "best_practice"]

        # Strengths
        if "pytorch_usage" in pattern_names or "tensorflow_usage" in pattern_names:
            strengths.append("Uses industry-standard deep learning framework")

        if "huggingface_transformers" in pattern_names:
            strengths.append("Leverages Hugging Face Transformers ecosystem")

        if "transformer_architecture" in pattern_names:
            strengths.append("Implements modern Transformer architecture")

        if len(bp_patterns) >= 5:
            strengths.append(
                "Strong ML engineering practices with multiple best patterns"
            )

        if "experiment_tracking" in pattern_names:
            strengths.append("Proper experiment tracking for reproducibility")

        if "hyperparameter_tuning" in pattern_names:
            strengths.append("Automated hyperparameter optimization implemented")

        # Weaknesses
        if "training_loop" not in pattern_names:
            weaknesses.append("No clear training loop structure detected")

        if "model_checkpointing" not in pattern_names:
            weaknesses.append("Missing model checkpointing for training recovery")

        if "experiment_tracking" not in pattern_names:
            weaknesses.append("No experiment tracking - reproducibility concerns")

        if len(bp_patterns) < 3:
            weaknesses.append("Limited ML best practices implemented")

        ipynb_files = file_analysis.get("files_by_extension", {}).get(".ipynb", 0)
        py_files = file_analysis.get("files_by_extension", {}).get(".py", 0)
        if ipynb_files > py_files:
            weaknesses.append(
                "Heavy reliance on notebooks - consider refactoring to modules"
            )

        return strengths, weaknesses
