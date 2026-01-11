"""
Domain Detection Utilities
==========================

Helper functions for domain detection, file analysis, and pattern matching.
These utilities support the domain evaluators with common operations.

Features:
    - Domain auto-detection from file contents and structure
    - File content analysis and statistics
    - Pattern matching utilities
    - Confidence calculation helpers

Author: EvalX Team
"""

from typing import Dict, List, Tuple, Optional, Set, Any
from pathlib import Path
import re
import logging
from collections import Counter

from evaluators.models import DomainType
from evaluators.constants import (
    ALL_DOMAIN_PATTERNS,
    DOMAIN_FILE_EXTENSIONS,
    DOMAIN_KEYWORDS,
)


logger = logging.getLogger(__name__)


def detect_domain_from_files(repo_path: str) -> Tuple[DomainType, float]:
    """
    Auto-detect the primary domain of a repository based on file analysis.

    Uses a multi-signal approach:
        1. File extensions present in the repository
        2. Pattern matches in file contents
        3. Keyword detection in README and documentation
        4. Dependency file analysis (package.json, requirements.txt, etc.)

    Args:
        repo_path: Absolute path to the repository root

    Returns:
        Tuple of (detected DomainType, confidence score)

    Example:
        >>> domain, confidence = detect_domain_from_files("/path/to/repo")
        >>> print(f"Detected: {domain.value} with {confidence:.0%} confidence")
    """
    repo = Path(repo_path)

    if not repo.exists():
        logger.warning(f"Repository path does not exist: {repo_path}")
        return DomainType.UNKNOWN, 0.0

    domain_scores: Dict[str, float] = {
        "web3": 0.0,
        "ml_ai": 0.0,
        "fintech": 0.0,
        "iot": 0.0,
        "ar_vr": 0.0,
    }

    # Signal 1: File extension analysis
    extension_scores = _analyze_file_extensions(repo)
    for domain, score in extension_scores.items():
        domain_scores[domain] += score * 0.25  # 25% weight

    # Signal 2: Dependency file analysis
    dependency_scores = _analyze_dependencies(repo)
    for domain, score in dependency_scores.items():
        domain_scores[domain] += score * 0.30  # 30% weight

    # Signal 3: README keyword analysis
    readme_scores = _analyze_readme(repo)
    for domain, score in readme_scores.items():
        domain_scores[domain] += score * 0.20  # 20% weight

    # Signal 4: Sample file content pattern matching
    content_scores = _analyze_file_contents(repo)
    for domain, score in content_scores.items():
        domain_scores[domain] += score * 0.25  # 25% weight

    # Find highest scoring domain
    if not any(domain_scores.values()):
        return DomainType.UNKNOWN, 0.0

    best_domain = max(domain_scores, key=domain_scores.get)
    best_score = domain_scores[best_domain]

    # Normalize confidence to 0-1 range
    confidence = min(1.0, best_score / 100.0)

    # Map string to DomainType
    domain_mapping = {
        "web3": DomainType.WEB3,
        "ml_ai": DomainType.ML_AI,
        "fintech": DomainType.FINTECH,
        "iot": DomainType.IOT,
        "ar_vr": DomainType.AR_VR,
    }

    detected_domain = domain_mapping.get(best_domain, DomainType.UNKNOWN)

    logger.info(
        f"Domain detection: {detected_domain.value} (confidence: {confidence:.2%})"
    )

    return detected_domain, confidence


def get_secondary_domains(
    repo_path: str, primary_domain: DomainType, threshold: float = 0.3
) -> List[DomainType]:
    """
    Identify secondary domains present in the repository.

    Some projects span multiple domains (e.g., ML + Fintech for fraud detection).
    This function identifies additional domains above a confidence threshold.

    Args:
        repo_path: Path to the repository
        primary_domain: The primary detected domain
        threshold: Minimum score ratio to primary for inclusion

    Returns:
        List of secondary DomainType values
    """
    repo = Path(repo_path)
    domain_scores = {}

    # Collect all signals
    extension_scores = _analyze_file_extensions(repo)
    dependency_scores = _analyze_dependencies(repo)
    readme_scores = _analyze_readme(repo)
    content_scores = _analyze_file_contents(repo)

    # Combine scores
    for domain in ["web3", "ml_ai", "fintech", "iot", "ar_vr"]:
        domain_scores[domain] = (
            extension_scores.get(domain, 0)
            + dependency_scores.get(domain, 0)
            + readme_scores.get(domain, 0)
            + content_scores.get(domain, 0)
        )

    # Get primary score
    primary_key = primary_domain.value.lower()
    if primary_key == "ml_ai":
        primary_key = "ml_ai"
    primary_score = domain_scores.get(primary_key, 1)

    # Find secondary domains above threshold
    secondary = []
    domain_mapping = {
        "web3": DomainType.WEB3,
        "ml_ai": DomainType.ML_AI,
        "fintech": DomainType.FINTECH,
        "iot": DomainType.IOT,
        "ar_vr": DomainType.AR_VR,
    }

    for domain, score in domain_scores.items():
        if domain == primary_key:
            continue
        if primary_score > 0 and (score / primary_score) >= threshold:
            secondary.append(domain_mapping[domain])

    return secondary


def _analyze_file_extensions(repo: Path) -> Dict[str, float]:
    """Analyze file extensions to score domains."""
    scores = {d: 0.0 for d in ["web3", "ml_ai", "fintech", "iot", "ar_vr"]}

    # Skip common non-code directories
    skip_dirs = {"node_modules", ".git", "venv", "__pycache__", "dist", "build"}

    extension_counts: Counter = Counter()

    for file_path in repo.rglob("*"):
        if file_path.is_dir():
            continue
        if any(skip in file_path.parts for skip in skip_dirs):
            continue
        extension_counts[file_path.suffix.lower()] += 1

    # Score based on domain-specific extensions
    for domain, extensions in DOMAIN_FILE_EXTENSIONS.items():
        for ext in extensions:
            count = extension_counts.get(ext, 0)
            # Weighted by uniqueness of extension
            if ext in [".sol", ".vy", ".move"]:  # Web3 unique
                scores[domain] += count * 10
            elif ext in [".ipynb"]:  # ML unique
                scores[domain] += count * 8
            elif ext in [".ino", ".h"]:  # IoT unique
                scores[domain] += count * 8
            elif ext in [".shader", ".hlsl", ".glsl"]:  # AR/VR unique
                scores[domain] += count * 8
            else:
                scores[domain] += count * 1

    return scores


def _analyze_dependencies(repo: Path) -> Dict[str, float]:
    """Analyze dependency files to score domains."""
    scores = {d: 0.0 for d in ["web3", "ml_ai", "fintech", "iot", "ar_vr"]}

    # Python dependencies (requirements.txt, Pipfile)
    python_deps = _read_python_dependencies(repo)

    # Web3 Python packages
    web3_packages = {"web3", "eth-brownie", "vyper", "solcx", "eth-abi"}
    # ML Python packages
    ml_packages = {
        "tensorflow",
        "torch",
        "pytorch",
        "keras",
        "sklearn",
        "scikit-learn",
        "transformers",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "seaborn",
        "mlflow",
        "wandb",
        "optuna",
    }
    # Fintech Python packages
    fintech_packages = {"stripe", "plaid", "yfinance", "ccxt", "alpaca-trade-api"}
    # IoT Python packages
    iot_packages = {"paho-mqtt", "adafruit", "RPi", "gpiozero", "spidev"}

    for dep in python_deps:
        dep_lower = dep.lower()
        if any(pkg in dep_lower for pkg in web3_packages):
            scores["web3"] += 15
        if any(pkg in dep_lower for pkg in ml_packages):
            scores["ml_ai"] += 15
        if any(pkg in dep_lower for pkg in fintech_packages):
            scores["fintech"] += 15
        if any(pkg in dep_lower for pkg in iot_packages):
            scores["iot"] += 15

    # JavaScript dependencies (package.json)
    js_deps = _read_js_dependencies(repo)

    # Web3 JS packages
    web3_js = {"ethers", "web3", "hardhat", "truffle", "@openzeppelin"}
    # ML JS packages
    ml_js = {"tensorflow", "@tensorflow", "brain.js", "ml5", "onnxruntime"}
    # Fintech JS packages
    fintech_js = {"stripe", "plaid", "braintree", "square"}
    # AR/VR JS packages
    arvr_js = {"three", "aframe", "babylon", "react-three-fiber", "webxr"}

    for dep in js_deps:
        dep_lower = dep.lower()
        if any(pkg in dep_lower for pkg in web3_js):
            scores["web3"] += 15
        if any(pkg in dep_lower for pkg in ml_js):
            scores["ml_ai"] += 15
        if any(pkg in dep_lower for pkg in fintech_js):
            scores["fintech"] += 15
        if any(pkg in dep_lower for pkg in arvr_js):
            scores["ar_vr"] += 15

    # Check for Unity/Unreal project files
    if (repo / "Assets").exists() or any(repo.glob("*.unity")):
        scores["ar_vr"] += 30
    if any(repo.glob("*.uproject")):
        scores["ar_vr"] += 30

    # Check for Hardhat/Truffle config
    if (repo / "hardhat.config.js").exists() or (repo / "truffle-config.js").exists():
        scores["web3"] += 25

    return scores


def _analyze_readme(repo: Path) -> Dict[str, float]:
    """Analyze README for domain keywords."""
    scores = {d: 0.0 for d in ["web3", "ml_ai", "fintech", "iot", "ar_vr"]}

    readme_files = list(repo.glob("README*")) + list(repo.glob("readme*"))

    content = ""
    for readme in readme_files:
        try:
            content += readme.read_text(encoding="utf-8", errors="ignore").lower()
        except Exception:
            continue

    if not content:
        return scores

    for domain, keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            count = content.count(keyword.lower())
            scores[domain] += count * 5

    return scores


def _analyze_file_contents(repo: Path, sample_size: int = 20) -> Dict[str, float]:
    """Sample file contents for pattern matching."""
    scores = {d: 0.0 for d in ["web3", "ml_ai", "fintech", "iot", "ar_vr"]}

    # Collect code files
    code_extensions = {".py", ".js", ".ts", ".sol", ".cs", ".cpp", ".java", ".go"}
    skip_dirs = {"node_modules", ".git", "venv", "__pycache__"}

    code_files = []
    for file_path in repo.rglob("*"):
        if file_path.is_dir():
            continue
        if any(skip in file_path.parts for skip in skip_dirs):
            continue
        if file_path.suffix.lower() in code_extensions:
            code_files.append(file_path)
        if len(code_files) >= sample_size * 5:
            break

    # Sample files
    import random

    sampled = random.sample(code_files, min(sample_size, len(code_files)))

    for file_path in sampled:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            # Check patterns for each domain
            for domain, patterns in ALL_DOMAIN_PATTERNS.items():
                for pattern_name, pattern_config in patterns.items():
                    regex = pattern_config.get("regex", "")
                    if regex and re.search(regex, content, re.IGNORECASE):
                        weight = pattern_config.get("weight", 1.0)
                        scores[domain] += weight * 3

        except Exception:
            continue

    return scores


def _read_python_dependencies(repo: Path) -> Set[str]:
    """Read Python dependencies from various files."""
    deps: Set[str] = set()

    # requirements.txt
    req_file = repo / "requirements.txt"
    if req_file.exists():
        try:
            content = req_file.read_text(encoding="utf-8", errors="ignore")
            for line in content.split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    # Extract package name (before ==, >=, etc.)
                    pkg = re.split(r"[=<>!]", line)[0].strip()
                    deps.add(pkg)
        except Exception:
            pass

    # Pipfile
    pipfile = repo / "Pipfile"
    if pipfile.exists():
        try:
            content = pipfile.read_text(encoding="utf-8", errors="ignore")
            # Simple extraction of package names
            for match in re.finditer(r"^(\w[\w-]*)\s*=", content, re.MULTILINE):
                deps.add(match.group(1))
        except Exception:
            pass

    # pyproject.toml
    pyproject = repo / "pyproject.toml"
    if pyproject.exists():
        try:
            content = pyproject.read_text(encoding="utf-8", errors="ignore")
            # Simple extraction
            for match in re.finditer(r'"([\w-]+)"', content):
                deps.add(match.group(1))
        except Exception:
            pass

    return deps


def _read_js_dependencies(repo: Path) -> Set[str]:
    """Read JavaScript dependencies from package.json."""
    deps: Set[str] = set()

    package_json = repo / "package.json"
    if package_json.exists():
        try:
            import json

            content = json.loads(package_json.read_text(encoding="utf-8"))

            for dep_type in ["dependencies", "devDependencies", "peerDependencies"]:
                if dep_type in content and isinstance(content[dep_type], dict):
                    deps.update(content[dep_type].keys())
        except Exception:
            pass

    return deps


def calculate_weighted_score(
    metrics: Dict[str, float], weights: Dict[str, float]
) -> float:
    """
    Calculate a weighted average score from multiple metrics.

    Args:
        metrics: Dictionary of metric names to scores (0-100)
        weights: Dictionary of metric names to weights (should sum to 1.0)

    Returns:
        Weighted average score (0-100)

    Example:
        >>> metrics = {"security": 80, "architecture": 70, "innovation": 90}
        >>> weights = {"security": 0.4, "architecture": 0.3, "innovation": 0.3}
        >>> score = calculate_weighted_score(metrics, weights)
        >>> print(f"Weighted score: {score:.1f}")
    """
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0

    weighted_sum = 0.0
    for metric, value in metrics.items():
        weight = weights.get(metric, 0)
        weighted_sum += value * weight

    return (
        weighted_sum / total_weight * (total_weight)
    )  # Normalize if weights don't sum to 1


def extract_code_context(content: str, line_number: int, context_lines: int = 3) -> str:
    """
    Extract code context around a specific line.

    Args:
        content: Full file content
        line_number: Target line number (1-indexed)
        context_lines: Number of lines before/after to include

    Returns:
        Code snippet with context
    """
    lines = content.split("\n")
    start = max(0, line_number - context_lines - 1)
    end = min(len(lines), line_number + context_lines)

    context = "\n".join(lines[start:end])
    return context[:500]  # Limit length


def count_pattern_categories(patterns: List[Any]) -> Dict[str, int]:
    """
    Count patterns by category for analysis.

    Args:
        patterns: List of PatternMatch objects

    Returns:
        Dictionary mapping categories to counts
    """
    counts: Dict[str, int] = {}

    for pattern in patterns:
        category = getattr(pattern, "category", "unknown")
        counts[category] = counts.get(category, 0) + 1

    return counts


def normalize_score(score: float, min_val: float = 0, max_val: float = 100) -> float:
    """Normalize a score to 0-100 range."""
    return max(min_val, min(max_val, score))
