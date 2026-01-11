"""
Domain Evaluators Subpackage
============================

Individual domain-specific evaluator implementations.

Each evaluator extends BaseDomainEvaluator and provides specialized
pattern detection and scoring for its respective technology domain.

Evaluators:
    - Web3Evaluator: Blockchain, smart contracts, DeFi
    - MLEvaluator: Machine learning, deep learning, MLOps
    - FintechEvaluator: Payments, banking, compliance
    - IoTEvaluator: Embedded systems, sensors, device management
    - ARVREvaluator: Augmented/Virtual reality, 3D, spatial computing

Author: EvalX Team
"""

from evaluators.domain.web3_evaluator import Web3Evaluator
from evaluators.domain.ml_evaluator import MLEvaluator
from evaluators.domain.fintech_evaluator import FintechEvaluator
from evaluators.domain.iot_evaluator import IoTEvaluator
from evaluators.domain.arvr_evaluator import ARVREvaluator

__all__ = [
    "Web3Evaluator",
    "MLEvaluator",
    "FintechEvaluator",
    "IoTEvaluator",
    "ARVREvaluator",
]
