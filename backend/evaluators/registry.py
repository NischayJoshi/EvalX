"""
Domain Evaluator Registry
=========================

Central registry for domain evaluator class mapping and instantiation.
Provides factory pattern for creating evaluator instances based on domain type.

Features:
    - Domain-to-evaluator class mapping
    - Lazy instantiation of evaluators
    - Support for custom evaluator registration
    - Evaluator capability introspection

Usage:
    from evaluators.registry import DomainRegistry

    registry = DomainRegistry()
    evaluator = registry.get_evaluator(DomainType.WEB3, repo_path="/path/to/repo")

    # Or get all available evaluators
    all_evaluators = registry.get_all_evaluators()

Author: EvalX Team
"""

from typing import Dict, Type, Optional, List, Any
import logging

from evaluators.base import BaseDomainEvaluator
from evaluators.models import DomainType
from evaluators.exceptions import UnsupportedDomainError

# Import all domain evaluators
from evaluators.domain.web3_evaluator import Web3Evaluator
from evaluators.domain.ml_evaluator import MLEvaluator
from evaluators.domain.fintech_evaluator import FintechEvaluator
from evaluators.domain.iot_evaluator import IoTEvaluator
from evaluators.domain.arvr_evaluator import ARVREvaluator


logger = logging.getLogger(__name__)


class DomainRegistry:
    """
    Registry for domain-specific evaluators.

    This class implements the Factory Pattern for creating domain evaluator
    instances. It maintains a mapping between DomainType enum values and
    their corresponding evaluator classes.

    Features:
        - Centralized evaluator management
        - Lazy instantiation to reduce memory footprint
        - Custom evaluator registration for extensibility
        - Domain capability introspection

    Attributes:
        _evaluators: Mapping of DomainType to evaluator classes
        _instances: Cache of instantiated evaluators (optional)

    Example:
        >>> registry = DomainRegistry()
        >>> web3_evaluator = registry.get_evaluator(DomainType.WEB3)
        >>> supported = registry.get_supported_domains()
    """

    # Default mapping of domains to evaluator classes
    DEFAULT_EVALUATORS: Dict[DomainType, Type[BaseDomainEvaluator]] = {
        DomainType.WEB3: Web3Evaluator,
        DomainType.ML_AI: MLEvaluator,
        DomainType.FINTECH: FintechEvaluator,
        DomainType.IOT: IoTEvaluator,
        DomainType.AR_VR: ARVREvaluator,
    }

    def __init__(self, use_cache: bool = False):
        """
        Initialize the domain registry.

        Args:
            use_cache: Whether to cache evaluator instances for reuse.
                       Default is False for thread safety.
        """
        self._evaluators: Dict[DomainType, Type[BaseDomainEvaluator]] = (
            self.DEFAULT_EVALUATORS.copy()
        )
        self._use_cache = use_cache
        self._cache: Dict[str, BaseDomainEvaluator] = {}

        logger.info(
            f"DomainRegistry initialized with {len(self._evaluators)} evaluators"
        )

    def get_evaluator(
        self,
        domain: DomainType,
        repo_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> BaseDomainEvaluator:
        """
        Get an evaluator instance for the specified domain.

        Args:
            domain: The DomainType to get an evaluator for
            repo_path: Optional repository path to initialize with
            config: Optional configuration dictionary

        Returns:
            Instantiated evaluator for the specified domain

        Raises:
            UnsupportedDomainError: If domain is not registered

        Example:
            >>> evaluator = registry.get_evaluator(
            ...     DomainType.WEB3,
            ...     repo_path="/path/to/defi-project"
            ... )
        """
        if domain not in self._evaluators:
            raise UnsupportedDomainError(
                domain.value,
                supported_domains=[d.value for d in self._evaluators.keys()],
            )

        # Check cache if enabled
        cache_key = f"{domain.value}:{repo_path}"
        if self._use_cache and cache_key in self._cache:
            logger.debug(f"Returning cached evaluator for {domain.value}")
            return self._cache[cache_key]

        # Instantiate new evaluator
        evaluator_class = self._evaluators[domain]
        evaluator = evaluator_class(repo_path=repo_path, config=config)

        # Cache if enabled
        if self._use_cache:
            self._cache[cache_key] = evaluator

        logger.info(f"Created {evaluator_class.__name__} for domain {domain.value}")
        return evaluator

    def register_evaluator(
        self,
        domain: DomainType,
        evaluator_class: Type[BaseDomainEvaluator],
        override: bool = False,
    ) -> None:
        """
        Register a custom evaluator for a domain.

        This allows extending the registry with custom evaluator implementations.

        Args:
            domain: The DomainType to register
            evaluator_class: The evaluator class to register
            override: Whether to override existing registration

        Raises:
            ValueError: If domain already registered and override is False
            TypeError: If evaluator_class doesn't inherit from BaseDomainEvaluator

        Example:
            >>> class CustomWeb3Evaluator(BaseDomainEvaluator):
            ...     pass
            >>> registry.register_evaluator(
            ...     DomainType.WEB3,
            ...     CustomWeb3Evaluator,
            ...     override=True
            ... )
        """
        # Validate evaluator class
        if not issubclass(evaluator_class, BaseDomainEvaluator):
            raise TypeError(
                f"{evaluator_class.__name__} must inherit from BaseDomainEvaluator"
            )

        # Check for existing registration
        if domain in self._evaluators and not override:
            raise ValueError(
                f"Domain {domain.value} already registered. "
                "Use override=True to replace."
            )

        self._evaluators[domain] = evaluator_class

        # Clear cache for this domain
        if self._use_cache:
            keys_to_remove = [
                k for k in self._cache.keys() if k.startswith(f"{domain.value}:")
            ]
            for key in keys_to_remove:
                del self._cache[key]

        logger.info(f"Registered {evaluator_class.__name__} for domain {domain.value}")

    def unregister_evaluator(self, domain: DomainType) -> bool:
        """
        Unregister an evaluator for a domain.

        Args:
            domain: The DomainType to unregister

        Returns:
            True if evaluator was removed, False if not found
        """
        if domain in self._evaluators:
            del self._evaluators[domain]
            logger.info(f"Unregistered evaluator for domain {domain.value}")
            return True
        return False

    def get_supported_domains(self) -> List[DomainType]:
        """
        Get list of all supported domain types.

        Returns:
            List of DomainType values that have registered evaluators
        """
        return list(self._evaluators.keys())

    def get_all_evaluators(
        self, repo_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None
    ) -> Dict[DomainType, BaseDomainEvaluator]:
        """
        Get instances of all registered evaluators.

        Args:
            repo_path: Optional repository path for all evaluators
            config: Optional shared configuration

        Returns:
            Dictionary mapping DomainType to evaluator instances
        """
        evaluators = {}
        for domain in self._evaluators.keys():
            evaluators[domain] = self.get_evaluator(
                domain, repo_path=repo_path, config=config
            )
        return evaluators

    def is_domain_supported(self, domain: DomainType) -> bool:
        """
        Check if a domain has a registered evaluator.

        Args:
            domain: The DomainType to check

        Returns:
            True if domain is supported, False otherwise
        """
        return domain in self._evaluators

    def get_evaluator_info(self, domain: DomainType) -> Dict[str, Any]:
        """
        Get information about a registered evaluator.

        Args:
            domain: The DomainType to get info for

        Returns:
            Dictionary containing evaluator metadata

        Raises:
            UnsupportedDomainError: If domain is not registered
        """
        if domain not in self._evaluators:
            raise UnsupportedDomainError(domain.value)

        evaluator_class = self._evaluators[domain]

        return {
            "domain": domain.value,
            "class_name": evaluator_class.__name__,
            "module": evaluator_class.__module__,
            "docstring": evaluator_class.__doc__,
            "file_extensions": evaluator_class(repo_path=None).get_file_extensions(),
        }

    def clear_cache(self) -> None:
        """Clear all cached evaluator instances."""
        self._cache.clear()
        logger.info("Cleared evaluator cache")

    def __len__(self) -> int:
        """Return number of registered evaluators."""
        return len(self._evaluators)

    def __contains__(self, domain: DomainType) -> bool:
        """Check if domain is registered."""
        return domain in self._evaluators

    def __repr__(self) -> str:
        domains = [d.value for d in self._evaluators.keys()]
        return f"DomainRegistry(domains={domains})"


# Global registry instance (singleton pattern)
_global_registry: Optional[DomainRegistry] = None


def get_registry() -> DomainRegistry:
    """
    Get the global domain registry instance.

    This provides a singleton-like access pattern for the registry,
    useful for applications that need a single shared registry.

    Returns:
        Global DomainRegistry instance

    Example:
        >>> from evaluators.registry import get_registry
        >>> registry = get_registry()
        >>> evaluator = registry.get_evaluator(DomainType.ML_AI)
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = DomainRegistry()
    return _global_registry
