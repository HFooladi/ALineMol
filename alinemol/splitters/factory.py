"""
Factory function and registry for molecular splitters.

This module provides a centralized way to create splitter instances by name,
list available splitters, and register new splitters.
"""

from typing import Dict, Type, Optional, Any, List

from alinemol.splitters.base import BaseMolecularSplitter


# Registry mapping splitter names to classes
_SPLITTER_REGISTRY: Dict[str, Type[BaseMolecularSplitter]] = {}

# Alias mapping for alternative names
_SPLITTER_ALIASES: Dict[str, str] = {
    # Molecular weight aliases
    "mw": "molecular_weight",
    "molecular-weight": "molecular_weight",
    "molecularweight": "molecular_weight",
    "mw_reverse": "molecular_weight_reverse",
    # LogP aliases
    "logp": "molecular_logp",
    "molecular-logp": "molecular_logp",
    "molecularlogp": "molecular_logp",
    # KMeans aliases
    "k-means": "kmeans",
    "k_means": "kmeans",
    # MaxDissimilarity aliases
    "maxdissimilarity": "max_dissimilarity",
    "max-dissimilarity": "max_dissimilarity",
    "maxdiss": "max_dissimilarity",
    # Butina aliases
    "taylor_butina": "butina",
    "taylor-butina": "butina",
    # Hi/Lo aliases
    "hisplit": "hi",
    "hi_split": "hi",
    "hi-split": "hi",
    "losplit": "lo",
    "lo_split": "lo",
    "lo-split": "lo",
    # UMAP aliases
    "umap_cluster": "umap",
    "umap-cluster": "umap",
    # DataSAIL aliases
    "data_sail": "datasail",
    "data-sail": "datasail",
    # Scaffold aliases
    "scaffold_generic": "scaffold_generic",
    "scaffold-generic": "scaffold_generic",
    "generic_scaffold": "scaffold_generic",
}


def register_splitter(
    name: str,
    aliases: Optional[List[str]] = None,
):
    """
    Decorator to register a splitter class in the registry.

    Use this decorator on splitter classes to make them available via
    the get_splitter() factory function.

    Args:
        name: Primary name for the splitter (lowercase, underscore-separated).
        aliases: Optional list of alternative names for the splitter.

    Returns:
        Decorator function that registers the class.

    Example:
        >>> @register_splitter("my_splitter", aliases=["my-splitter", "mysplit"])
        ... class MySplitter(BaseMolecularSplitter):
        ...     def _iter_indices(self, X, y=None, groups=None):
        ...         yield train_idx, test_idx
        ...
        >>> splitter = get_splitter("my_splitter")
        >>> splitter = get_splitter("mysplit")  # Also works via alias
    """

    def decorator(cls: Type[BaseMolecularSplitter]) -> Type[BaseMolecularSplitter]:
        _SPLITTER_REGISTRY[name] = cls
        if aliases:
            for alias in aliases:
                _SPLITTER_ALIASES[alias.lower()] = name
        return cls

    return decorator


def get_splitter(
    name: str,
    **kwargs: Any,
) -> BaseMolecularSplitter:
    """
    Factory function to create a splitter instance by name.

    Args:
        name: Name of the splitter (case-insensitive). Supports aliases.
            Available splitters can be listed with list_splitters().
        **kwargs: Configuration parameters passed to splitter constructor.
            Common parameters include:
            - n_splits: Number of splits (default varies by splitter)
            - test_size: Proportion or count for test set
            - random_state: Random seed for reproducibility

    Returns:
        Configured splitter instance.

    Raises:
        ValueError: If splitter name is not recognized.

    Example:
        >>> # Create a scaffold splitter
        >>> splitter = get_splitter("scaffold", make_generic=True, n_splits=5)
        >>> for train_idx, test_idx in splitter.split(smiles_list):
        ...     pass
        ...
        >>> # Create a KMeans splitter
        >>> splitter = get_splitter("kmeans", n_clusters=10, test_size=0.2)
        ...
        >>> # Using aliases
        >>> splitter = get_splitter("mw", generalize_to_larger=True)  # molecular_weight
    """
    # Normalize name
    normalized_name = name.lower().strip().replace("-", "_")

    # Resolve alias
    if normalized_name in _SPLITTER_ALIASES:
        normalized_name = _SPLITTER_ALIASES[normalized_name]

    # Get splitter class
    if normalized_name not in _SPLITTER_REGISTRY:
        available = sorted(_SPLITTER_REGISTRY.keys())
        raise ValueError(f"Unknown splitter: '{name}'. Available splitters: {available}")

    splitter_cls = _SPLITTER_REGISTRY[normalized_name]

    return splitter_cls(**kwargs)


def list_splitters() -> Dict[str, Type[BaseMolecularSplitter]]:
    """
    Return dictionary of all registered splitters.

    Returns:
        Dictionary mapping splitter names to their classes.

    Example:
        >>> splitters = list_splitters()
        >>> print(splitters.keys())
        dict_keys(['scaffold', 'kmeans', 'molecular_weight', ...])
    """
    return dict(_SPLITTER_REGISTRY)


def get_splitter_names() -> List[str]:
    """
    Return list of all registered splitter names.

    Returns:
        Sorted list of splitter names.

    Example:
        >>> names = get_splitter_names()
        >>> print(names)
        ['butina', 'datasail', 'hi', 'kmeans', ...]
    """
    return sorted(_SPLITTER_REGISTRY.keys())


def get_splitter_aliases() -> Dict[str, str]:
    """
    Return dictionary mapping aliases to canonical splitter names.

    Returns:
        Dictionary mapping alias strings to canonical names.

    Example:
        >>> aliases = get_splitter_aliases()
        >>> print(aliases["mw"])
        'molecular_weight'
    """
    return dict(_SPLITTER_ALIASES)


def is_splitter_registered(name: str) -> bool:
    """
    Check if a splitter name (or alias) is registered.

    Args:
        name: Splitter name or alias to check.

    Returns:
        True if the splitter is registered, False otherwise.

    Example:
        >>> is_splitter_registered("scaffold")
        True
        >>> is_splitter_registered("unknown_splitter")
        False
    """
    normalized_name = name.lower().strip().replace("-", "_")
    if normalized_name in _SPLITTER_REGISTRY:
        return True
    if normalized_name in _SPLITTER_ALIASES:
        return _SPLITTER_ALIASES[normalized_name] in _SPLITTER_REGISTRY
    return False
