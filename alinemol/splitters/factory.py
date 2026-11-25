"""
Factory function and registry for molecular splitters.

This module provides a centralized way to create splitter instances by name,
list available splitters, and register new splitters.
"""

import inspect
from typing import Dict, Type, Optional, Any, List

from alinemol.splitters.base import BaseMolecularSplitter


# Registry mapping splitter names to classes
_SPLITTER_REGISTRY: Dict[str, Type[BaseMolecularSplitter]] = {}

# Descriptive metadata per registered splitter (see ``register_splitter``)
_SPLITTER_INFO: Dict[str, Dict[str, Any]] = {}

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


def _init_parameters(cls: type) -> Dict[str, inspect.Parameter]:
    """Named ``__init__`` parameters of a splitter class (without self / *args / **kwargs)."""
    return {
        name: param
        for name, param in inspect.signature(cls.__init__).parameters.items()
        if name != "self" and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
    }


def _accepts_var_kwargs(cls: type) -> bool:
    return any(p.kind is p.VAR_KEYWORD for p in inspect.signature(cls.__init__).parameters.values())


def splitter_params(splitter: Any) -> Dict[str, Any]:
    """Constructor parameters of a splitter instance with their current values.

    Works for every registered splitter regardless of its base class (values are read
    from the attribute of the same name, or the ``_``-prefixed one). Use it to record
    exactly how a split was produced.

    Example:
        >>> splitter_params(get_splitter("kmeans", n_clusters=7))["n_clusters"]
        7
    """
    missing = object()
    params: Dict[str, Any] = {}
    for name in _init_parameters(type(splitter)):
        value = getattr(splitter, name, missing)
        if value is missing:
            value = getattr(splitter, f"_{name}", missing)
        if value is not missing:
            params[name] = value
    return params


def _splitter_repr(self: Any) -> str:
    params = ", ".join(f"{name}={value!r}" for name, value in splitter_params(self).items() if name != "smiles")
    return f"{type(self).__name__}({params})"


def register_splitter(
    name: str,
    aliases: Optional[List[str]] = None,
    *,
    family: str = "other",
    test_size: str = "approximate",
    needs_labels: bool = False,
    deterministic: bool = False,
    requires: Optional[str] = None,
):
    """
    Decorator to register a splitter class in the registry.

    Use this decorator on splitter classes to make them available via
    the get_splitter() factory function.

    Args:
        name: Primary name for the splitter (lowercase, underscore-separated).
        aliases: Optional list of alternative names for the splitter.
        family: What the split is based on: ``"random"``, ``"scaffold"``, ``"property"``,
            ``"clustering"``, ``"similarity"`` or ``"other"``.
        test_size: How the requested ``test_size`` is treated: ``"exact"``,
            ``"approximate"`` (whole groups are assigned, so the realised size can differ)
            or ``"ignored"`` (the method decides the sizes itself).
        needs_labels: Whether ``split()`` requires activity values as ``y``.
        deterministic: Whether every split is identical (``n_splits`` and
            ``random_state`` then have no effect).
        requires: Name of an optional package the splitter needs, if any.

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
        _SPLITTER_INFO[name] = {
            "family": family,
            "test_size": test_size,
            "needs_labels": needs_labels,
            "deterministic": deterministic,
            "requires": requires,
        }
        if aliases:
            for alias in aliases:
                _SPLITTER_ALIASES[alias.lower()] = name
        cls.__repr__ = _splitter_repr  # type: ignore[method-assign]
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

    if not _accepts_var_kwargs(splitter_cls):
        accepted = _init_parameters(splitter_cls)
        unknown = sorted(set(kwargs) - set(accepted))
        if unknown:
            raise TypeError(
                f"Splitter '{normalized_name}' does not accept {unknown}. Accepted parameters: {sorted(accepted)}"
            )

    return splitter_cls(**kwargs)


def resolve_splitter_name(name: str) -> str:
    """Canonical registry name for a splitter name or alias (raises ``ValueError`` if unknown)."""
    normalized_name = name.lower().strip().replace("-", "_")
    normalized_name = _SPLITTER_ALIASES.get(normalized_name, normalized_name)
    if normalized_name not in _SPLITTER_REGISTRY:
        raise ValueError(f"Unknown splitter: '{name}'. Available splitters: {sorted(_SPLITTER_REGISTRY)}")
    return normalized_name


def describe_splitters() -> List[Dict[str, Any]]:
    """Machine-readable description of every registered splitter.

    One dict per splitter, sorted by name, with JSON-serialisable values:

    - ``name``, ``class``, ``aliases``, ``summary`` (first docstring line)
    - ``family``: random / scaffold / property / clustering / similarity / other
    - ``test_size``: ``exact`` / ``approximate`` / ``ignored``
    - ``needs_labels``: ``split()`` needs activity values as ``y``
    - ``deterministic``: every split is identical
    - ``requires``: optional package needed, or ``None``
    - ``parameters``: constructor parameters and their defaults

    This is the single source for listings in the CLI, scripts and docs.
    """
    described = []
    for name in sorted(_SPLITTER_REGISTRY):
        cls = _SPLITTER_REGISTRY[name]
        doc = inspect.getdoc(cls) or ""
        parameters = {}
        for param_name, param in _init_parameters(cls).items():
            default = None if param.default is param.empty else param.default
            parameters[param_name] = (
                default if isinstance(default, (str, int, float, bool, type(None))) else repr(default)
            )
        described.append(
            {
                "name": name,
                "class": cls.__name__,
                "aliases": sorted(
                    alias for alias, target in _SPLITTER_ALIASES.items() if target == name and alias != name
                ),
                "summary": doc.strip().splitlines()[0] if doc.strip() else "",
                **_SPLITTER_INFO.get(name, {}),
                "parameters": parameters,
            }
        )
    return described


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
