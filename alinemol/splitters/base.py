"""
Base classes for molecular splitters.

This module provides the abstract base class and mixins that all molecular
splitters in ALineMol should inherit from to ensure a consistent API.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Iterator, Tuple, List

import numpy as np
from sklearn.model_selection import BaseCrossValidator


class BaseMolecularSplitter(BaseCrossValidator, ABC):
    """
    Abstract base class for all molecular splitters in ALineMol.

    All splitters should inherit from this class to ensure a consistent API.
    The primary API is SMILES-first, with optional feature input support.

    Args:
        n_splits: Number of re-shuffling & splitting iterations.
        test_size: Proportion or absolute number of samples for test set.
        train_size: Proportion or absolute number of samples for train set.
        random_state: Random seed for reproducibility.

    Attributes:
        n_splits: Number of splitting iterations.
        test_size: Size of the test set.
        train_size: Size of the training set.
        random_state: Random state for reproducibility.

    Example:
        >>> class MySplitter(BaseMolecularSplitter):
        ...     def _iter_indices(self, X, y=None, groups=None):
        ...         # Implementation here
        ...         yield train_idx, test_idx
        ...
        >>> splitter = MySplitter(n_splits=5, test_size=0.2)
        >>> for train_idx, test_idx in splitter.split(smiles_list):
        ...     train = [smiles_list[i] for i in train_idx]
        ...     test = [smiles_list[i] for i in test_idx]
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[int] = None,
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self._smiles: Optional[List[str]] = None

    @abstractmethod
    def _iter_indices(
        self,
        X: Union[List[str], np.ndarray],
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate (train, test) indices. Must be implemented by subclasses.

        Args:
            X: SMILES strings or feature array.
            y: Target variable (optional).
            groups: Group labels for group-based splitting (optional).

        Yields:
            Tuple of (train_indices, test_indices) as numpy arrays.
        """
        pass

    def split(
        self,
        X: Union[List[str], np.ndarray],
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test sets.

        Args:
            X: SMILES strings or feature array. If SMILES strings are provided,
                they will be converted to features internally when needed.
            y: Target variable (optional). Used for stratified splitting.
            groups: Group labels (optional). Used for group-based splitting.

        Yields:
            train_indices: Array of training set indices.
            test_indices: Array of test set indices.

        Example:
            >>> splitter = MySplitter(n_splits=3, test_size=0.2)
            >>> smiles = ["CCO", "c1ccccc1", "CCN", "CCCC", "CC(C)C"]
            >>> for train_idx, test_idx in splitter.split(smiles):
            ...     print(f"Train: {train_idx}, Test: {test_idx}")
        """
        yield from self._iter_indices(X, y, groups)

    def get_n_splits(
        self,
        X: Optional[Union[List[str], np.ndarray]] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        """
        Return the number of splitting iterations.

        Args:
            X: Ignored, present for API compatibility.
            y: Ignored, present for API compatibility.
            groups: Ignored, present for API compatibility.

        Returns:
            Number of splitting iterations (n_splits).
        """
        return self.n_splits

    @staticmethod
    def _is_smiles_input(X: Union[List[str], np.ndarray]) -> bool:
        """
        Check if input is a list/array of SMILES strings.

        Args:
            X: Input data to check.

        Returns:
            True if X appears to be SMILES strings, False otherwise.
        """
        if X is None:
            return False
        if isinstance(X, np.ndarray) and X.ndim > 1:
            return False
        # Check first few elements to determine if they're strings
        try:
            sample = X[: min(5, len(X))] if len(X) > 0 else []
            return all(isinstance(x, str) for x in sample)
        except (TypeError, IndexError):
            return False

    def _convert_smiles_to_features(
        self,
        smiles: List[str],
        fp_type: str = "ecfp",
        n_jobs: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Convert SMILES strings to molecular fingerprints.

        Args:
            smiles: List of SMILES strings.
            fp_type: Fingerprint type (default: "ecfp" for ECFP4).
            n_jobs: Number of parallel jobs (-1 for all CPUs).
            **kwargs: Additional arguments passed to fingerprint function.

        Returns:
            2D numpy array of fingerprints (n_samples, n_bits).

        Raises:
            ImportError: If datamol is not installed.
        """
        try:
            import datamol as dm
        except ImportError:
            raise ImportError(
                "datamol is required for SMILES to fingerprint conversion. Install it with: pip install datamol"
            )

        # Set default fingerprint parameters for ECFP4
        if fp_type == "ecfp" and "radius" not in kwargs:
            kwargs["radius"] = 2
        if fp_type == "ecfp" and "nBits" not in kwargs:
            kwargs["nBits"] = 1024

        def _to_fp(smi: str) -> np.ndarray:
            mol = dm.to_mol(smi)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smi}")
            return dm.to_fp(mol, fp_type=fp_type, **kwargs)

        if n_jobs is None:
            n_jobs = -1

        fps = dm.utils.parallelized(_to_fp, smiles, n_jobs=n_jobs)
        return np.array(fps)

    def set_smiles(self, smiles: List[str]) -> "BaseMolecularSplitter":
        """
        Set SMILES for splitting when features are passed to split().

        This is useful when you want to pass pre-computed features to split()
        but the splitter needs access to the original SMILES strings.

        Args:
            smiles: List of SMILES strings.

        Returns:
            Self, for method chaining.

        Example:
            >>> features = compute_fingerprints(smiles_list)
            >>> splitter = ScaffoldSplit().set_smiles(smiles_list)
            >>> for train_idx, test_idx in splitter.split(features):
            ...     pass
        """
        self._smiles = list(smiles)
        return self

    def _resolve_smiles(
        self,
        X: Union[List[str], np.ndarray],
    ) -> List[str]:
        """
        Resolve SMILES from either X or stored _smiles.

        Args:
            X: Input that may be SMILES strings or features.

        Returns:
            List of SMILES strings.

        Raises:
            ValueError: If X is not SMILES and no SMILES were stored via set_smiles().
        """
        if self._is_smiles_input(X):
            return list(X)
        if self._smiles is not None:
            return self._smiles
        raise ValueError(
            "Input is not SMILES strings and no SMILES were provided. "
            "Either pass SMILES as X or call set_smiles() before split()."
        )

    def _get_test_size(self, n_samples: int) -> int:
        """
        Calculate absolute test size from test_size parameter.

        Args:
            n_samples: Total number of samples.

        Returns:
            Absolute number of test samples.
        """
        if self.test_size is None:
            return int(n_samples * 0.2)  # Default 20%
        elif isinstance(self.test_size, float):
            return int(n_samples * self.test_size)
        else:
            return int(self.test_size)

    def _get_train_size(self, n_samples: int, n_test: int) -> int:
        """
        Calculate absolute train size from train_size parameter.

        Args:
            n_samples: Total number of samples.
            n_test: Number of test samples.

        Returns:
            Absolute number of train samples.
        """
        if self.train_size is None:
            return n_samples - n_test
        elif isinstance(self.train_size, float):
            return int(n_samples * self.train_size)
        else:
            return int(self.train_size)

    def __repr__(self) -> str:
        """Return string representation of the splitter."""
        params = [
            f"n_splits={self.n_splits}",
            f"test_size={self.test_size}",
            f"random_state={self.random_state}",
        ]
        return f"{self.__class__.__name__}({', '.join(params)})"
