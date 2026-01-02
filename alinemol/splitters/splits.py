# The goal is implementing a custom splitter that splits the molecular datasets.
import datamol as dm
from typing import Optional, Union, List, Tuple, Iterator

import numpy as np
from loguru import logger
from sklearn.model_selection import BaseShuffleSplit, ShuffleSplit
from sklearn.model_selection._split import _validate_shuffle_split  # noqa W0212
from sklearn.utils.validation import _num_samples  # noqa W0212
from dgl.data.utils import Subset
from sklearn.model_selection import StratifiedShuffleSplit

from alinemol.utils.typing import LabeledDataset, DatasetSplit, KFoldSplit, RandomStateType, SMILESList
from alinemol.splitters.factory import register_splitter


def stratified_split_dataset(
    dataset: LabeledDataset,
    frac_list: Optional[List[float]] = None,
    shuffle: bool = True,
    random_state: RandomStateType = None,
) -> DatasetSplit:
    """Split dataset into training, validation and test set with stratified shffle splitting.

    Args:
        dataset : LabeledDataset
            We assume ``len(dataset)`` gives the number of datapoints and ``dataset[i]``
            gives the ith datapoint.
        frac_list : list or None, optional
            A list of length 3 containing the fraction to use for training,
            validation and test. If None, we will use [0.8, 0.1, 0.1].
        shuffle : bool, optional
            If True, we will first randomly shuffle the dataset.
        random_state : None, int or array_like, optional
            Random seed used to initialize the pseudo-random number generator.
            Can be any integer between 0 and 2**32 - 1 inclusive, an array
            (or other sequence) of such integers, or None (the default).
            If seed is None, then RandomState will try to read data from /dev/urandom
            (or the Windows analogue) if available or seed from the clock otherwise.

    Returns:
        list of length 3
            Subsets for training, validation and test.
    """
    if frac_list is None:
        frac_list = [0.8, 0.1, 0.1]
    frac_list = np.asarray(frac_list)
    assert np.allclose(np.sum(frac_list), 1.0), "Expect frac_list sum to 1, got {:.4f}".format(np.sum(frac_list))
    num_data = len(dataset)

    # Validate dataset has labels attribute
    if not hasattr(dataset, "labels"):
        raise AttributeError("Dataset must have a 'labels' attribute for stratified splitting")

    # Validate labels are appropriate for stratification
    if len(dataset.labels) == 0:
        raise ValueError("Dataset is empty")

    # Check for sufficient class representation
    unique_labels = np.unique(dataset.labels)
    if len(unique_labels) < 2:
        raise ValueError(f"Need at least 2 classes for stratification, found {len(unique_labels)}")

    # Step 1: Split into train+val and test
    split = StratifiedShuffleSplit(n_splits=1, test_size=frac_list[2], random_state=random_state)
    train_val_indices, test_indices = next(split.split(np.zeros(num_data), dataset.labels))

    # Step 2: Split train+val into train and val
    val_relative_ratio = frac_list[1] / (frac_list[0] + frac_list[1])  # Adjusted for train+val proportion
    split = StratifiedShuffleSplit(n_splits=1, test_size=val_relative_ratio, random_state=random_state)
    train_indices, val_indices = next(split.split(np.zeros(len(train_val_indices)), dataset.labels[train_val_indices]))

    return [
        Subset(dataset, train_val_indices[train_indices]),
        Subset(dataset, train_val_indices[val_indices]),
        Subset(dataset, test_indices),
    ]


@register_splitter("molecular_logp", aliases=["logp", "molecular-logp"])
class MolecularLogPSplit(BaseShuffleSplit):
    """
    Split a molecular dataset by sorting molecules according to their LogP values.

    This splitter is designed for chemical domain shift experiments, where
    you want to evaluate how well models generalize to molecules with different
    physical properties than those they were trained on. LogP (octanol-water partition
    coefficient) is a measure of lipophilicity, which affects molecular solubility,
    permeability, and binding properties.

    The splitter works by:
    1. Calculating LogP values for all molecules
    2. Sorting molecules by their LogP values
    3. Splitting the sorted list according to train/test size parameters

    When generalize_to_larger=True (default), the training set contains molecules with
    lower LogP values, and the test set contains those with higher LogP values. This
    mimics the real-world scenario of testing on molecules with properties outside the
    training distribution.

    Args:
        generalize_to_larger: bool, default=True
            If True, train set will have smaller LogP values, test set will have larger values.
            If False, train set will have larger LogP values, test set will have smaller values.
        n_splits: int, default=5
            Number of re-shuffling & splitting iterations. Note that for this deterministic
            splitter, all iterations will produce the same split.
        smiles: List[str], optional
            List of SMILES strings if not provided directly as input in split() or _iter_indices().
            Useful when the input X to those methods is not a list of SMILES strings but some
            other feature representation.
        test_size: float or int, optional
            If float, represents the proportion of the dataset to include in the test split.
            If int, represents the absolute number of test samples.
            If None, the value is set to the complement of the train size.
        train_size: float or int, optional
            If float, represents the proportion of the dataset to include in the train split.
            If int, represents the absolute number of train samples.
            If None, the value is automatically set to the complement of the test size.
        random_state: int or RandomState instance, optional
            Controls the randomness of the training and testing indices produced.
            Note that this splitter is deterministic, so random_state only affects
            the implementation of _validate_shuffle_split.

    Examples:
        >>> from alinemol.splitters import MolecularLogPSplit
        >>> import numpy as np
        >>> # Example with list of SMILES
        >>> smiles = ["CCO", "CC(=O)O", "c1ccccc1", "CCN", "CCCCCCC"]
        >>> splitter = MolecularLogPSplit(generalize_to_larger=True, test_size=0.4)
        >>> for train_idx, test_idx in splitter.split(smiles):
        ...     print(f"Training on: {[smiles[i] for i in train_idx]}")
        ...     print(f"Testing on: {[smiles[i] for i in test_idx]}")
        ...     break  # Just show the first split

        >>> # Example with separate features and target
        >>> X = np.random.randn(5, 10)  # Some molecular features
        >>> y = np.random.randint(0, 2, 5)  # Binary target
        >>> splitter = MolecularLogPSplit(smiles=smiles, test_size=0.4)
        >>> for train_idx, test_idx in splitter.split(X, y):
        ...     X_train, X_test = X[train_idx], X[test_idx]
        ...     y_train, y_test = y[train_idx], y[test_idx]
        ...     break  # Just show the first split

    Notes:
        - LogP values are calculated using the Crippen method implemented in datamol
        - This splitter is deterministic - calling split() multiple times will
          produce the same split regardless of n_splits value
        - Useful for testing model extrapolation to molecules with different
          physical-chemical properties than the training set
    """

    def __init__(
        self,
        generalize_to_larger: bool = True,
        n_splits: int = 5,
        smiles: Optional[SMILESList] = None,
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        random_state: RandomStateType = None,
    ) -> None:
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self._smiles = smiles
        self._generalize_to_larger = generalize_to_larger

    def _iter_indices(
        self,
        X: Union[SMILESList, np.ndarray],
        y: Optional[np.ndarray] = None,
        groups: Optional[Union[int, np.ndarray]] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate indices to split data into training and test sets based on LogP values.

        Args:
            X : list of strings or numpy array
                List of SMILES strings to split, or features array if smiles
                was provided in the constructor.
            y : numpy array, optional
                Target variable for supervised learning problems.
                Not used, present for API consistency.
            groups : numpy array, optional
                Group labels for the samples.
                Not used, present for API consistency.

        Yields:
            train_indices : numpy array
                Indices of training samples, sorted by LogP values.
            test_indices : numpy array
                Indices of testing samples, sorted by LogP values.

        Raises:
            ValueError:
                If X is not a list of SMILES strings and no SMILES list was
                provided during initialization.
        """

        requires_smiles = X is None or not all(isinstance(x, str) for x in X)
        if self._smiles is None and requires_smiles:
            raise ValueError("If the input is not a list of SMILES, you need to provide the SMILES to the constructor.")

        smiles = self._smiles if requires_smiles else X

        n_samples = _num_samples(smiles)
        n_train, n_test = _validate_shuffle_split(
            n_samples,
            self.test_size,
            self.train_size,
            default_test_size=self._default_test_size,
        )

        mols = dm.utils.parallelized(dm.to_mol, smiles, n_jobs=1, progress=False)
        mlogps = dm.utils.parallelized(dm.descriptors.clogp, mols, n_jobs=1, progress=False)

        sorted_idx = np.argsort(mlogps)

        if self.n_splits > 1:
            logger.warning(
                f"n_splits={self.n_splits} > 1, but {self.__class__.__name__} is deterministic "
                f"and will always return the same split!"
            )

        for i in range(self.n_splits):
            if self._generalize_to_larger:
                yield sorted_idx[:n_train], sorted_idx[n_train:]
            else:
                yield sorted_idx[n_test:], sorted_idx[:n_test]


@register_splitter("random", aliases=["random_split"])
class RandomSplit(ShuffleSplit):
    """
    Splits the dataset randomly.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        random_state: RandomStateType = None,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )

    def _iter_indices(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ):
        """Generate (train, test) indices"""
        if X is None:
            raise ValueError(f"{self.__class__.__name__} requires X to be provided.")
        _ = _num_samples(X)
        for train, test in super()._iter_indices(X, y, groups):
            yield train, test


class StratifiedRandomSplit(object):
    """Randomly reorder datasets and then split them. make sure that the label distribution
    among the training, validation and test sets are the same as the original dataset.

    The dataset is split with permutation and the splitting is hence stratified random.
    """

    @staticmethod
    def train_val_test_split(
        dataset: LabeledDataset,
        frac_train: float = 0.8,
        frac_val: float = 0.1,
        frac_test: float = 0.1,
        random_state: RandomStateType = None,
    ) -> DatasetSplit:
        """Randomly permute the dataset and then stratified split it into
        three consecutive chunks for training, validation and test.

        Args:
            dataset: LabeledDataset
                We assume ``len(dataset)`` gives the size for the dataset and ``dataset[i]``
                gives the ith datapoint.
            frac_train: float
                Fraction of data to use for training. By default, we set this to be 0.8, i.e.
                80% of the dataset is used for training.
            frac_val: float
                Fraction of data to use for validation. By default, we set this to be 0.1, i.e.
                10% of the dataset is used for validation.
            frac_test: float
                Fraction of data to use for test. By default, we set this to be 0.1, i.e.
                10% of the dataset is used for test.
            random_state: None, int or array_like, optional
                Random seed used to initialize the pseudo-random number generator.
                Can be any integer between 0 and 2**32 - 1 inclusive, an array
                (or other sequence) of such integers, or None (the default).
                If seed is None, then RandomState will try to read data from /dev/urandom
                (or the Windows analogue) if available or seed from the clock otherwise.

        Returns:
            list of length 3
                Subsets for training, validation and test, which also have ``len(dataset)``
                and ``dataset[i]`` behaviors.
        """

        return stratified_split_dataset(
            dataset, frac_list=[frac_train, frac_val, frac_test], shuffle=True, random_state=random_state
        )

    @staticmethod
    def k_fold_split(
        dataset: LabeledDataset, k: int = 5, random_state: RandomStateType = None, log: bool = True
    ) -> KFoldSplit:
        """Performs stratified k-fold split of the dataset.

        Args:
            dataset: LabeledDataset
                We assume ``len(dataset)`` gives the size for the dataset and ``dataset[i]``
                gives the ith datapoint. The dataset should have a 'labels' attribute.
            k: int
                Number of folds. Default is 5.
            random_state: None, int or array_like, optional
                Random seed used to initialize the pseudo-random number generator.
                Can be any integer between 0 and 2**32 - 1 inclusive, an array
                (or other sequence) of such integers, or None (the default).
            log: bool
                Whether to log information about the split. Default is True.

        Returns:
            list of tuples
                Each tuple contains (train_set, val_set) where train_set and val_set
                are Subset objects of the original dataset.
        """
        from sklearn.model_selection import StratifiedKFold

        # Validate dataset has labels attribute
        if not hasattr(dataset, "labels"):
            raise AttributeError("Dataset must have a 'labels' attribute for stratified splitting")

        num_data = len(dataset)
        if log:
            logger.info(f"Performing {k}-fold stratified cross-validation on {num_data} samples")

        # Initialize the stratified k-fold splitter
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

        # Create the folds
        fold_datasets = []
        for train_idx, val_idx in skf.split(np.zeros(num_data), dataset.labels):
            train_set = Subset(dataset, train_idx)
            val_set = Subset(dataset, val_idx)
            fold_datasets.append((train_set, val_set))

            if log:
                logger.info(f"Split created: train size = {len(train_idx)}, validation size = {len(val_idx)}")

        return fold_datasets
