# The goal is implementing a custom splitter that splits the molecular datasets.
import datamol as dm
from typing import Optional, Union, Sequence

import numpy as np
from loguru import logger
from numpy.random import RandomState
from sklearn.model_selection import BaseShuffleSplit, ShuffleSplit
from sklearn.model_selection._split import _validate_shuffle_split  # noqa W0212
from sklearn.utils.validation import _num_samples  # noqa W0212
from dgl.data.utils import Subset
from sklearn.model_selection import StratifiedShuffleSplit


def stratified_split_dataset(dataset, frac_list=None, shuffle=True, random_state=None):
    """Split dataset into training, validation and test set with stratified shffle splitting.

    Args:
        dataset
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

    # Step 1: Split into train+val and test
    split = StratifiedShuffleSplit(n_splits=1, test_size=frac_list[2], random_state=random_state)
    for train_val_indices, test_indices in split.split(np.zeros(num_data), dataset.labels):
        pass

    # Step 2: Split train+val into train and val
    val_ralative_ratio = frac_list[1] / (frac_list[0] + frac_list[1])  # Adjusted for train+val proportion
    split = StratifiedShuffleSplit(n_splits=1, test_size=val_ralative_ratio, random_state=random_state)

    for train_indices, val_indices in split.split(np.zeros(len(train_val_indices)), dataset.labels[train_val_indices]):
        pass

    return [
        Subset(dataset, train_val_indices[train_indices]),
        Subset(dataset, train_val_indices[val_indices]),
        Subset(dataset, test_indices),
    ]


class MolecularLogPSplit(BaseShuffleSplit):
    """
    Splits the dataset by sorting the molecules by their LogP values
    and then finding an appropriate cutoff to split the molecules in two sets.
    """

    def __init__(
        self,
        generalize_to_larger: bool = True,
        n_splits: int = 5,
        smiles: Optional[Sequence[str]] = None,
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[Union[int, RandomState]] = None,
    ):
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
        X: Union[Sequence[str], np.ndarray],
        y: Optional[np.ndarray] = None,
        groups: Optional[Union[int, np.ndarray]] = None,
    ):
        """Generate (train, test) indices"""

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


class RandomSplit(ShuffleSplit):
    """
    Splits the dataset randomly.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[Union[int, RandomState]] = None,
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
        n_samples = _num_samples(X)
        for train, test in super()._iter_indices(X, y, groups):
            yield train, test


class StratifiedRandomSplitter(object):
    """Randomly reorder datasets and then split them. make sure that the label distribution
    among the training, validation and test sets are the same as the original dataset.

    The dataset is split with permutation and the splitting is hence stratified random.
    """

    @staticmethod
    def train_val_test_split(dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=None):
        """Randomly permute the dataset and then stratified split it into
        three consecutive chunks for training, validation and test.

        Args:
            dataset
                We assume ``len(dataset)`` gives the size for the dataset and ``dataset[i]``
                gives the ith datapoint.
            frac_train : float
                Fraction of data to use for training. By default, we set this to be 0.8, i.e.
                80% of the dataset is used for training.
            frac_val : float
                Fraction of data to use for validation. By default, we set this to be 0.1, i.e.
                10% of the dataset is used for validation.
            frac_test : float
                Fraction of data to use for test. By default, we set this to be 0.1, i.e.
                10% of the dataset is used for test.
            random_state : None, int or array_like, optional
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
    def k_fold_split(dataset, k=5, random_state=None, log=True):
        """

        Args:


        Returen:

        """
        pass
