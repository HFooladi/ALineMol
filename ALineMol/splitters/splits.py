from dataclasses import dataclass

import numpy as np
from astartes.samplers import AbstractSampler
from astartes.samplers.extrapolation import Scaffold
from astartes.utils.aimsim_featurizer import featurize_molecules


class TargetProperty(AbstractSampler):
    def _sample(self):
        """
        This sampler partitions the data based on the regression target y. It first sorts the
        data by y value and then constructs the training set to have either the smallest (largest)
        y values, the validation set to have the next smallest (largest) set of y values, and the
        testing set to have the largest (smallest) y values.
        Implements the target property sampler to create an extrapolation split.
        """
        data = [(y, idx) for y, idx in zip(self.y, np.arange(len(self.y)))]

        # by default, the smallest property values are placed in the training set
        sorted_list = sorted(data, reverse=self.get_config("descending", False))

        self._samples_idxs = np.array([idx for time, idx in sorted_list], dtype=int)


class MolecularWeight(AbstractSampler):
    def _before_sample(self):
        # check for invalid data types using the method in the Scaffold sampler
        Scaffold._validate_input(self.X)
        # calculate the average molecular weight of the molecule
        self.y_backup = self.y
        self.y = featurize_molecules(
            (Scaffold.str_to_mol(i) for i in self.X), "mordred:MW", fprints_hopts={}
        )

    def _after_sample(self):
        # restore the original y values
        self.y = self.y_backup
