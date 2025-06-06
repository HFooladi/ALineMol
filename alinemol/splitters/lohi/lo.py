"""
This code is adapted from https://github.com/datamol-io/splito/blob/main/splito/lohi/_lo.py

The LoSplit needs a continuous activity value, and not a binary label.
"""

from rdkit import DataStructs
import numpy as np
from tqdm import tqdm
import datamol as dm
import functools
from loguru import logger
from typing import List, Tuple, Optional, Any


class LoSplit:
    def __init__(
        self,
        threshold: float = 0.4,
        min_cluster_size: int = 5,
        max_clusters: int = 50,
        std_threshold: float = 0.60,
    ):
        """
        A splitter that prepares data for training ML models for Lead Optimization or to guide
        molecular generative models. These models must be sensitive to minor modifications of
        molecules, and this splitter constructs a test that allows the evaluation of a model's
        ability to distinguish those modifications.

        Args:
            threshold: ECFP4 1024-bit Tanimoto similarity threshold.
                Molecules more similar than this threshold are considered too similar and can be grouped together in one cluster.
            min_cluster_size: the minimum number of molecules per cluster.
            max_clusters: the maximum number of selected clusters. The remaining molecules go to the training set.
                This can be useful for limiting your test set to get more molecules in the train set.
            std_threshold: the lower bound of the acceptable standard deviation for a cluster's values. It should be greater than the measurement noise.
                For ChEMBL-like data set it to 0.60 for logKi and 0.70 for logIC50.
                Set it lower if you have a high-quality dataset.

        For more information, see a tutorial in the docs and Steshin 2023, Lo-Hi: Practical ML Drug Discovery Benchmark.
        """
        self.threshold = threshold
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.std_threshold = std_threshold

    def split(
        self, smiles: List[str], values: List[float], n_jobs: int = -1, verbose: int = 1
    ) -> Tuple[List[int], List[List[int]]]:
        """
        Split the dataset into test clusters and train.

        Args:
            smiles: list of SMILES strings representing molecules
            values: list of their continuous activity values
            n_jobs: number of parallel jobs to run, -1 means use all processors
            verbose: set to 0 to turn off progressbar

        Returns:
            train_idx: list of indices for training set
            clusters_idx: list of lists containing indices for each cluster
        """
        # Convert inputs to numpy arrays if needed
        if not isinstance(smiles, np.ndarray):
            smiles = np.array(smiles)
        if not isinstance(values, np.ndarray):
            values = np.array(values)

        # Get clusters and training indices
        train_idx, clusters_idx, central_nodes = self._select_distinct_clusters(smiles, values, n_jobs, verbose)
        train_idx = list(train_idx) + central_nodes

        if not clusters_idx:
            logger.warninig("No clusters were found. Was your std_threshold too constrained?")

        return train_idx, clusters_idx

    def _select_distinct_clusters(
        self, smiles: np.ndarray, values: np.ndarray, n_jobs: int, verbose: int
    ) -> Tuple[np.ndarray, List[np.ndarray], List[int]]:
        """
        A greedy algorithm to select clusters from neighborhood graph of molecules.

        Args:
            smiles: array of SMILES strings
            values: array of activity values
            n_jobs: number of parallel jobs
            verbose: whether to show progress bar

        Returns:
            train_nodes: array of indices for training set
            all_clusters_nodes: list of arrays containing indices for each cluster
            central_nodes: list of indices for central nodes of each cluster
        """
        if verbose:
            progress_bar = tqdm(total=self.max_clusters, desc="Collecting clusters")

        # At first, all the nodes are in the train set. Some will be moved to the list of clusters.
        train_nodes = np.array(range(len(smiles)))

        # Generate fingerprints for all molecules
        train_fps = dm.parallelized(
            functools.partial(dm.to_fp, as_array=False, radius=2, fpSize=1024),
            smiles,
            n_jobs=n_jobs,
        )
        all_clusters_nodes = []  # the test clusters of nodes
        central_nodes = []  # central nodes of the clusters

        # Keep finding clusters until we hit max_clusters or run out of valid clusters
        while len(all_clusters_nodes) < self.max_clusters:
            total_neighbours, stds = self._get_neighborhood(train_fps, values)
            central_idx = self._get_central_idx(total_neighbours, stds)
            if central_idx is None:
                break  # there are no more clusters
            central_nodes.append(train_nodes[central_idx])

            cluster_indices = self._collect_cluster(central_idx, train_fps)

            # Save the cluster nodes
            all_clusters_nodes.append(train_nodes[cluster_indices])

            # Remove neighbours of the cluster from the rest of nodes
            nearest_sim = self._get_nearest_sim(train_fps, cluster_indices + [central_idx])
            rest_idx = []
            for idx, sim in enumerate(nearest_sim):
                if sim < self.threshold:
                    rest_idx.append(idx)
            train_nodes = train_nodes[rest_idx]
            values = values[rest_idx]
            train_fps = [train_fps[idx] for idx in rest_idx]

            if verbose:
                progress_bar.update(1)
        if verbose:
            progress_bar.close()
        logger.info(f"Found {len(all_clusters_nodes)} clusters.")
        return train_nodes, all_clusters_nodes, central_nodes

    def _get_neighborhood(self, train_fps: List[Any], values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        For each node find number of neighbours and std of their values.

        Args:
            train_fps: list of molecular fingerprints
            values: array of activity values

        Returns:
            total_neighbours: array containing number of neighbors for each molecule
            stds: array containing standard deviations of values for each neighborhood
        """
        total_neighbours = []
        stds = []
        for fps in train_fps:
            sims = DataStructs.BulkTanimotoSimilarity(fps, train_fps)
            is_neighbor = np.array(sims) > self.threshold
            total_neighbours.append(is_neighbor.sum())
            stds.append(values[is_neighbor].std())

        total_neighbours = np.array(total_neighbours)
        stds = np.array(stds)
        return total_neighbours, stds

    def _get_central_idx(self, total_neighbours: np.ndarray, stds: np.ndarray) -> Optional[int]:
        """
        Find the most distant cluster and return the index of its centroid.

        Args:
            total_neighbours: array containing number of neighbors for each molecule
            stds: array containing standard deviations of values for each neighborhood

        Returns:
            central_idx: index of the chosen central node, or None if no valid cluster found
        """
        central_idx = None
        least_neighbours = max(total_neighbours)
        for idx, n_neighbours in enumerate(total_neighbours):
            if n_neighbours > self.min_cluster_size:
                if n_neighbours < least_neighbours:
                    if stds[idx] >= self.std_threshold:
                        least_neighbours = n_neighbours
                        central_idx = idx
        return central_idx

    def _collect_cluster(self, central_idx: int, train_fps: List[Any]) -> List[int]:
        """
        Collect list of neighbours of the central_idx.

        Args:
            central_idx: index of the central node
            train_fps: list of molecular fingerprints

        Returns:
            cluster_indices: list of indices for molecules in the cluster (excluding central)
        """
        sims = DataStructs.BulkTanimotoSimilarity(train_fps[central_idx], train_fps)
        is_neighbour = np.array(sims) > self.threshold
        cluster_indices = []
        for idx, value in enumerate(is_neighbour):
            if value:
                if idx != central_idx:
                    cluster_indices.append(idx)
        return cluster_indices

    def _get_nearest_sim(self, train_fps: List[Any], indices_to_remove: List[int]) -> List[float]:
        """
        For each train molecule find the maximal similarity to molecules in the cluster.

        Args:
            train_fps: list of molecular fingerprints
            indices_to_remove: indices of molecules to compare against

        Returns:
            nearest_sim: list of maximum similarities for each molecule
        """
        cluster_fps = [train_fps[idx] for idx in indices_to_remove]
        nearest_sim = []
        for train_fp in train_fps:
            sims = DataStructs.BulkTanimotoSimilarity(train_fp, cluster_fps)
            nearest_sim.append(max(sims))
        return nearest_sim
