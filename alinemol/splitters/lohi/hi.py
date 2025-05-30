"""
This code is adapted from https://github.com/SteshinSS/lohi_splitter

The original code is licensed under the MIT License.
"""

import networkx as nx
import numpy as np
from typing import List, Tuple, Set, Optional, Dict, Iterator, Union

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
import mip

from sklearn.model_selection import BaseShuffleSplit
from alinemol.utils.typing import SMILESList


def get_neighborhood_graph(smiles: List[str], threshold: float) -> nx.Graph:
    """
    Builds a neighborhood graph from smiles list.

    A neighborhood graph is a graph whose nodes correspond to the smiles,
    and two nodes are connected iff the corresponding smiles have ECFP4 Tanimoto similarity greater than the threshold.

    Args:
        smiles: List of SMILES strings representing molecules
        threshold: Molecules with similarity larger than this number are considered neighbors

    Returns:
        nx.Graph: Neighborhood graph where nodes are molecules and edges represent high similarity
    """
    # Generate Morgan fingerprints for each molecule
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    fps = [fp_generator.GetFingerprint(x) for x in mols]

    # Calculate pairwise similarity matrix
    similarity_matrix = []
    for fp in fps:
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, fps))
        similarity_matrix.append(sims)

    # Create graph and add edges for similar molecules
    G = nx.Graph()
    for i in range(len(mols)):
        for j in range(i, len(mols)):
            similarity = similarity_matrix[i][j]
            if similarity > threshold:
                G.add_edge(i, j, weight=similarity)
    nx.set_node_attributes(G, 1.0, "weight")
    return G


def get_giant_component(G: nx.Graph) -> Tuple[nx.Graph, List[Set[int]]]:
    """
    Returns the giant component and smaller components.

    Most neighborhood graphs of molecular datasets contain one giant component and many small ones.
    We apply min vertex k-cut algorithm to the giant component and assign small ones between folds manually.

    Args:
        G: Neighborhood graph (nx.Graph)

    Returns:
        Tuple containing:
            - nx.Graph: The giant component
            - List[Set[int]]: List of small components, where each component is a set of node ids
    """
    # Find all connected components
    components = [x for x in nx.connected_components(G)]
    biggest_component_idx = np.argmax([len(x) for x in components])
    biggest_component = components[biggest_component_idx]
    S = G.subgraph(biggest_component).copy()

    # Collect all other components
    small_components = []
    for i, component in enumerate(components):
        if i == biggest_component_idx:
            continue
        small_components.append(component)
    return S, small_components


def coarse_graph(graph: nx.Graph, threshold: float) -> Tuple[nx.Graph, np.ndarray]:
    """
    Clusters nodes with a large number of neighbors and return condensed graph.

    Min vertex k-cut might take a long time. To speed it up we can condense graph,
    and cut clusters of molecules instead of individual molecules.

    Args:
        graph: Graph to condense (nx.Graph)
        threshold: Molecules with ECFP4 Tanimoto similarity greater than this number will be considered for clustering

    Returns:
        Tuple containing:
            - nx.Graph: Condensed graph
            - np.ndarray: Array that maps original graph's nodes to nodes of the condensed graph's nodes
    """
    # Calculate number of neighbors for each node
    n_neighbors = []
    for node in graph.nodes():
        edges = graph.edges(node, data=True)
        total_neighbors = sum(1 for edge in edges if edge[2]["weight"] > threshold)
        n_neighbors.append((total_neighbors, len(n_neighbors)))

    # Cluster nodes starting with those having most neighbors
    n_neighbors = sorted(n_neighbors, key=lambda x: -x[0])
    node_to_cluster = [-1] * len(graph)
    current_cluster = 0

    for _, node in n_neighbors:
        if node_to_cluster[node] != -1:
            continue

        # Create new cluster with this node and its neighbors
        node_to_cluster[node] = current_cluster
        edges = graph.edges(node, data=True)
        for edge in edges:
            if edge[2]["weight"] > threshold:
                adjacent_node = edge[1]
                if node_to_cluster[adjacent_node] == -1:
                    node_to_cluster[adjacent_node] = current_cluster
        current_cluster += 1

    node_to_cluster = np.array(node_to_cluster)

    # Build condensed graph
    condensed_graph = nx.Graph()

    # Add nodes with weights based on cluster sizes
    cluster_size = np.unique(node_to_cluster, return_counts=True)[1]
    for cluster in range(current_cluster):
        condensed_graph.add_node(cluster, weight=cluster_size[cluster])

    # Add edges between clusters
    for cluster in range(current_cluster):
        connected_clusters = set()
        this_cluster_indices = np.where(node_to_cluster == cluster)[0]
        for node in this_cluster_indices:
            edges = graph.edges(node, data=True)
            for edge in edges:
                connected_clusters.add(node_to_cluster[edge[1]])
        for connected_cluster in connected_clusters:
            condensed_graph.add_edge(cluster, connected_cluster)

    return condensed_graph, node_to_cluster


def get_total_weight(S: nx.Graph) -> float:
    """
    Return sum of weight property of the nodes of the graph S.

    Args:
        S: Input graph

    Returns:
        float: Total weight of all nodes
    """
    return sum(S.nodes[node]["weight"] for node in S.nodes())


def get_linear_problem_no_size_constraints(
    S: nx.Graph, k: int
) -> Tuple[mip.Model, List[List[mip.Var]], List[List[float]]]:
    """
    Formulates linear programming problem for minimal vertex k-cut, without size constraints.
    See paper Simon Steshin, "Lo-Hi: Practical ML Drug Discovery Benchmark", 2023

    Args:
        S: Input graph
        k: Number of partitions

    Returns:
        Tuple containing:
            - mip.Model: The MIP model
            - List[List[mip.Var]]: Binary variables for node assignments
            - List[List[float]]: Weights for each node in each partition
    """
    m = mip.Model(sense=mip.MAXIMIZE)

    # Create binary variables for node assignments
    x = []
    for i in range(len(S)):
        per_node_x = []
        for j in range(k):
            per_node_x.append(m.add_var(var_type=mip.BINARY))
        x.append(per_node_x)

    # Get node weights
    w = []
    for i in range(len(S)):
        node = S.nodes[i]
        per_node_w = [node["weight"]] * k
        w.append(per_node_w)

    # Set objective function
    objective = []
    for i in range(len(x)):
        for k_i in range(k):
            objective.append(w[i][k_i] * x[i][k_i])
    m.objective = mip.xsum(objective)

    # Add constraints
    # Each node in one partition only
    for x_k in x:
        m += mip.xsum(x_k) <= 1

    # No edges between partitions
    for edge in S.edges:
        i, j = edge
        for k_1 in range(k):
            for k_2 in range(k):
                if k_1 == k_2:
                    continue
                m += x[i][k_1] + x[j][k_2] <= 1

    return m, x, w


def get_linear_problem_train_test(
    S: nx.Graph, train_min_frac: float, test_min_frac: float, verbose: bool = True
) -> mip.Model:
    """
    Formulate min vertex k-cut problem for k=2 (train-test split).

    Args:
        S: Connected graph to cut
        train_min_frac: Minimal fraction for train set (e.g., 0.7)
        test_min_frac: Minimal fraction for test set (e.g., 0.1)
        verbose: Whether to print status messages

    Returns:
        mip.Model: The MIP model for train-test split
    """
    k = 2
    m, x, w = get_linear_problem_no_size_constraints(S, k)

    total_weight = get_total_weight(S)
    min_train_size = total_weight * train_min_frac
    min_test_size = total_weight * test_min_frac

    if verbose:
        print("Total molecules in the giant component:", total_weight)
        print("Min train size", int(min_train_size))
        print("Min test size", int(min_test_size))

    # Add size constraints for train and test sets
    # Partitions are balanced
    train_weight = []
    test_weight = []
    for i in range(len(x)):
        train_weight.append(w[i][0] * x[i][0])
        test_weight.append(w[i][1] * x[i][1])
    m += mip.xsum(train_weight) >= min_train_size
    m += mip.xsum(test_weight) >= min_test_size

    return m


def get_linear_problem_k_fold(S: nx.Graph, fold_min_frac: float, k: int, verbose: bool = True) -> mip.Model:
    """
    Formulate min vertex k-cut problem for k.
    It is useful for k-fold cross-validation.

    Args:
        S: connected graph to cut (nx.Graph())
        fold_min_frac: minimal fraction of a part. e.g, 0.2 from the whole dataset
            It is possible that k-cut is not possible without discarding some molecules,
            so ensure k*fold_min_frac < 1.0.
        k: number of folds
        verbose: set to False, if you don't want messages.
    Returns:
        mip.Model: MIP model
    """
    m, x, w = get_linear_problem_no_size_constraints(S, k)

    total_weight = get_total_weight(S)
    lower_bound = int(total_weight * fold_min_frac)

    if verbose:
        print("Total molecules in the giant component:", total_weight)
        print("Min size of a partition:", lower_bound)

    # Partitions are balanced
    for k_i in range(k):
        partition_weight = []
        for i in range(len(x)):
            partition_weight.append(w[i][k_i] * x[i][k_i])
        m += mip.xsum(partition_weight) >= lower_bound

    return m


def solve_linear_problem(m: mip.Model, max_mip_gap: float = 0.1, verbose: bool = True) -> mip.Model:
    """
    Solves MIP linear model with default parameters.

    Args:
        m: MIP model
        max_mip_gap: value to stop optimization when the cost function is close enough to
            the optimal solution. Set it to 0.5, and you will get quick but not-optimal
            solution. Set it to 0.01, and you will get optimal solution, but it will take forever.
            See more in MIP Python documentation.
        verbose: set to False, if you don't want messages.

    Returns:
        mip.Model: Solved MIP model
    """
    m.max_mip_gap = max_mip_gap
    m.threads = -1
    m.emphasis = 2
    m.verbose = verbose
    m.optimize()
    return m


def process_results(model: mip.Model, S: nx.Graph, k: int) -> np.ndarray:
    """
    Process solution of the linear programming problem.

    Args:
        model: solved MIP model
        S: connected graph of the model
        k: number of partitions

    Return:
        np.ndarray: array. len(split) = len(S). i'th element
                    is equal to partition of the i'th node or
                    -1 if the node is discarded.
    """
    result = np.array([a.x for a in model.vars])
    split = [-1] * len(S)
    for i in range(len(S)):
        for j in range(k):
            if result[i * k + j]:
                split[i] = j
                break
    return split


def uncoarse_results(split: np.ndarray, node_to_cluster: np.ndarray) -> np.ndarray:
    """
    If the linear programming problem was sovled for coarsed graph,
    the result need to be mapped into original nodes.

    Args:
        split: array. len(split) = len(S). i'th element
               is equal to partition of the i'th node or
               -1 if the node is discarded.
        node_to_cluster: array. len(node_to_cluster) = len(S). i'th element

    Returns:
        np.ndarray: array. len(uncoarsed_split) = len(S). i'th element
                    is equal to partition of the i'th node or
                    -1 if the node is discarded.
    """
    uncoarsed_split = []
    for node in range(len(node_to_cluster)):
        coarsed_id = node_to_cluster[node]
        uncoarsed_split.append(split[coarsed_id])
    return uncoarsed_split


def map_split_to_original_idx(split: np.ndarray, k: int, new_nodes_to_old: Dict[int, int]) -> List[List[int]]:
    """
    Maps partitioning of the nodes of the giant component, to nodes of
    the original non-connected neighborhood graph.

    Args:
        split: array. len(split) = len(S). i'th element
               is equal to partition of the i'th node or
               -1 if the node is discarded.
        k: number of partitions
        new_nodes_to_old: dictionary. key is the index of the node in the original graph,
                            value is the index of the node in the giant component.

    Returns:
        List[List[int]]: list of lists. Each list contains the indices of molecules in that partition.
    """
    partitions = []
    for _ in range(k):
        partitions.append([])

    for S_idx, partition in enumerate(split):
        if partition == -1:
            continue
        G_idx = new_nodes_to_old[S_idx]
        partitions[partition].append(G_idx)
    return partitions


def assign_small_components_uniformly(
    small_components: List[List[int]], partitions: List[List[int]]
) -> List[List[int]]:
    """
    Assigns small components to the partitions uniformly.

    Args:
        small_components: list of lists. Each list contains the indices of molecules in that component.
        partitions: list of lists. Each list contains the indices of molecules in that partition.

    Returns:
        List[List[int]]: list of lists. Each list contains the indices of molecules in that partition.
    """
    for component in small_components:
        smallest_partition_idx = np.argmin([len(partition) for partition in partitions])
        partitions[smallest_partition_idx].extend(component)
    return partitions


def assign_small_components_train_test(
    small_components: List[List[int]], partitions: List[List[int]], train_min_frac: float, test_min_frac: float
) -> List[List[int]]:
    """
    Assigns small components to the partitions based on the train_min_frac and test_min_frac.

    Args:
        small_components: list of lists. Each list contains the indices of molecules in that component.
        partitions: list of lists. Each list contains the indices of molecules in that partition.
        train_min_frac: minimum fraction for the train set
        test_min_frac: minimum fraction for the test set

    Returns:
        List[List[int]]: list of lists. Each list contains the indices of molecules in that partition.
    """
    goal_ratio = train_min_frac / test_min_frac

    for component in small_components:
        current_ratio = len(partitions[0]) / len(partitions[1])
        if current_ratio > goal_ratio:
            partitions[1].extend(component)
        else:
            partitions[0].extend(component)
    return partitions


def print_partition_analysis(partitions: List[List[int]], total_molecules: int):
    """
    Prints the analysis of the partitions.

    Args:
        partitions: list of lists. Each list contains the indices of molecules in that partition.
        total_molecules: total number of molecules.

    Returns:
        None
    """
    lost_molecules = total_molecules
    for part in partitions:
        lost_molecules -= len(part)

    print()
    print("Total partitions:", len(partitions))
    print("Number of discarded molecules:", lost_molecules)
    for k in range(len(partitions)):
        print("Molecules in partition", k, ":", len(partitions[k]))


def check_model_status(model: mip.Model, is_coarsed: bool, is_train_test: bool):
    """
    Checks the status of the model.

    Args:
        model: MIP model
        is_coarsed: whether the model is coarsed
        is_train_test: whether the model is train-test split

    Returns:
        None
    """
    if model.status == mip.OptimizationStatus.INFEASIBLE or model.status == mip.OptimizationStatus.NO_SOLUTION_FOUND:
        if is_coarsed:
            if is_train_test:
                raise ValueError(
                    "No solution is found. Try to decrease train_min_frac and test_min_frac. Or try to increase coarsening_threshold."
                )
            else:
                raise ValueError(
                    "No solution is found. Try to decrease fold_min_frac. Or try to increase coarsening_threshold."
                )
        else:
            if is_train_test:
                raise ValueError("No solution is found. Try to decrease train_min_frac and test_min_frac.")
            else:
                raise ValueError("No solution is found. Try to decrease fold_min_frac.")


def hi_train_test_split(
    smiles: List[str],
    similarity_threshold=0.4,
    train_min_frac=0.7,
    test_min_frac=0.1,
    coarsening_threshold=None,
    verbose: bool = True,
    max_mip_gap: float = 0.1,
) -> List[List[int]]:
    """
    Splits a list of smiles into train and test sets such that no molecule in the test
    has ECFP4 Tanimoto similarity to the train > similarity_threshold.

    Args:
        smiles: List of smiles to split.
        similarity_threshold: ECFP4 Tanimoto threshold. Molecules in the test set won't
            have a similarity greater than similarity_threshold to those in the train set.
        train_min_frac: Minimum fraction for the train set, e.g., 0.7 of the entire dataset.
        test_min_frac: Minimum fraction for the test set, e.g., 0.1 of the entire dataset.
            It's possible that the k-cut might not be feasible without discarding some molecules,
            so ensure that the sum of train_min_frac and test_min_frac is less than 1.0.
        coarsening_threshold: Molecules with a similarity greater than the coarsening_threshold will be
            clustered together. It speeds up execution, but makes the solution less optimal.
                None -- Disables clustering (default value).
                1.0 -- Won't do anything
                0.90 -- will cluster molecules with similarity > 0.90 together
        verbose: If set to False, suppresses status messages.
        max_mip_gap: Determines when to halt optimization based on proximity to the optimal solution.
            For example, setting it to 0.5 yields a faster but less optimal solution, while 0.01 aims for a more
            optimal solution, potentially at the cost of more computation time. See more in MIP Python documentation.

    Returns:
        List[List[int]]: list of two lists. The first contains indices of train smiles, and the second contains indices of test smiles.
    """
    if not isinstance(smiles, np.ndarray):
        smiles = np.array(smiles)

    neighborhood_graph = get_neighborhood_graph(smiles, similarity_threshold)
    main_component, small_components = get_giant_component(neighborhood_graph)

    old_nodes_to_new = dict(zip(main_component.nodes(), range(main_component.number_of_nodes())))
    new_nodes_to_old = {v: k for k, v in old_nodes_to_new.items()}
    main_component = nx.relabel_nodes(main_component, old_nodes_to_new)

    if coarsening_threshold:
        coarsed_giant_component, node_to_cluster = coarse_graph(main_component, coarsening_threshold)
        model = get_linear_problem_train_test(coarsed_giant_component, train_min_frac, test_min_frac, verbose=True)
        model = solve_linear_problem(model, max_mip_gap, verbose)
        check_model_status(model, is_coarsed=True, is_train_test=True)
        giant_component_partition = process_results(model, coarsed_giant_component, k=2)
        giant_component_partition = uncoarse_results(giant_component_partition, node_to_cluster)
    else:
        model = get_linear_problem_train_test(main_component, train_min_frac, test_min_frac, verbose=True)
        model = solve_linear_problem(model, max_mip_gap, verbose)
        check_model_status(model, is_coarsed=False, is_train_test=True)
        giant_component_partition = process_results(model, main_component, k=2)

    partitions = map_split_to_original_idx(giant_component_partition, k=2, new_nodes_to_old=new_nodes_to_old)
    partitions = assign_small_components_train_test(small_components, partitions, train_min_frac, test_min_frac)
    if verbose:
        print_partition_analysis(partitions, len(smiles))
    return partitions


def hi_k_fold_split(
    smiles: List[str],
    similarity_threshold: float = 0.4,
    fold_min_frac: Optional[float] = None,
    k: int = 3,
    coarsening_threshold: Optional[float] = None,
    verbose: bool = True,
    max_mip_gap: float = 0.1,
) -> List[List[int]]:
    """
    Splits the list of smiles into k folds such that no molecule in any fold has an ECFP4 Tanimoto
    similarity greater than similarity_threshold when compared to molecules in another fold.

    Args:
        smiles: List of smiles to split
        similarity_threshold: ECFP4 Tanimoto threshold. Molecules in one fold won't
            have a similarity greater than similarity_threshold to those in another fold.
        fold_min_frac: Minimum fraction of a fold (e.g., 0.2 of the entire dataset).
            If not specified (None), it defaults to 0.9 / k.
        k: number of folds
        coarsening_threshold: Molecules with a similarity greater than the coarsening_threshold will be
            clustered together. It speeds up execution, but makes the solution less optimal.
                None -- Disables clustering (default value).
                1.0 -- Won't do anything
                0.90 -- will cluster molecules with similarity > 0.90 together
        verbose: If set to False, suppresses status messages.
        max_mip_gap: Determines when to halt optimization based on proximity to the optimal solution.
            For example, setting it to 0.5 yields a faster but less optimal solution, while 0.01 aims for a more
            optimal solution, potentially at the cost of more computation time. See more in MIP Python documentation.

    Returns:
        List[List[int]]: list of lists. Each list contains the indices of smiles in that fold.
    """
    if not isinstance(smiles, np.ndarray):
        smiles = np.array(smiles)

    if fold_min_frac is None:
        fold_min_frac = 0.9 / k

    neighborhood_graph = get_neighborhood_graph(smiles, similarity_threshold)
    main_component, small_components = get_giant_component(neighborhood_graph)

    old_nodes_to_new = dict(zip(main_component.nodes(), range(main_component.number_of_nodes())))
    new_nodes_to_old = {v: k for k, v in old_nodes_to_new.items()}
    main_component = nx.relabel_nodes(main_component, old_nodes_to_new)
    if coarsening_threshold:
        coarsed_giant_component, node_to_cluster = coarse_graph(main_component, coarsening_threshold)
        model = get_linear_problem_k_fold(coarsed_giant_component, fold_min_frac, k, verbose=True)
        model = solve_linear_problem(model, max_mip_gap, verbose)
        check_model_status(model, is_coarsed=True, is_train_test=False)
        giant_component_partition = process_results(model, coarsed_giant_component, k)
        giant_component_partition = uncoarse_results(giant_component_partition, node_to_cluster)
    else:
        model = get_linear_problem_k_fold(main_component, fold_min_frac, k, verbose=True)
        model = solve_linear_problem(model, max_mip_gap, verbose)
        check_model_status(model, is_coarsed=False, is_train_test=False)
        giant_component_partition = process_results(model, main_component, k)

    partitions = map_split_to_original_idx(giant_component_partition, k, new_nodes_to_old=new_nodes_to_old)
    partitions = assign_small_components_uniformly(small_components, partitions)
    if verbose:
        print_partition_analysis(partitions, len(smiles))
    return partitions


class HiSplit(BaseShuffleSplit):
    def __init__(
        self,
        similarity_threshold: float = 0.4,
        train_min_frac: float = 0.70,
        test_min_frac: float = 0.15,
        coarsening_threshold: Optional[float] = None,
        verbose: bool = True,
        max_mip_gap: float = 0.1,
    ):
        """
        A splitter that creates train/test splits with no molecules in the test set having
        ECFP4 Tanimoto similarity greater than similarity_threshold to molecules in the train set.

        This splitter is designed for evaluating model generalization to structurally dissimilar molecules.
        It uses a min vertex k-cut algorithm to optimally partition molecules while respecting
        similarity constraints.

        Args:
            similarity_threshold: ECFP4 Tanimoto threshold. Molecules in the test set won't
                have a similarity greater than this threshold to those in the train set.
            train_min_frac: Minimum fraction for the train set, e.g., 0.7 of the entire dataset.
            test_min_frac: Minimum fraction for the test set, e.g., 0.1 of the entire dataset.
                It's possible that the k-cut might not be feasible without discarding some molecules,
                so ensure that the sum of train_min_frac and test_min_frac is less than 1.0.
            coarsening_threshold: Molecules with a similarity greater than the coarsening_threshold will be
                clustered together. It speeds up execution, but makes the solution less optimal.
                    None -- Disables clustering (default value).
                    1.0 -- Won't do anything
                    0.90 -- will cluster molecules with similarity > 0.90 together
            verbose: If set to False, suppresses status messages.
            max_mip_gap: Determines when to halt optimization based on proximity to the optimal solution.
                For example, setting it to 0.5 yields a faster but less optimal solution, while 0.01 aims for a more
                optimal solution, potentially at the cost of more computation time.
        """
        self.similarity_threshold = similarity_threshold
        self.train_min_frac = train_min_frac
        self.test_min_frac = test_min_frac
        self.coarsening_threshold = coarsening_threshold
        self.verbose = verbose
        self.max_mip_gap = max_mip_gap

    def _iter_indices(
        self,
        X: Union[SMILESList, np.ndarray],
        y: Optional[np.ndarray] = None,
        groups: Optional[Union[int, np.ndarray]] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate indices to split data into training and test sets based on molecular similarity.

        Args:
            X: List of SMILES strings to split, or features array if smiles
                was provided in the constructor.
            y: Target variable for supervised learning problems.
                Not used, present for API consistency.
            groups: Group labels for the samples.
                Not used, present for API consistency.

        Yields:
            train_indices: numpy array of indices for training samples
            test_indices: numpy array of indices for test samples

        Raises:
            ValueError: If X is not a list of SMILES strings and no SMILES list was
                provided during initialization.
        """
        requires_smiles = X is None or not all(isinstance(x, str) for x in X)
        if self._smiles is None and requires_smiles:
            raise ValueError("If the input is not a list of SMILES, you need to provide the SMILES to the constructor.")

        smiles = self._smiles if requires_smiles else X

        # Use the existing hi_train_test_split function
        partitions = hi_train_test_split(
            smiles=smiles,
            similarity_threshold=self.similarity_threshold,
            train_min_frac=self.train_min_frac,
            test_min_frac=self.test_min_frac,
            coarsening_threshold=self.coarsening_threshold,
            verbose=self.verbose,
            max_mip_gap=self.max_mip_gap,
        )

        train_indices = np.array(partitions[0])
        test_indices = np.array(partitions[1])

        # Yield for each split (typically just once for this deterministic splitter)
        for i in range(self.n_splits):
            yield train_indices, test_indices

    def split(self, smiles: List[str]) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Split the dataset into train and test sets such that no molecule in the test
        has ECFP4 Tanimoto similarity to the train > similarity_threshold.

        Args:
            smiles: List of SMILES strings representing molecules

        Returns:
            Tuple containing:
                - List[int]: Indices of training molecules
                - List[int]: Indices of test molecules

        Example:
            >>> from alinemol.splitters.lohi import HiSplit
            >>> splitter = HiSplit()
            >>> for train_indices, test_indices in splitter.split(smiles):
            >>>     print(train_indices)
            >>>     print(test_indices)
        """
        partitions = hi_train_test_split(
            smiles=smiles,
            similarity_threshold=self.similarity_threshold,
            train_min_frac=self.train_min_frac,
            test_min_frac=self.test_min_frac,
            coarsening_threshold=self.coarsening_threshold,
            verbose=self.verbose,
            max_mip_gap=self.max_mip_gap,
        )

        yield np.array(partitions[0]), np.array(partitions[1])  # train_indices, test_indices

    def k_fold_split(self, smiles: List[str], k: int = 3, fold_min_frac: Optional[float] = None) -> List[List[int]]:
        """
        Split the dataset into k folds such that no molecule in any fold has an ECFP4 Tanimoto
        similarity greater than similarity_threshold when compared to molecules in another fold.

        Args:
            smiles: List of SMILES strings representing molecules
            k: Number of folds
            fold_min_frac: Minimum fraction of a fold (e.g., 0.2 of the entire dataset).
                If not specified (None), it defaults to 0.9 / k.

        Returns:
            List[List[int]]: List of lists, where each list contains the indices of molecules in that fold
        """
        partitions = hi_k_fold_split(
            smiles=smiles,
            similarity_threshold=self.similarity_threshold,
            fold_min_frac=fold_min_frac,
            k=k,
            coarsening_threshold=self.coarsening_threshold,
            verbose=self.verbose,
            max_mip_gap=self.max_mip_gap,
        )

        return partitions
