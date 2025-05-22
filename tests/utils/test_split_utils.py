from alinemol.utils.split_utils import (
    EmpiricalKernelMapTransformer,
    convert_to_default_feats_if_smiles,
    get_scaffold,
    pairwise_dataset_distance,
    retrieve_k_nearest_neighbors_Tanimoto,
    retrive_index,
    train_test_dataset_distance_retrieve,
    MOLECULE_DEFAULT_DISTANCE_METRIC,
)

import numpy as np
import pandas as pd


def test_featurize():
    pass


def test_compute_similarities():
    pass


def test_split_hypers():
    pass


def test_split_molecules_train_test():
    pass


def test_split_molecules_train_val_test():
    pass


def test_get_scaffold(manual_smiles_for_scaffold):
    scaffold = [get_scaffold(smiles) for smiles in manual_smiles_for_scaffold]
    for i in range(len(manual_smiles_for_scaffold)):
        assert isinstance(scaffold[i], str)
        assert len(scaffold[i]) > 0
        assert len(scaffold[i]) <= len(manual_smiles_for_scaffold[i])


def test_pairwise_dataset_distance_with_smiles():
    smiles = ["CCO", "CCN", "CCC"]
    metric = "jaccard"
    distance_matrix = pairwise_dataset_distance(smiles, metric)
    assert distance_matrix.shape == (3, 3)
    assert np.all(distance_matrix >= 0)
    assert np.all(distance_matrix <= 1)


def test_pairwise_dataset_distance_with_features():
    features = np.random.rand(3, 2048)
    metric = "euclidean"
    distance_matrix = pairwise_dataset_distance(features, metric)
    assert distance_matrix.shape == (3, 3)
    assert np.all(distance_matrix >= 0)


def test_convert_to_default_feats_if_smiles_with_smiles():
    smiles = ["CCO", "CCN", "CCC"]
    metric = "jaccard"
    features, updated_metric = convert_to_default_feats_if_smiles(smiles, metric)
    assert isinstance(features, list)
    assert len(features) == len(smiles)
    assert updated_metric == MOLECULE_DEFAULT_DISTANCE_METRIC


def test_convert_to_default_feats_if_smiles_with_features():
    features = np.random.rand(3, 2048)
    metric = "euclidean"
    updated_features, updated_metric = convert_to_default_feats_if_smiles(features, metric)
    assert np.array_equal(features, updated_features)
    assert updated_metric == metric


def test_convert_to_default_feats_if_smiles_with_mixed_input():
    mixed_input = ["CCO", np.random.rand(2048)]
    metric = "jaccard"
    features, updated_metric = convert_to_default_feats_if_smiles(mixed_input, metric)
    assert features == mixed_input
    assert updated_metric == metric


def test_empirical_kernel_map_transformer():
    X = np.random.rand(10, 2048)
    n_samples = 5
    metric = "euclidean"
    transformer = EmpiricalKernelMapTransformer(n_samples=n_samples, metric=metric, random_state=42)

    # Test transform method
    transformed_X = transformer.transform(X)
    assert transformed_X.shape == (10, n_samples)
    assert np.all(transformed_X >= 0)

    # Test __call__ method
    transformed_X_call = transformer(X)
    assert np.array_equal(transformed_X, transformed_X_call)

    # Test with different random state
    transformer_diff_state = EmpiricalKernelMapTransformer(n_samples=n_samples, metric=metric, random_state=24)
    transformed_X_diff_state = transformer_diff_state.transform(X)
    assert transformed_X_diff_state.shape == (10, n_samples)
    assert not np.array_equal(transformed_X, transformed_X_diff_state)

    # Test with different metric
    transformer_diff_metric = EmpiricalKernelMapTransformer(n_samples=n_samples, metric="cityblock", random_state=42)
    transformed_X_diff_metric = transformer_diff_metric.transform(X)
    assert transformed_X_diff_metric.shape == (10, n_samples)
    assert not np.array_equal(transformed_X, transformed_X_diff_metric)


def test_retrive_index():
    original_df = pd.DataFrame({"smiles": ["CCO", "CCN", "CCC", "CCF"], "label": [0, 1, 0, 1]})
    splitted_df = pd.DataFrame({"smiles": ["CCO", "CCC"], "label": [0, 0]})
    expected_indices = np.array([0, 2])
    indices = retrive_index(original_df, splitted_df)
    assert np.array_equal(indices, expected_indices)


def test_retrive_index_with_empty_splitted_df():
    original_df = pd.DataFrame({"smiles": ["CCO", "CCN", "CCC", "CCF"], "label": [0, 1, 0, 1]})
    splitted_df = pd.DataFrame({"smiles": [], "label": []})
    expected_indices = np.array([])
    indices = retrive_index(original_df, splitted_df)
    assert np.array_equal(indices, expected_indices)


def test_retrive_index_with_non_matching_smiles():
    original_df = pd.DataFrame({"smiles": ["CCO", "CCN", "CCC", "CCF"], "label": [0, 1, 0, 1]})
    splitted_df = pd.DataFrame({"smiles": ["CCB", "CCL"], "label": [0, 0]})
    expected_indices = np.array([])
    indices = retrive_index(original_df, splitted_df)
    assert np.array_equal(indices, expected_indices)


def test_retrive_index_with_partial_matching_smiles():
    original_df = pd.DataFrame({"smiles": ["CCO", "CCN", "CCC", "CCF"], "label": [0, 1, 0, 1]})
    splitted_df = pd.DataFrame({"smiles": ["CCO", "CCL"], "label": [0, 0]})
    expected_indices = np.array([0])
    indices = retrive_index(original_df, splitted_df)
    assert np.array_equal(indices, expected_indices)


def test_train_test_dataset_distance_retrieve():
    original_df = pd.DataFrame({"smiles": ["CCO", "CCN", "CCC", "CCF"], "label": [0, 1, 0, 1]})
    train_df = pd.DataFrame({"smiles": ["CCO", "CCC"], "label": [0, 0]})
    test_df = pd.DataFrame({"smiles": ["CCN", "CCF"], "label": [1, 1]})
    pairwise_distance = np.array(
        [[0.0, 0.5, 0.2, 0.3], [0.5, 0.0, 0.4, 0.1], [0.2, 0.4, 0.0, 0.6], [0.3, 0.1, 0.6, 0.0]]
    )
    expected_distance = np.array([[0.5, 0.3], [0.4, 0.6]])
    distance = train_test_dataset_distance_retrieve(original_df, train_df, test_df, pairwise_distance)
    assert np.array_equal(distance, expected_distance)


def test_train_test_dataset_distance_retrieve_with_empty_train_df():
    original_df = pd.DataFrame({"smiles": ["CCO", "CCN", "CCC", "CCF"], "label": [0, 1, 0, 1]})
    train_df = pd.DataFrame({"smiles": [], "label": []})
    test_df = pd.DataFrame({"smiles": ["CCN", "CCF"], "label": [1, 1]})
    pairwise_distance = np.array(
        [[0.0, 0.5, 0.2, 0.3], [0.5, 0.0, 0.4, 0.1], [0.2, 0.4, 0.0, 0.6], [0.3, 0.1, 0.6, 0.0]]
    )
    expected_distance = np.array([]).reshape(0, 2)
    distance = train_test_dataset_distance_retrieve(original_df, train_df, test_df, pairwise_distance)
    assert np.array_equal(distance, expected_distance)


def test_train_test_dataset_distance_retrieve_with_empty_test_df():
    original_df = pd.DataFrame({"smiles": ["CCO", "CCN", "CCC", "CCF"], "label": [0, 1, 0, 1]})
    train_df = pd.DataFrame({"smiles": ["CCO", "CCC"], "label": [0, 0]})
    test_df = pd.DataFrame({"smiles": [], "label": []})
    pairwise_distance = np.array(
        [[0.0, 0.5, 0.2, 0.3], [0.5, 0.0, 0.4, 0.1], [0.2, 0.4, 0.0, 0.6], [0.3, 0.1, 0.6, 0.0]]
    )
    expected_distance = np.array([]).reshape(2, 0)
    distance = train_test_dataset_distance_retrieve(original_df, train_df, test_df, pairwise_distance)
    assert np.array_equal(distance, expected_distance)


def test_train_test_dataset_distance_retrieve_with_non_matching_smiles():
    original_df = pd.DataFrame({"smiles": ["CCO", "CCN", "CCC", "CCF"], "label": [0, 1, 0, 1]})
    train_df = pd.DataFrame({"smiles": ["CCB", "CCL"], "label": [0, 0]})
    test_df = pd.DataFrame({"smiles": ["CCB", "CCL"], "label": [0, 0]})
    pairwise_distance = np.array(
        [[0.0, 0.5, 0.2, 0.3], [0.5, 0.0, 0.4, 0.1], [0.2, 0.4, 0.0, 0.6], [0.3, 0.1, 0.6, 0.0]]
    )
    expected_distance = np.array([]).reshape(0, 0)
    distance = train_test_dataset_distance_retrieve(original_df, train_df, test_df, pairwise_distance)
    assert np.array_equal(distance, expected_distance)


def test_train_test_dataset_distance_retrieve_with_partial_matching_smiles():
    original_df = pd.DataFrame({"smiles": ["CCO", "CCN", "CCC", "CCF"], "label": [0, 1, 0, 1]})
    train_df = pd.DataFrame({"smiles": ["CCO", "CCL"], "label": [0, 0]})
    test_df = pd.DataFrame({"smiles": ["CCN", "CCF"], "label": [1, 1]})
    pairwise_distance = np.array(
        [[0.0, 0.5, 0.2, 0.3], [0.5, 0.0, 0.4, 0.1], [0.2, 0.4, 0.0, 0.6], [0.3, 0.1, 0.6, 0.0]]
    )
    expected_distance = np.array([[0.5, 0.3]])
    distance = train_test_dataset_distance_retrieve(original_df, train_df, test_df, pairwise_distance)
    assert np.array_equal(distance, expected_distance)


def test_retrieve_k_nearest_neighbors():
    original_df = pd.DataFrame({"smiles": ["CCO", "CCN", "CCC", "CCF"], "label": [0, 1, 0, 1]})
    train_df = pd.DataFrame({"smiles": ["CCO", "CCC"], "label": [0, 0]})
    test_df = pd.DataFrame({"smiles": ["CCN", "CCF"], "label": [1, 1]})
    pairwise_distance = np.array(
        [[0.0, 0.5, 0.2, 0.3], [0.5, 0.0, 0.4, 0.1], [0.2, 0.4, 0.0, 0.6], [0.3, 0.1, 0.6, 0.0]]
    )
    k = 1
    expected_similarity = np.array([0.4, 0.3])
    similarity = retrieve_k_nearest_neighbors_Tanimoto(pairwise_distance, original_df, train_df, test_df, k)
    assert np.array_equal(similarity, expected_similarity)
