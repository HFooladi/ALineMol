from alinemol.utils.split_utils import *
import numpy as np
import pytest

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

def test_pairwise_dataset_distance_with_invalid_metric():
    smiles = ["CCO", "CCN", "CCC"]
    metric = "invalid_metric"
    try:
        pairwise_dataset_distance(smiles, metric)
    except ValueError:
        assert True
    else:
        assert False

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
    try:
        convert_to_default_feats_if_smiles(mixed_input, metric)
    except AssertionError:
        assert True
    else:
        assert False
    
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









