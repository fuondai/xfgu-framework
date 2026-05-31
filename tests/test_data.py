"""Unit tests for the data pipeline.

One test per preprocessing step, exercised on the curated corpus.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from utils.seed import set_seed
from data.contract_graph_builder import (
    extract_solidity_features,
    detect_vulnerability_type,
    build_graph_from_contracts,
)
from data.generate_sc_graph import add_cross_chain_edges, split_data
from data.dataset import parse_curated_contracts

CURATED_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "datasets", "smartbugs-curated", "dataset"
)
REENTRANCY_DIR = os.path.join(CURATED_ROOT, "reentrancy")


def _first_sol_file(directory):
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".sol"):
            return os.path.join(directory, fname)
    raise FileNotFoundError(f"No .sol file found in {directory}")


@pytest.fixture(scope="module")
def curated_features():
    """Feature/label matrix from the curated corpus."""
    set_seed(42)
    features, labels = parse_curated_contracts(CURATED_ROOT, feature_dim=64)
    return features, labels


def _build_small_graph(features, labels, n=40, seed=42):
    rng = np.random.RandomState(seed)
    n = min(n, len(labels))
    idx = rng.choice(len(labels), size=n, replace=False)
    return build_graph_from_contracts(features[idx], labels[idx],
                                      threshold=0.6, max_neighbors=3)


class TestExtractSolidityFeatures:
    def test_feature_dim(self):
        with open(_first_sol_file(REENTRANCY_DIR), "r", errors="ignore") as f:
            source = f.read()
        feat = extract_solidity_features(source, feature_dim=64)
        assert feat.shape == (64,)

    def test_features_finite(self):
        with open(_first_sol_file(REENTRANCY_DIR), "r", errors="ignore") as f:
            source = f.read()
        feat = extract_solidity_features(source, feature_dim=64)
        assert np.all(np.isfinite(feat))

    def test_reentrancy_label(self):
        with open(_first_sol_file(REENTRANCY_DIR), "r", errors="ignore") as f:
            source = f.read()
        label = detect_vulnerability_type(source, directory_name="reentrancy")
        assert label == 1


class TestSplitData:
    def test_ratios(self, curated_features):
        features, labels = curated_features
        data = _build_small_graph(features, labels, n=120)
        data = split_data(data, train_ratio=0.6, val_ratio=0.2, seed=42)
        n = data.num_nodes
        assert abs(data.train_mask.sum().item() / n - 0.6) < 0.15
        assert abs(data.val_mask.sum().item() / n - 0.2) < 0.15
        assert abs(data.test_mask.sum().item() / n - 0.2) < 0.15

    def test_no_overlap(self, curated_features):
        features, labels = curated_features
        data = _build_small_graph(features, labels, n=120)
        data = split_data(data, 0.6, 0.2, seed=42)
        assert (data.train_mask & data.val_mask).sum() == 0
        assert (data.train_mask & data.test_mask).sum() == 0
        assert (data.val_mask & data.test_mask).sum() == 0

    def test_covers_all(self, curated_features):
        features, labels = curated_features
        data = _build_small_graph(features, labels, n=120)
        data = split_data(data, 0.6, 0.2, seed=42)
        total = data.train_mask.sum() + data.val_mask.sum() + data.test_mask.sum()
        assert total == data.num_nodes


class TestCrossChainEdges:
    def _two_chains(self, curated_features):
        features, labels = curated_features
        chains = {}
        for i, name in enumerate(["A", "B"]):
            data = _build_small_graph(features, labels, n=40, seed=42 + i)
            data = split_data(data, 0.6, 0.2, seed=42 + i)
            chains[name] = data
        return chains

    def test_adds_edges(self, curated_features):
        chains = self._two_chains(curated_features)
        orig_edges_a = chains["A"].edge_index.size(1)
        orig_edges_b = chains["B"].edge_index.size(1)
        add_cross_chain_edges(chains, cross_chain_ratio=0.1, seed=42)
        assert chains["A"].edge_index.size(1) > orig_edges_a
        assert chains["B"].edge_index.size(1) > orig_edges_b

    def test_cross_chain_attribute(self, curated_features):
        chains = self._two_chains(curated_features)
        add_cross_chain_edges(chains, cross_chain_ratio=0.1, seed=42)
        assert hasattr(chains["A"], "cross_chain_edges")
        assert len(chains["A"].cross_chain_edges) > 0
