"""Integration test for the XFGU pipeline on a small subset.

Runs the full data pipeline (curated SmartBugs corpus -> cross-chain
graphs -> federated training -> XFGU unlearning -> evaluation) end-to-end on a
small two-chain partition.

Run with: python -m pytest tests/test_integration.py -v
"""
import sys
import os
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from utils.seed import set_seed
from data.generate_sc_graph import build_cross_chain_data
from models.gnn import build_model
from federated.server import federated_train
from graphs.influence_zone import select_forget_nodes
from unlearning.xfgu import xfgu_unlearn, calibrate_dp_noise
from evaluation.evaluate import evaluate_model


@pytest.fixture(scope="module")
def cfg():
    return {
        "seed": 42,
        "device": "cpu",
        "data": {
            "archive_csv": "smartbug-scrawid-irfuzz.csv",
            "num_chains": 2,
            "chain_names": ["ETH", "BSC"],
            "chain_sizes": [40, 30],
            "chain_vuln_counts": [16, 11],
            "feature_dim": 64,
            "num_classes": 2,
            "cross_chain_ratio": 0.05,
            "cross_chain_threshold": 0.70,
            "edge_similarity_threshold": 0.60,
            "max_neighbors": 3,
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
        },
        "model": {"arch": "GCN", "hidden_dim": 32, "num_layers": 2, "dropout": 0.3},
        "federated": {
            "num_rounds": 3,
            "local_epochs": 2,
            "lr": 0.01,
            "weight_decay": 0.0005,
            "aggregation": "fedavg",
        },
        "unlearning": {
            "forget_ratio": 0.1,
            "l_hop": 2,
            "reverse_lr": 0.01,
            "reverse_steps": 2,
            "finetune_lr": 0.01,
            "finetune_steps": 3,
            "dp_epsilon": 2.0,
            "dp_delta": 1.0e-5,
            "clip_norm": 1.0,
        },
        "evaluation": {"mia_shadow_models": 1, "mia_epochs": 3, "mia_lr": 0.001},
    }


@pytest.fixture(scope="module")
def chains(cfg):
    set_seed(cfg["seed"])
    built = build_cross_chain_data(cfg)
    requested = dict(zip(cfg["data"]["chain_names"], cfg["data"]["chain_sizes"]))
    for name, data in built.items():
        if data.num_nodes < requested[name] or len(data.y.unique()) < 2:
            pytest.skip(
                "corpus insufficient for the integration subset; "
                "place a balanced SmartBugs CSV under archive/ to enable this test"
            )
    return built


@pytest.fixture(scope="module")
def trained_model(cfg, chains):
    set_seed(cfg["seed"])
    model = build_model(cfg)
    trained, history = federated_train(model, chains, cfg)
    return trained, history


@pytest.fixture(scope="module")
def forget_sets(cfg, chains):
    return {
        name: select_forget_nodes(data, cfg["unlearning"]["forget_ratio"], seed=42)
        for name, data in chains.items()
    }


class TestRealDataPipeline:
    def test_chain_count(self, cfg, chains):
        assert len(chains) == cfg["data"]["num_chains"]

    def test_feature_dim(self, cfg, chains):
        for data in chains.values():
            assert data.x.shape[1] == cfg["data"]["feature_dim"]

    def test_masks_partition(self, chains):
        for name, data in chains.items():
            total = data.train_mask.sum() + data.val_mask.sum() + data.test_mask.sum()
            assert total == data.num_nodes, f"{name}: masks must cover all nodes"

    def test_binary_labels(self, chains):
        for data in chains.values():
            assert set(data.y.unique().tolist()).issubset({0, 1})


class TestTrainingCompletes:
    def test_history_consistent(self, trained_model):
        model, history = trained_model
        assert model is not None
        assert len(history["rounds"]) == len(history["train_loss"])


class TestUnlearning:
    def test_xfgu_unlearn(self, cfg, chains, trained_model, forget_sets):
        model, _ = trained_model
        unlearned, zone_sizes, dp_sigma = xfgu_unlearn(
            copy.deepcopy(model), chains, forget_sets, cfg
        )
        assert unlearned is not None
        assert dp_sigma > 0
        assert len(zone_sizes) == len(chains)

    def test_dp_noise_calibration(self):
        sigma = calibrate_dp_noise(epsilon=2.0, delta=1e-5, clip_norm=1.0, num_steps=10)
        assert sigma > 0
        sigma_high_eps = calibrate_dp_noise(epsilon=10.0, delta=1e-5, clip_norm=1.0, num_steps=10)
        assert sigma_high_eps < sigma


class TestEvaluation:
    def test_evaluate_model_keys(self, cfg, chains, trained_model):
        model, _ = trained_model
        results = evaluate_model(model, chains, cfg)
        assert "overall" in results
        for name in chains:
            assert name in results
        for key in ("f1", "accuracy", "auc"):
            assert key in results["overall"]
