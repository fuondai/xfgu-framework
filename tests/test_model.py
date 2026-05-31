"""Unit tests for GNN model architectures."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from utils.seed import set_seed
from models.gnn import build_model, GCNModel, GATModel, GraphSAGEModel


@pytest.fixture(scope="module")
def model_cfg():
    set_seed(42)
    return {
        "seed": 42,
        "device": "cpu",
        "data": {"feature_dim": 64, "num_classes": 2},
        "model": {"arch": "GCN", "hidden_dim": 32, "num_layers": 2, "dropout": 0.3},
    }


@pytest.fixture
def minimal_graph():
    """Create a minimal graph for forward pass testing."""
    x = torch.randn(20, 64)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 1, 2, 3, 4, 5],
         [1, 2, 3, 4, 5, 0, 1, 2, 3, 4]],
        dtype=torch.long,
    )
    return x, edge_index


class TestGCNModel:
    def test_forward_shape(self, minimal_graph):
        x, edge_index = minimal_graph
        model = GCNModel(in_dim=64, hidden_dim=128, out_dim=2,
                         dropout=0.3, num_layers=2)
        out = model(x, edge_index)
        assert out.shape == (20, 2)

    def test_parameter_count(self):
        model = GCNModel(in_dim=64, hidden_dim=128, out_dim=2,
                         dropout=0.3, num_layers=2)
        n_params = sum(p.numel() for p in model.parameters())
        # GCNConv(64,128) + GCNConv(128,2) = 8320 + 258 = 8578
        assert n_params == 8578, f"Expected 8578 params, got {n_params}"


class TestGATModel:
    def test_forward_shape(self, minimal_graph):
        x, edge_index = minimal_graph
        model = GATModel(in_dim=64, hidden_dim=128, out_dim=2,
                         heads=4, dropout=0.3, num_layers=2)
        out = model(x, edge_index)
        assert out.shape == (20, 2)


class TestGraphSAGEModel:
    def test_forward_shape(self, minimal_graph):
        x, edge_index = minimal_graph
        model = GraphSAGEModel(in_dim=64, hidden_dim=128, out_dim=2,
                                dropout=0.3, num_layers=2)
        out = model(x, edge_index)
        assert out.shape == (20, 2)


class TestBuildModel:
    def test_build_gcn(self, model_cfg):
        model = build_model(model_cfg)
        assert isinstance(model, GCNModel)

    def test_build_gat(self, model_cfg):
        cfg = dict(model_cfg)
        cfg["model"] = dict(model_cfg["model"])
        cfg["model"]["arch"] = "GAT"
        model = build_model(cfg)
        assert isinstance(model, GATModel)

    def test_build_graphsage(self, model_cfg):
        cfg = dict(model_cfg)
        cfg["model"] = dict(model_cfg["model"])
        cfg["model"]["arch"] = "GraphSAGE"
        model = build_model(cfg)
        assert isinstance(model, GraphSAGEModel)

    def test_invalid_arch(self, model_cfg):
        cfg = dict(model_cfg)
        cfg["model"] = dict(model_cfg["model"])
        cfg["model"]["arch"] = "InvalidArch"
        with pytest.raises(ValueError):
            build_model(cfg)
