"""GNN model architectures for smart contract vulnerability detection.

Implements the three backbone architectures evaluated in the paper (Table VIII):
GCN (Kipf & Welling, 2017), GAT (Velickovic et al., 2018), and GraphSAGE
(Hamilton et al., 2017). All architectures use 2 layers, 128 hidden units,
ReLU activation, and dropout p=0.3 (Section V-A).
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


class GCNModel(nn.Module):
    """Two-layer GCN for node classification (Kipf & Welling, ICLR 2017).

    Args:
        in_dim: Input feature dimensionality (d=64 in the paper).
        hidden_dim: Hidden layer width (128, Section V-A).
        out_dim: Number of output classes (C=2 for binary classification).
        num_layers: Number of GCN layers (L=2, Section V-A).
        dropout: Dropout probability (p=0.3, Section V-A).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, out_dim))
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Compute node-level logits via L-layer message passing.

        Args:
            x: Node feature matrix of shape (N, d).
            edge_index: COO edge tensor of shape (2, E).

        Returns:
            Logit matrix of shape (N, C).
        """
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def get_embeddings(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Return penultimate-layer node embeddings.

        Args:
            x: Node feature matrix of shape (N, d).
            edge_index: COO edge tensor of shape (2, E).

        Returns:
            Embedding matrix of shape (N, hidden_dim).
        """
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x


class GATModel(nn.Module):
    """Two-layer GAT with multi-head attention (Velickovic et al., ICLR 2018).

    Paper (Table VIII): 4 attention heads, 128 hidden units per head.

    Args:
        in_dim: Input feature dimensionality.
        hidden_dim: Hidden units per attention head (128).
        out_dim: Number of output classes.
        num_layers: Number of GAT layers (2).
        dropout: Dropout probability (0.3).
        heads: Number of attention heads (4, Table VIII).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
            )
        self.convs.append(
            GATConv(
                hidden_dim * heads, out_dim, heads=1, concat=False, dropout=dropout
            )
        )
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Compute node-level logits via multi-head attention layers.

        Args:
            x: Node feature matrix of shape (N, d).
            edge_index: COO edge tensor of shape (2, E).

        Returns:
            Logit matrix of shape (N, C).
        """
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def get_embeddings(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Return penultimate-layer node embeddings.

        Args:
            x: Node feature matrix of shape (N, d).
            edge_index: COO edge tensor of shape (2, E).

        Returns:
            Embedding matrix of shape (N, hidden_dim * heads).
        """
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
        return x


class GraphSAGEModel(nn.Module):
    """GraphSAGE with mean aggregator (Hamilton et al., NeurIPS 2017).

    Paper (Table VIII): 128 hidden units, mean aggregator.

    Args:
        in_dim: Input feature dimensionality.
        hidden_dim: Hidden layer width (128).
        out_dim: Number of output classes.
        num_layers: Number of SAGE layers (2).
        dropout: Dropout probability (0.3).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, out_dim))
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Compute node-level logits via neighbourhood sampling and aggregation.

        Args:
            x: Node feature matrix of shape (N, d).
            edge_index: COO edge tensor of shape (2, E).

        Returns:
            Logit matrix of shape (N, C).
        """
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def get_embeddings(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Return penultimate-layer node embeddings.

        Args:
            x: Node feature matrix of shape (N, d).
            edge_index: COO edge tensor of shape (2, E).

        Returns:
            Embedding matrix of shape (N, hidden_dim).
        """
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """Instantiate a GNN model from the configuration dictionary.

    Args:
        cfg: Experiment configuration containing model.arch, model.hidden_dim, etc.

    Returns:
        An initialised GNN model matching the specified architecture.
    """
    arch: str = cfg["model"]["arch"]
    in_dim: int = cfg["data"]["feature_dim"]
    hidden_dim: int = cfg["model"]["hidden_dim"]
    out_dim: int = cfg["data"]["num_classes"]
    num_layers: int = cfg["model"]["num_layers"]
    dropout: float = cfg["model"]["dropout"]

    if arch == "GCN":
        return GCNModel(in_dim, hidden_dim, out_dim, num_layers, dropout)
    elif arch == "GAT":
        return GATModel(in_dim, hidden_dim, out_dim, num_layers, dropout)
    elif arch == "GraphSAGE":
        return GraphSAGEModel(in_dim, hidden_dim, out_dim, num_layers, dropout)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
