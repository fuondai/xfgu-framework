"""Influence-zone discovery via L-hop BFS (Section IV-A, Eq. 2)."""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.utils import k_hop_subgraph


def get_influence_zone(
    edge_index: Tensor,
    forget_nodes: List[int],
    num_hops: int,
    num_nodes: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return the L-hop influence zone around forget nodes (Eq. 2).

    Args:
        edge_index: Full graph edge index.
        forget_nodes: List of forget node indices.
        num_hops: Number of BFS hops L.
        num_nodes: Total number of nodes.

    Returns:
        Tuple of (subset_nodes, sub_edge_index, mapping).
    """
    if len(forget_nodes) == 0:
        return torch.tensor([], dtype=torch.long), edge_index, torch.arange(edge_index.max() + 1)

    forget_tensor = torch.tensor(forget_nodes, dtype=torch.long)
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        forget_tensor, num_hops, edge_index, relabel_nodes=False, num_nodes=num_nodes
    )
    return subset, sub_edge_index, mapping


def get_forget_retain_masks(
    data: Any, forget_nodes: List[int]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute boolean masks for forget and retain training nodes.

    Args:
        data: PyG Data object with train_mask attribute.
        forget_nodes: List of forget node indices.

    Returns:
        Tuple of (forget_mask, forget_train_mask, retain_train_mask).
    """
    n = data.num_nodes
    forget_mask = torch.zeros(n, dtype=torch.bool)
    forget_mask[forget_nodes] = True

    retain_train_mask = data.train_mask & ~forget_mask
    forget_train_mask = data.train_mask & forget_mask

    return forget_mask, forget_train_mask, retain_train_mask


def select_forget_nodes(
    data: Any, forget_ratio: float, seed: int = 42
) -> List[int]:
    """Randomly select a fraction of training nodes to forget.

    Args:
        data: PyG Data object with train_mask.
        forget_ratio: Fraction of training nodes to forget (paper: 0.05).
        seed: Random seed.

    Returns:
        List of forget node indices.
    """
    rng = np.random.RandomState(seed)
    train_indices = data.train_mask.nonzero(as_tuple=True)[0].numpy()
    n_forget = max(1, int(len(train_indices) * forget_ratio))
    forget_indices = rng.choice(train_indices, size=n_forget, replace=False)
    return forget_indices.tolist()
