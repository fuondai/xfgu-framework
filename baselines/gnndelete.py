"""GNNDelete baseline (Cheng et al., 2023).

Reference:
    Cheng, M. et al. "GNNDelete: A General Strategy for Unlearning in
    Graph Neural Networks." ICLR 2023.

    Constrained fine-tuning enforcing *deleted edge consistency* (DEC) and
    *neighbourhood influence* (NI).  Applied per client on local graphs
    and aggregated via FedAvg (Section V-A).

Configuration keys (under ``gnndelete``):
    lr            : float – learning rate (default 1e-3)
    steps         : int   – fine-tuning steps (default 50)
    lambda_dec    : float – weight for DEC loss (default 1.0)
    lambda_ni     : float – weight for NI loss (default 1.0)
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from federated.server import fedavg_aggregate
from graphs.influence_zone import get_forget_retain_masks


def _deleted_edge_consistency_loss(
    model: torch.nn.Module,
    data: Any,
    forget_edges: Tensor,
    device: torch.device,
) -> Tensor:
    """Deleted Edge Consistency: embeddings of endpoints of deleted edges
    should be no more similar than random pairs (Cheng et al., 2023, Def. 1).

    We approximate this by pushing cosine similarity of forget-edge endpoints
    toward zero.
    """
    if forget_edges.size(1) == 0:
        return torch.tensor(0.0, device=device)

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    model.train()
    out = model(data.x, data.edge_index)

    src_emb = out[forget_edges[0]]
    dst_emb = out[forget_edges[1]]
    cos_sim = F.cosine_similarity(src_emb, dst_emb, dim=1)
    return cos_sim.pow(2).mean()


def _neighbourhood_influence_loss(
    model: torch.nn.Module,
    data: Any,
    forget_nodes_set: set,
    retain_train_mask: Tensor,
    original_out: Tensor,
    device: torch.device,
) -> Tensor:
    """Neighbourhood Influence: predictions on retain neighbours of forget
    nodes should remain close to the original model's predictions
    (Cheng et al., 2023, Def. 2).
    """
    out = model(data.x, data.edge_index)
    if retain_train_mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    diff = (out[retain_train_mask] - original_out[retain_train_mask]).pow(2)
    return diff.mean()


def _get_forget_edges(
    edge_index: Tensor, forget_nodes: List[int]
) -> Tensor:
    """Return columns of edge_index where at least one endpoint is a forget node."""
    forget_set = set(forget_nodes)
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    mask = torch.tensor(
        [s in forget_set or d in forget_set for s, d in zip(src, dst)],
        dtype=torch.bool,
    )
    return edge_index[:, mask]


def gnndelete_unlearn(
    global_model: torch.nn.Module,
    chains: Dict[str, Any],
    forget_sets: Dict[str, List[int]],
    cfg: Dict[str, Any],
) -> torch.nn.Module:
    """Execute GNNDelete unlearning per client with FedAvg aggregation.

    Args:
        global_model: Trained federated GNN model.
        chains: Mapping from chain name to PyG Data objects.
        forget_sets: Mapping from chain name to forget node indices.
        cfg: Experiment configuration.

    Returns:
        Unlearned global model.
    """
    device = cfg["device"]
    gd_cfg = cfg.get("gnndelete", {})
    lr: float = gd_cfg.get("lr", 1e-3)
    steps: int = gd_cfg.get("steps", 50)
    lambda_dec: float = gd_cfg.get("lambda_dec", 1.0)
    lambda_ni: float = gd_cfg.get("lambda_ni", 1.0)
    num_classes: int = cfg["data"]["num_classes"]

    unlearned_states: list[Dict[str, Tensor]] = []
    client_weights: list[int] = []

    for chain_name, data in chains.items():
        forget_nodes = forget_sets.get(chain_name, [])
        local_model = copy.deepcopy(global_model).to(device)

        if len(forget_nodes) == 0:
            unlearned_states.append(local_model.state_dict())
            client_weights.append(int(data.train_mask.sum().item()))
            continue

        data_dev = data.to(device)
        _, forget_train_mask, retain_train_mask = get_forget_retain_masks(data, forget_nodes)
        forget_edges = _get_forget_edges(data.edge_index, forget_nodes).to(device)

        # Cache original predictions for NI loss
        local_model.eval()
        with torch.no_grad():
            original_out = local_model(data_dev.x, data_dev.edge_index).detach()

        optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
        local_model.train()

        for _ in range(steps):
            optimizer.zero_grad()

            out = local_model(data_dev.x, data_dev.edge_index)

            # Retain-set classification loss
            if retain_train_mask.sum() > 0:
                y_retain = data_dev.y[retain_train_mask]
                class_counts = torch.zeros(num_classes, device=device)
                for c in range(num_classes):
                    class_counts[c] = (y_retain == c).sum().float()
                weight = None
                if (class_counts > 0).sum() > 1:
                    total = class_counts.sum()
                    weight = total / (num_classes * class_counts.clamp(min=1))
                ce_loss = F.cross_entropy(
                    out[retain_train_mask], data_dev.y[retain_train_mask], weight=weight
                )
            else:
                ce_loss = torch.tensor(0.0, device=device)

            # DEC: push forget-edge endpoint similarity to zero
            dec_loss = _deleted_edge_consistency_loss(
                local_model, data_dev, forget_edges, device
            )

            # NI: keep retain-neighbour predictions close to original
            ni_loss = _neighbourhood_influence_loss(
                local_model, data_dev, set(forget_nodes),
                retain_train_mask, original_out, device,
            )

            loss = ce_loss + lambda_dec * dec_loss + lambda_ni * ni_loss
            loss.backward()
            optimizer.step()

        unlearned_states.append(local_model.state_dict())
        client_weights.append(max(1, int(retain_train_mask.sum().item())))

    unlearned_global = copy.deepcopy(global_model)
    unlearned_global = fedavg_aggregate(unlearned_global, unlearned_states, client_weights)
    return unlearned_global
