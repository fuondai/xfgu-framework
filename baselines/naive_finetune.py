"""Naive Fine-Tune baseline (Section V-A).

Reference:
    Baseline 3 in Table II.  Fine-tunes the global model on retain nodes
    for T_f = 15 steps (Section V-A) without any reversal or DP mechanism.
    This isolates the contribution of XFGU's reverse-gradient phase.

Configuration keys (under ``naive_finetune``):
    steps : int   – number of SGD steps (default 15, paper Section V-A)
    lr    : float – learning rate (defaults to federated.lr = 5e-3)
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor


def naive_finetune_unlearn(
    global_model: torch.nn.Module,
    chains: Dict[str, Any],
    forget_sets: Dict[str, List[int]],
    cfg: Dict[str, Any],
) -> torch.nn.Module:
    """Fine-tune the global model on retain nodes only.

    Args:
        global_model: Trained federated GNN model.
        chains: Mapping from chain name to PyG Data objects.
        forget_sets: Mapping from chain name to forget node indices.
        cfg: Experiment configuration dictionary.

    Returns:
        Fine-tuned model with forget nodes excluded from training.
    """
    device: str = cfg["device"]
    nft_cfg = cfg.get("naive_finetune", {})
    steps: int = nft_cfg.get("steps", 15)
    lr: float = nft_cfg.get("lr", cfg["federated"]["lr"])
    num_classes: int = cfg["data"]["num_classes"]

    model = copy.deepcopy(global_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    model.train()
    for _step in range(steps):
        for chain_name, data in chains.items():
            data = data.to(device)
            forget_nodes = forget_sets.get(chain_name, [])
            retain_mask = data.train_mask.clone()
            if len(forget_nodes) > 0:
                f_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                f_mask[forget_nodes] = True
                retain_mask = retain_mask & ~f_mask

            if retain_mask.sum() == 0:
                continue

            y_retain = data.y[retain_mask]
            class_counts = torch.zeros(num_classes, device=device)
            for c in range(num_classes):
                class_counts[c] = (y_retain == c).sum().float()
            weight = None
            if (class_counts > 0).sum() > 1:
                total = class_counts.sum()
                weight = total / (num_classes * class_counts.clamp(min=1))

            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[retain_mask], data.y[retain_mask], weight=weight)
            loss.backward()
            optimizer.step()

    return model
