"""Federated client: local training and evaluation.

Each client owns one chain's local graph and trains for E local epochs
per round (Section III-C).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def client_train(
    model: torch.nn.Module,
    data: Any,
    cfg: Dict[str, Any],
    mask: Optional[Tensor] = None,
) -> Tuple[Dict[str, Tensor], int, float]:
    """Train the local model for E local epochs on the chain's training nodes.

    Args:
        model: Local copy of the global model.
        data: PyG Data object for the local chain.
        cfg: Experiment configuration.
        mask: Optional override for training node mask.

    Returns:
        Tuple of (state_dict, num_train_samples, final_loss).
    """
    device = cfg["device"]
    model = model.to(device)
    data = data.to(device)
    model.train()

    lr = cfg["federated"]["lr"]
    wd = cfg["federated"]["weight_decay"]
    epochs = cfg["federated"]["local_epochs"]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    if mask is None:
        mask = data.train_mask

    y_train = data.y[mask]
    num_classes = cfg["data"]["num_classes"]
    class_counts = torch.zeros(num_classes, device=device)
    for c in range(num_classes):
        class_counts[c] = (y_train == c).sum().float()

    weight = None
    if (class_counts > 0).sum() > 1:
        total = class_counts.sum()
        weight = total / (num_classes * class_counts.clamp(min=1))

    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[mask], data.y[mask], weight=weight)
        loss.backward()
        optimizer.step()

    return model.state_dict(), mask.sum().item(), loss.item()


def client_evaluate(
    model: torch.nn.Module,
    data: Any,
    mask: Tensor,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the model on nodes selected by mask.

    Args:
        model: GNN model to evaluate.
        data: PyG Data object.
        mask: Boolean mask selecting evaluation nodes.
        device: Computation device.

    Returns:
        Tuple of (y_true, y_pred, y_prob) as numpy arrays.
    """
    model = model.to(device)
    data = data.to(device)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        prob = F.softmax(out, dim=1)
        pred = out.argmax(dim=1)
    y_true = data.y[mask].cpu().numpy()
    y_pred = pred[mask].cpu().numpy()
    y_prob_full = prob[mask].cpu().numpy()
    num_classes = prob.shape[1]
    if num_classes == 2:
        y_prob = y_prob_full[:, 1]
    else:
        y_prob = y_prob_full
    return y_true, y_pred, y_prob
