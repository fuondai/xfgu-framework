"""Federated server: FedAvg aggregation and training orchestration.

Implements the FedAvg protocol described in Section III-C, Eq. 1.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def fedavg_aggregate(
    global_model: torch.nn.Module,
    client_states: List[Dict[str, Tensor]],
    client_weights: List[int],
) -> torch.nn.Module:
    """Weighted FedAvg aggregation (Eq. 1).

    Args:
        global_model: Global model whose state dict is overwritten.
        client_states: List of client state dicts.
        client_weights: Per-client sample counts for weighting.

    Returns:
        Global model with aggregated parameters.
    """
    total_weight = sum(client_weights)
    global_state = global_model.state_dict()

    for key in global_state:
        global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
        for state, w in zip(client_states, client_weights):
            global_state[key] += state[key].float() * (w / total_weight)

    global_model.load_state_dict(global_state)
    return global_model


def federated_train(
    global_model: torch.nn.Module,
    chains: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Run T rounds of federated training (Section III-C, Algorithm 1 lines 1-6).

    Args:
        global_model: Initial global GNN model.
        chains: Mapping from chain name to PyG Data objects.
        cfg: Experiment configuration.

    Returns:
        Tuple of (trained global model, training history dict).
    """
    from federated.client import client_train, client_evaluate
    from utils.metrics import compute_classification_metrics

    num_rounds = cfg["federated"]["num_rounds"]
    device = cfg["device"]
    history = {"rounds": [], "train_loss": [], "val_metrics": {}}

    for name in chains:
        history["val_metrics"][name] = []

    for rnd in range(num_rounds):
        client_states = []
        client_weights = []
        losses = []

        for name, data in chains.items():
            local_model = copy.deepcopy(global_model)
            state, n_samples, loss = client_train(local_model, data, cfg)
            client_states.append(state)
            client_weights.append(n_samples)
            losses.append(loss)

        global_model = fedavg_aggregate(global_model, client_states, client_weights)

        avg_loss = sum(losses) / len(losses)
        history["rounds"].append(rnd)
        history["train_loss"].append(avg_loss)

        for name, data in chains.items():
            y_true, y_pred, y_prob = client_evaluate(global_model, data, data.val_mask, device)
            m = compute_classification_metrics(y_true, y_pred, y_prob)
            history["val_metrics"][name].append(m)

        if (rnd + 1) % 10 == 0 or rnd == 0:
            val_f1s = []
            for name in chains:
                val_f1s.append(history["val_metrics"][name][-1]["f1"])
            avg_f1 = sum(val_f1s) / len(val_f1s)
            logger.info(
                "Round %d/%d | Loss: %.4f | Avg Val F1: %.4f",
                rnd + 1, num_rounds, avg_loss, avg_f1,
            )

    return global_model, history
