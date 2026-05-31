"""Full Retrain baseline (Section V-A).

Reference:
    Baseline 1 in Table II.  Trains a fresh model from scratch on
    all retain data for T = 80 federated rounds.  Serves as the
    gold-standard reference for unlearning quality.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import torch
from torch import Tensor

from federated.client import client_train
from federated.server import fedavg_aggregate


def full_retrain(
    model_cls: Any,
    chains: Dict[str, Any],
    forget_sets: Dict[str, List[int]],
    cfg: Dict[str, Any],
) -> torch.nn.Module:
    """Retrain from scratch excluding forget nodes.

    Args:
        model_cls: Unused (kept for interface compatibility).
        chains: Mapping from chain name to PyG Data objects.
        forget_sets: Mapping from chain name to forget node indices.
        cfg: Experiment configuration.

    Returns:
        Freshly trained global model.
    """
    from models.gnn import build_model
    fresh_model = build_model(cfg)
    device = cfg["device"]
    num_rounds = cfg["federated"]["num_rounds"]

    modified_chains = {}
    for chain_name, data in chains.items():
        forget_nodes = forget_sets.get(chain_name, [])
        new_data = data.clone()
        if len(forget_nodes) > 0:
            forget_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            forget_mask[forget_nodes] = True
            new_data.train_mask = data.train_mask & ~forget_mask
        modified_chains[chain_name] = new_data

    for rnd in range(num_rounds):
        client_states = []
        client_weights = []

        for name, data in modified_chains.items():
            local_model = copy.deepcopy(fresh_model)
            state, n_samples, loss = client_train(local_model, data, cfg)
            client_states.append(state)
            client_weights.append(n_samples)

        fresh_model = fedavg_aggregate(fresh_model, client_states, client_weights)

    return fresh_model
