"""SISA-Graph baseline (Bourtoule et al., 2021).

Reference:
    Baseline 2 in Table II.  Partitions each chain's training nodes into
    K = 5 shards and trains an independent model per shard using the FULL
    graph for message passing but computing loss only on the shard's nodes.
    On an unlearning request, only affected shards are retrained.
    Per-chain models are aggregated by weighted averaging of shard models,
    and chains are combined via FedAvg.

Configuration keys (under ``sisa``):
    num_shards     : int   – number of shards K (default 5)
    retrain_epochs : int   – epochs to retrain affected shards (default 30)
    lr             : float – learning rate (default 5e-3)
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from models.gnn import build_model
from federated.client import client_train
from federated.server import fedavg_aggregate


def partition_into_shards(data, num_shards, seed=42):
    """Randomly partition training nodes into K shards."""
    rng = np.random.RandomState(seed)
    train_idx = data.train_mask.nonzero(as_tuple=True)[0].numpy()
    rng.shuffle(train_idx)
    shards = np.array_split(train_idx, num_shards)
    return [s.tolist() for s in shards]


def find_shard_containing(shards, forget_nodes):
    """Identify which shards contain forget nodes."""
    forget_set = set(forget_nodes)
    affected = []
    for i, shard in enumerate(shards):
        if forget_set & set(shard):
            affected.append(i)
    return affected


def train_shard_model_federated(model, data, shard_indices, cfg, exclude_nodes=None):
    """Train a shard model on the full graph with loss computed only on shard nodes.
    
    Uses FedAvg-style training rounds, scaling total epochs inversely with
    shard size to match the paper's 30-epoch baseline at the target shard
    size (~375 nodes/shard for ETH chain).
    """
    device = cfg["device"]
    lr = cfg["sisa"]["lr"]
    num_classes = cfg["data"]["num_classes"]

    model = model.to(device)
    data = data.to(device)

    # Build training mask for this shard
    mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    indices = torch.tensor(shard_indices, dtype=torch.long, device=device)
    mask[indices] = True

    if exclude_nodes is not None and len(exclude_nodes) > 0:
        exclude_t = torch.tensor(exclude_nodes, dtype=torch.long, device=device)
        for e in exclude_t:
            mask[e] = False

    n_samples = int(mask.sum().item())
    if n_samples == 0:
        return model

    # Scale epochs: ensure total node-level updates matches paper (~11k for 375 nodes)
    target_total_updates = cfg["sisa"]["retrain_epochs"] * 375
    total_epochs = max(cfg["sisa"]["retrain_epochs"],
                       int(target_total_updates / max(n_samples, 1)))
    total_epochs = min(total_epochs, 500)

    y_train = data.y[mask]
    class_counts = torch.zeros(num_classes, device=device)
    for c in range(num_classes):
        class_counts[c] = (y_train == c).sum().float()

    weight = None
    if (class_counts > 0).sum() > 1:
        total = class_counts.sum()
        weight = total / (num_classes * class_counts.clamp(min=1))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    model.train()
    for _ in range(total_epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[mask], data.y[mask], weight=weight)
        loss.backward()
        optimizer.step()

    return model


def sisa_unlearn(chains, forget_sets, cfg):
    num_shards = cfg["sisa"]["num_shards"]

    chain_models = {}
    chain_weights = {}

    for chain_name, data in chains.items():
        forget_nodes = forget_sets.get(chain_name, [])
        shards = partition_into_shards(data, num_shards, seed=cfg["seed"])
        affected_shard_ids = find_shard_containing(shards, forget_nodes)

        shard_states = []
        shard_weights = []

        for sid, shard in enumerate(shards):
            if len(shard) == 0:
                shard_weights.append(0)
                continue
            m = build_model(cfg)
            if sid in affected_shard_ids:
                m = train_shard_model_federated(m, data, shard, cfg, exclude_nodes=forget_nodes)
            else:
                m = train_shard_model_federated(m, data, shard, cfg)
            shard_states.append(m.state_dict())
            shard_weights.append(len(shard))

        if len(shard_states) == 0:
            chain_state = build_model(cfg).state_dict()
        else:
            chain_state = {}
            total_w = sum(shard_weights)
            for key in shard_states[0]:
                chain_state[key] = torch.zeros_like(shard_states[0][key], dtype=torch.float32)
                for state, w in zip(shard_states, shard_weights):
                    if w > 0:
                        chain_state[key] += state[key].float() * (w / total_w)

        n_retain = data.train_mask.sum().item() - len(forget_nodes)
        chain_models[chain_name] = chain_state
        chain_weights[chain_name] = max(1, n_retain)

    final_model = build_model(cfg)
    combined_state = final_model.state_dict()
    total_w = sum(chain_weights.values())
    if total_w == 0:
        return final_model
    for key in combined_state:
        combined_state[key] = torch.zeros_like(combined_state[key], dtype=torch.float32)
        for cn in chains:
            w = chain_weights[cn] / total_w
            combined_state[key] += chain_models[cn][key] * w

    final_model.load_state_dict(combined_state)
    return final_model
