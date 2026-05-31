"""GraphEraser baseline (Chen et al., 2022).

Reference:
    Baseline 4 in Table II.  Louvain community partitioning applied
    independently on each chain's local graph; affected partitions are
    retrained locally, and per-chain outputs are aggregated via FedAvg
    (Section V-A).

Configuration keys (under ``graph_eraser``):
    num_partitions  : int   – number of Louvain partitions (default 5)
    retrain_epochs  : int   – epochs to retrain affected partitions
    lr              : float – learning rate
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from federated.server import fedavg_aggregate
from models.gnn import build_model


def community_partition(data, num_partitions, seed=42):
    import networkx as nx
    edge_index = data.edge_index.cpu().numpy()
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    for i in range(edge_index.shape[1]):
        s, d = int(edge_index[0, i]), int(edge_index[1, i])
        if s != d:
            G.add_edge(s, d)

    try:
        communities = nx.community.louvain_communities(G, seed=seed)
    except Exception:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(data.num_nodes)
        splits = np.array_split(indices, num_partitions)
        return [s.tolist() for s in splits]

    partitions = [list(c) for c in communities]

    while len(partitions) > num_partitions:
        partitions.sort(key=len)
        merged = partitions[0] + partitions[1]
        partitions = [merged] + partitions[2:]

    while len(partitions) < num_partitions:
        partitions.sort(key=len, reverse=True)
        largest = partitions[0]
        mid = len(largest) // 2
        partitions = [largest[:mid], largest[mid:]] + partitions[1:]

    return partitions


def find_partition_containing(partitions, forget_nodes):
    forget_set = set(forget_nodes)
    affected = []
    for i, part in enumerate(partitions):
        if forget_set & set(part):
            affected.append(i)
    return affected


def train_partition_model(model, data, partition_indices, cfg, exclude_nodes=None):
    """Train a partition model on the full graph with loss only on partition nodes.
    
    Scales total epochs inversely with partition size for fair comparison.
    """
    device = cfg["device"]
    num_classes = cfg["data"]["num_classes"]
    lr = cfg.get("graph_eraser", {}).get("lr", cfg["sisa"]["lr"])
    base_epochs = cfg.get("graph_eraser", {}).get("retrain_epochs", cfg["sisa"]["retrain_epochs"])

    model = model.to(device)
    data = data.to(device)

    mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_indices = [i for i in partition_indices if data.train_mask[i]]
    if exclude_nodes:
        exclude_set = set(exclude_nodes)
        train_indices = [i for i in train_indices if i not in exclude_set]

    n_samples = len(train_indices)
    if n_samples == 0:
        return model

    # Scale epochs to match paper's training budget (~11k node-level updates)
    target_total_updates = base_epochs * 375
    total_epochs = max(base_epochs, int(target_total_updates / max(n_samples, 1)))
    total_epochs = min(total_epochs, 500)

    idx_tensor = torch.tensor(train_indices, dtype=torch.long, device=device)
    mask[idx_tensor] = True

    y_sub = data.y[mask]
    class_counts = torch.zeros(num_classes, device=device)
    for c in range(num_classes):
        class_counts[c] = (y_sub == c).sum().float()
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


def graph_eraser_unlearn(chains, forget_sets, cfg):
    num_partitions = cfg.get("graph_eraser", {}).get("num_partitions", cfg["sisa"]["num_shards"])
    device = cfg["device"]

    chain_models = {}
    chain_weights = {}

    for chain_name, data in chains.items():
        forget_nodes = forget_sets.get(chain_name, [])
        partitions = community_partition(data, num_partitions, seed=cfg["seed"])
        affected_ids = find_partition_containing(partitions, forget_nodes)

        models = []
        weights = []
        for pid, part in enumerate(partitions):
            m = build_model(cfg).to(device)
            if pid in affected_ids:
                m = train_partition_model(m, data, part, cfg, exclude_nodes=forget_nodes)
            else:
                m = train_partition_model(m, data, part, cfg)
            models.append(m)
            n_train = sum(1 for i in part if data.train_mask[i])
            weights.append(max(1, n_train))

        chain_state = models[0].state_dict()
        total_w = sum(weights)
        for key in chain_state:
            chain_state[key] = torch.zeros_like(chain_state[key], dtype=torch.float32)
            for m, w in zip(models, weights):
                chain_state[key] += m.state_dict()[key].float() * (w / total_w)

        n_retain = data.train_mask.sum().item() - len(forget_nodes)
        chain_models[chain_name] = chain_state
        chain_weights[chain_name] = max(1, n_retain)

    final_model = build_model(cfg)
    combined_state = final_model.state_dict()
    total_w = sum(chain_weights.values())
    for key in combined_state:
        combined_state[key] = torch.zeros_like(combined_state[key], dtype=torch.float32)
        for cn in chains:
            w = chain_weights[cn] / total_w
            combined_state[key] += chain_models[cn][key] * w

    final_model.load_state_dict(combined_state)
    return final_model
