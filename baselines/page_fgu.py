"""PAGE-FGU baseline (Wang et al., 2024).

Reference:
    Baseline 6 in Table II.  Momentum-based gradient correction
    approximating the retrained gradient, inherently federated
    (Section V-A).

Configuration keys (under ``page_fgu``):
    lr               : float – learning rate (default reverse_lr)
    correction_steps : int   – correction iterations (default 10)
    momentum         : float – momentum coefficient (default 0.9)
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from federated.server import fedavg_aggregate
from graphs.influence_zone import get_forget_retain_masks


def compute_gradient_on_data(model, data, mask, device):
    model.train()
    model.to(device)
    data = data.to(device)

    if mask.sum() == 0:
        return {n: torch.zeros_like(p) for n, p in model.named_parameters()}

    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[mask], data.y[mask])
    model.zero_grad()
    loss.backward()

    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.clone().detach()
        else:
            grads[name] = torch.zeros_like(param)
    return grads


def page_fgu_unlearn(global_model, chains, forget_sets, cfg):
    device = cfg["device"]
    lr = cfg.get("page_fgu", {}).get("lr", cfg["unlearning"]["reverse_lr"])
    correction_steps = cfg.get("page_fgu", {}).get("correction_steps", 10)
    momentum = cfg.get("page_fgu", {}).get("momentum", 0.9)

    unlearned_states = []
    client_weights = []

    for chain_name, data in chains.items():
        forget_nodes = forget_sets.get(chain_name, [])
        if len(forget_nodes) == 0:
            local_model = copy.deepcopy(global_model)
            unlearned_states.append(local_model.state_dict())
            client_weights.append(data.train_mask.sum().item())
            continue

        local_model = copy.deepcopy(global_model)
        _, forget_train_mask, retain_train_mask = get_forget_retain_masks(data, forget_nodes)

        full_grad = compute_gradient_on_data(local_model, data, data.train_mask, device)
        forget_grad = compute_gradient_on_data(local_model, data, forget_train_mask, device)

        n_total = data.train_mask.sum().float().item()
        n_forget = forget_train_mask.sum().float().item()

        if n_total > 0 and n_forget > 0:
            scale = n_total / (n_total - n_forget)
        else:
            scale = 1.0

        with torch.no_grad():
            correction = {}
            for name in full_grad:
                correction[name] = scale * (full_grad[name] - (n_forget / n_total) * forget_grad[name]) - full_grad[name]

        velocity = {name: torch.zeros_like(p) for name, p in local_model.named_parameters()}

        for step in range(correction_steps):
            retain_grad = compute_gradient_on_data(local_model, data, retain_train_mask, device)

            with torch.no_grad():
                for name, param in local_model.named_parameters():
                    projected_grad = retain_grad.get(name, torch.zeros_like(param))
                    total_update = projected_grad + correction[name] / correction_steps
                    velocity[name] = momentum * velocity[name] + total_update
                    param.sub_(lr * velocity[name])

        unlearned_states.append(local_model.state_dict())
        client_weights.append(max(1, retain_train_mask.sum().item()))

    unlearned_global = copy.deepcopy(global_model)
    unlearned_global = fedavg_aggregate(unlearned_global, unlearned_states, client_weights)
    return unlearned_global
