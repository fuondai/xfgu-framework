"""EraseRectify baseline (Yang et al., 2025).

Reference:
    Yang, X. et al. "Erase Then Rectify: Training-Free Parameter Editing
    for Graph Unlearning." 2025.

    A training-free parameter editing approach that:
      1. Computes parameter importance scores from local graph gradients.
      2. Masks critical parameters (top-k by Fisher information).
      3. Applies a gradient approximation to erase forget-set influence.
      4. Rectifies remaining parameters to restore utility.
    Applied per client and aggregated via FedAvg (Section V-A).

Configuration keys (under ``erase_rectify``):
    mask_ratio : float – fraction of parameters to mask (default 0.1)
    alpha      : float – erasure scaling factor (default 1.0)
    beta       : float – rectification scaling factor (default 0.5)
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor

from federated.server import fedavg_aggregate
from graphs.influence_zone import get_forget_retain_masks


def _compute_fisher_importance(
    model: torch.nn.Module,
    data: Any,
    mask: Tensor,
    device: torch.device,
) -> Dict[str, Tensor]:
    """Compute Fisher information diagonal as parameter importance scores.

    Args:
        model: GNN model.
        data: PyG Data object.
        mask: Boolean mask over nodes to use for importance estimation.
        device: Computation device.

    Returns:
        Dictionary mapping parameter names to importance tensors.
    """
    model.train()
    model.to(device)
    data = data.to(device)

    importance: Dict[str, Tensor] = {
        name: torch.zeros_like(p, device=device)
        for name, p in model.named_parameters()
    }

    if mask.sum() == 0:
        return importance

    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[mask], data.y[mask])
    model.zero_grad()
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            importance[name] = param.grad.detach().pow(2)

    return importance


def _compute_forget_gradient(
    model: torch.nn.Module,
    data: Any,
    forget_mask: Tensor,
    device: torch.device,
) -> Dict[str, Tensor]:
    """Compute gradient of the forget-set loss.

    Args:
        model: GNN model.
        data: PyG Data object.
        forget_mask: Boolean mask over forget training nodes.
        device: Computation device.

    Returns:
        Dictionary mapping parameter names to gradient tensors.
    """
    model.train()
    model.to(device)
    data = data.to(device)

    grads: Dict[str, Tensor] = {}
    if forget_mask.sum() == 0:
        for name, p in model.named_parameters():
            grads[name] = torch.zeros_like(p)
        return grads

    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[forget_mask], data.y[forget_mask])
    model.zero_grad()
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().clone()
        else:
            grads[name] = torch.zeros_like(param)
    return grads


def erase_rectify_unlearn(
    global_model: torch.nn.Module,
    chains: Dict[str, Any],
    forget_sets: Dict[str, List[int]],
    cfg: Dict[str, Any],
) -> torch.nn.Module:
    """Execute EraseRectify unlearning per client with FedAvg aggregation.

    Args:
        global_model: Trained federated GNN model.
        chains: Mapping from chain name to PyG Data objects.
        forget_sets: Mapping from chain name to forget node indices.
        cfg: Experiment configuration.

    Returns:
        Unlearned global model.
    """
    device = cfg["device"]
    er_cfg = cfg.get("erase_rectify", {})
    mask_ratio: float = er_cfg.get("mask_ratio", 0.1)
    alpha: float = er_cfg.get("alpha", 1.0)
    beta: float = er_cfg.get("beta", 0.5)

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

        # Step 1: Compute parameter importance from retain set
        importance = _compute_fisher_importance(
            local_model, data_dev, retain_train_mask, device
        )

        # Step 2: Create critical parameter mask (top mask_ratio by importance)
        all_importance = torch.cat([v.flatten() for v in importance.values()])
        if all_importance.numel() > 0 and all_importance.max() > 0:
            threshold = torch.quantile(all_importance, 1.0 - mask_ratio)
            critical_mask: Dict[str, Tensor] = {
                name: (imp >= threshold).float()
                for name, imp in importance.items()
            }
        else:
            critical_mask = {
                name: torch.ones_like(imp) for name, imp in importance.items()
            }

        # Step 3: Compute forget-set gradient for erasure
        forget_grad = _compute_forget_gradient(
            local_model, data_dev, forget_train_mask, device
        )

        # Step 4: Erase (gradient ascent on forget set, masked)
        # and Rectify (gradient descent on retain set, masked)
        retain_grad = _compute_forget_gradient(
            local_model, data_dev, retain_train_mask, device
        )

        n_forget = max(1, int(forget_train_mask.sum().item()))
        n_retain = max(1, int(retain_train_mask.sum().item()))

        with torch.no_grad():
            for name, param in local_model.named_parameters():
                mask = critical_mask.get(name, torch.ones_like(param))
                fg = forget_grad.get(name, torch.zeros_like(param))
                rg = retain_grad.get(name, torch.zeros_like(param))

                # Erase: ascend on forget gradient (masked to non-critical)
                erase_update = (1.0 - mask) * fg / n_forget
                # Rectify: descend on retain gradient (masked to critical)
                rectify_update = mask * rg / n_retain

                param.add_(alpha * erase_update - beta * rectify_update)

        unlearned_states.append(local_model.state_dict())
        client_weights.append(max(1, int(retain_train_mask.sum().item())))

    unlearned_global = copy.deepcopy(global_model)
    unlearned_global = fedavg_aggregate(unlearned_global, unlearned_states, client_weights)
    return unlearned_global
