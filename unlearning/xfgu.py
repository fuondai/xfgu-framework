"""XFGU: Cross-chain Federated Graph Unlearning (Algorithm 1).

Implements the four-phase unlearning procedure described in Section IV:
  1. Influence-zone discovery via BFS (Section IV-A, Eq. 2)
  2. DP-protected reverse-gradient ascent (Section IV-B, Eqs. 3--7)
  3. Retain-set fine-tuning within the zone (Section IV-C, Eq. 8)
  4. Single-round FedAvg aggregation (Section IV-D, Eq. 9)

Noise calibration follows the Gaussian mechanism (Theorem 1, Eq. 10):
  sigma = sqrt(2 * ln(1.25 / delta)) / epsilon
"""

import copy
import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from graphs.influence_zone import get_forget_retain_masks, get_influence_zone


def calibrate_dp_noise(
    epsilon: float, delta: float, clip_norm: float, num_steps: int
) -> float:
    """Compute per-coordinate Gaussian noise std for (epsilon, delta)-DP.

    Implements Eq. 10: sigma = sqrt(2 * ln(1.25 / delta)) / epsilon.
    Returns sigma * C, the per-coordinate standard deviation of xi_t.

    Args:
        epsilon: Per-step privacy budget (epsilon = 2.0, Section V-A).
        delta: Privacy failure probability (delta = 1e-5, Section V-A).
        clip_norm: Per-sample gradient clipping bound C (C = 1.0, Section V-A).
        num_steps: Number of reverse-gradient steps T_r (unused in calibration;
            composition is handled separately via the moments accountant).

    Returns:
        The noise standard deviation sigma * C per coordinate.
    """
    if epsilon <= 0:
        return 0.0
    sigma_multiplier = math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
    return clip_norm * sigma_multiplier


def _compute_clipped_gradient(
    model: torch.nn.Module,
    data: Any,
    forget_idx: Tensor,
    edge_index: Tensor,
    clip_norm: float,
    device: torch.device,
) -> Dict[str, Tensor]:
    """Compute per-sample clipped and aggregated gradient over forget nodes.

    Implements Eqs. 3--5: per-sample gradient, per-sample clipping, aggregation.

    Args:
        model: Current GNN model.
        data: PyG Data object for the chain.
        forget_idx: Indices of forget training nodes.
        edge_index: Full edge index for the chain graph.
        clip_norm: Per-sample clipping bound C.
        device: Computation device.

    Returns:
        Dictionary mapping parameter names to aggregated clipped gradients.
    """
    accum: Dict[str, Tensor] = {
        name: torch.zeros_like(p, device=device)
        for name, p in model.named_parameters()
    }

    for idx in forget_idx:
        out = model(data.x, edge_index)
        single_loss = F.cross_entropy(
            out[idx].unsqueeze(0), data.y[idx].unsqueeze(0)
        )
        model.zero_grad()
        single_loss.backward()

        sq_norm = sum(
            p.grad.norm(2).item() ** 2
            for p in model.parameters()
            if p.grad is not None
        )
        per_sample_norm = math.sqrt(sq_norm)
        clip_coef = min(1.0, clip_norm / (per_sample_norm + 1e-8))

        for name, param in model.named_parameters():
            if param.grad is not None:
                accum[name].add_(param.grad.detach() * clip_coef)

    n_forget = len(forget_idx)
    return {name: g / n_forget for name, g in accum.items()}


def _apply_zone_projected_noise(
    avg_grad: Dict[str, Tensor], dp_sigma: float
) -> Dict[str, Tensor]:
    """Add zone-projected Gaussian noise to the aggregated gradient (Eq. 6).

    The zone-projection matrix M_Z zeros out noise coordinates where the
    aggregated gradient is zero, preserving parameters outside the influence zone.

    Args:
        avg_grad: Aggregated clipped gradient per parameter.
        dp_sigma: Per-coordinate noise std (sigma * C).

    Returns:
        Noisy gradient dictionary (g_avg + M_Z * xi_t).
    """
    noisy: Dict[str, Tensor] = {}
    for name, g in avg_grad.items():
        noise = torch.randn_like(g) * dp_sigma
        zone_mask = (g.abs() > 0).float()
        noisy[name] = g + noise * zone_mask
    return noisy


def reverse_gradient_step(
    model: torch.nn.Module,
    data: Any,
    forget_mask: Tensor,
    edge_index: Tensor,
    lr: float,
    clip_norm: float,
    dp_sigma: float,
    device: torch.device,
) -> None:
    """Execute one reverse-gradient ascent step (Eq. 7).

    theta^(t) = theta^(t-1) + eta_r * (g_avg + M_Z * xi_t)

    Args:
        model: GNN model to update in-place.
        data: PyG Data object for the chain.
        forget_mask: Boolean mask over forget training nodes.
        edge_index: Full chain edge index.
        lr: Reverse learning rate eta_r (2e-3, Section V-A).
        clip_norm: Per-sample clipping bound C (1.0).
        dp_sigma: Noise std sigma * C (approx 2.42).
        device: Computation device.
    """
    model.train()
    model.to(device)
    data_dev = data.to(device)
    edge_idx_dev = edge_index.to(device)

    forget_indices = forget_mask.nonzero(as_tuple=True)[0]
    if len(forget_indices) == 0:
        return

    avg_grad = _compute_clipped_gradient(
        model, data_dev, forget_indices, edge_idx_dev, clip_norm, device
    )
    noisy_grad = _apply_zone_projected_noise(avg_grad, dp_sigma)

    with torch.no_grad():
        for name, param in model.named_parameters():
            param.add_(lr * noisy_grad[name])


def finetune_retain(
    model: torch.nn.Module,
    data: Any,
    retain_mask: Tensor,
    edge_index: Tensor,
    lr: float,
    steps: int,
    device: torch.device,
) -> None:
    """Retain-set fine-tuning within the influence zone (Section IV-C, Eq. 8).

    Performs T_f gradient descent steps on the retain nodes within Z_L
    to restore utility after reverse-gradient perturbation.

    Args:
        model: GNN model to fine-tune in-place.
        data: PyG Data object for the chain.
        retain_mask: Boolean mask over retain training nodes in the zone.
        edge_index: Full chain edge index.
        lr: Fine-tuning learning rate eta_f (5e-3, Section V-A).
        steps: Number of fine-tuning steps T_f (15, Section V-A).
        device: Computation device.
    """
    model.train()
    model.to(device)
    data_dev = data.to(device)
    edge_idx_dev = edge_index.to(device)

    if retain_mask.sum() == 0:
        return

    num_classes = int(data_dev.y.max().item()) + 1
    y_retain = data_dev.y[retain_mask]
    class_counts = torch.zeros(num_classes, device=device)
    for c in range(num_classes):
        class_counts[c] = (y_retain == c).sum().float()
    weight: Tensor | None = None
    if (class_counts > 0).sum() > 1:
        total = class_counts.sum()
        weight = total / (num_classes * class_counts.clamp(min=1))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    for _ in range(steps):
        optimizer.zero_grad()
        out = model(data_dev.x, edge_idx_dev)
        loss = F.cross_entropy(out[retain_mask], data_dev.y[retain_mask], weight=weight)
        loss.backward()
        optimizer.step()


def xfgu_unlearn(
    global_model: torch.nn.Module,
    chains: Dict[str, Any],
    forget_sets: Dict[str, List[int]],
    cfg: Dict[str, Any],
) -> Tuple[torch.nn.Module, Dict[str, int], float]:
    """Execute the full XFGU unlearning procedure (Algorithm 1).

    Args:
        global_model: Trained federated GNN model theta.
        chains: Mapping from chain name to PyG Data objects.
        forget_sets: Mapping from chain name to forget node indices.
        cfg: Experiment configuration dictionary.

    Returns:
        Tuple of (unlearned global model theta', zone sizes per chain, dp_sigma).
    """
    device = cfg["device"]
    l_hop: int = cfg["unlearning"]["l_hop"]
    reverse_lr: float = cfg["unlearning"]["reverse_lr"]
    reverse_steps: int = cfg["unlearning"]["reverse_steps"]
    finetune_lr: float = cfg["unlearning"]["finetune_lr"]
    finetune_steps: int = cfg["unlearning"]["finetune_steps"]
    clip_norm: float = cfg["unlearning"]["clip_norm"]
    epsilon: float = cfg["unlearning"]["dp_epsilon"]
    delta: float = cfg["unlearning"]["dp_delta"]

    dp_sigma = calibrate_dp_noise(epsilon, delta, clip_norm, reverse_steps)

    unlearned_states: List[Dict[str, Tensor]] = []
    client_weights: List[int] = []
    zone_sizes: Dict[str, int] = {}

    for chain_name, data in chains.items():
        forget_nodes = forget_sets.get(chain_name, [])
        if len(forget_nodes) == 0:
            local_model = copy.deepcopy(global_model)
            unlearned_states.append(local_model.state_dict())
            client_weights.append(int(data.train_mask.sum().item()))
            zone_sizes[chain_name] = 0
            continue

        local_model = copy.deepcopy(global_model)

        influence_nodes, sub_edge_index, _ = get_influence_zone(
            data.edge_index, forget_nodes, l_hop, data.num_nodes
        )
        zone_sizes[chain_name] = len(influence_nodes)

        _, forget_train_mask, retain_train_mask = get_forget_retain_masks(
            data, forget_nodes
        )

        for _ in range(reverse_steps):
            reverse_gradient_step(
                local_model, data, forget_train_mask,
                data.edge_index, reverse_lr, clip_norm, dp_sigma, device,
            )

        zone_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        zone_mask[influence_nodes] = True
        zone_retain_mask = retain_train_mask & zone_mask

        finetune_retain(
            local_model, data, zone_retain_mask, data.edge_index,
            finetune_lr, finetune_steps, device,
        )

        unlearned_states.append(local_model.state_dict())
        client_weights.append(max(1, int(retain_train_mask.sum().item())))

    from federated.server import fedavg_aggregate

    unlearned_global = copy.deepcopy(global_model)
    unlearned_global = fedavg_aggregate(
        unlearned_global, unlearned_states, client_weights
    )

    return unlearned_global, zone_sizes, dp_sigma
