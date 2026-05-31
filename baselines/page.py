"""PAGE baseline (Ai et al., 2026).

Reference:
    Ai, Y. et al. "PAGE: Unified Federated Graph Unlearning via Adversarial
    Graph Generation and Negative Knowledge Distillation." 2026.

    Inherently federated graph unlearning that:
      1. Generates adversarial graph perturbations to detect residual knowledge.
      2. Applies negative knowledge distillation on influenced clients to
         suppress forget-set representations.
    Adapted to our five-chain benchmark (Section V-A).

Configuration keys (under ``page``):
    lr            : float – distillation learning rate (default 1e-3)
    steps         : int   – distillation steps (default 30)
    temperature   : float – KD temperature (default 2.0)
    lambda_nkd    : float – negative KD weight (default 1.0)
    perturb_ratio : float – fraction of edges to perturb (default 0.05)
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


def _generate_adversarial_edges(
    edge_index: Tensor,
    forget_nodes: List[int],
    num_nodes: int,
    perturb_ratio: float,
    rng: np.random.RandomState,
) -> Tensor:
    """Generate adversarial edge perturbations around forget nodes.

    Adds random edges between forget nodes and non-forget nodes to amplify
    residual knowledge leakage, enabling detection via distillation.

    Args:
        edge_index: Original edge index.
        forget_nodes: Forget node indices.
        num_nodes: Total number of nodes.
        perturb_ratio: Fraction of existing edges to add as perturbations.
        rng: Random state for reproducibility.

    Returns:
        Perturbed edge index.
    """
    n_perturb = max(1, int(edge_index.size(1) * perturb_ratio))
    forget_set = set(forget_nodes)
    non_forget = [i for i in range(num_nodes) if i not in forget_set]

    if len(forget_nodes) == 0 or len(non_forget) == 0:
        return edge_index

    src = rng.choice(forget_nodes, size=n_perturb, replace=True)
    dst = rng.choice(non_forget, size=n_perturb, replace=True)

    new_edges = torch.tensor(
        [list(src) + list(dst), list(dst) + list(src)],
        dtype=torch.long,
    )
    return torch.cat([edge_index, new_edges], dim=1)


def _negative_kd_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    forget_mask: Tensor,
    temperature: float,
) -> Tensor:
    """Negative knowledge distillation loss (PAGE, Sec. 3.2).

    Maximises KL divergence between the student and teacher on forget nodes,
    suppressing the teacher's knowledge of the forget set.

    Args:
        student_logits: Current model output.
        teacher_logits: Original (pre-unlearning) model output.
        forget_mask: Boolean mask over forget training nodes.
        temperature: Softmax temperature for KD.

    Returns:
        Negative KD loss scalar.
    """
    if forget_mask.sum() == 0:
        return torch.tensor(0.0, device=student_logits.device)

    s = F.log_softmax(student_logits[forget_mask] / temperature, dim=1)
    t = F.softmax(teacher_logits[forget_mask] / temperature, dim=1)
    # Negative KD: we want to *maximise* divergence, so negate the KL
    kl = F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)
    return -kl


def page_unlearn(
    global_model: torch.nn.Module,
    chains: Dict[str, Any],
    forget_sets: Dict[str, List[int]],
    cfg: Dict[str, Any],
) -> torch.nn.Module:
    """Execute PAGE unlearning per client with FedAvg aggregation.

    Args:
        global_model: Trained federated GNN model.
        chains: Mapping from chain name to PyG Data objects.
        forget_sets: Mapping from chain name to forget node indices.
        cfg: Experiment configuration.

    Returns:
        Unlearned global model.
    """
    device = cfg["device"]
    pg_cfg = cfg.get("page", {})
    lr: float = pg_cfg.get("lr", 1e-3)
    steps: int = pg_cfg.get("steps", 30)
    temperature: float = pg_cfg.get("temperature", 2.0)
    lambda_nkd: float = pg_cfg.get("lambda_nkd", 1.0)
    perturb_ratio: float = pg_cfg.get("perturb_ratio", 0.05)
    num_classes: int = cfg["data"]["num_classes"]

    rng = np.random.RandomState(cfg["seed"])

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

        # Cache teacher (original) predictions
        local_model.eval()
        with torch.no_grad():
            teacher_logits = local_model(data_dev.x, data_dev.edge_index).detach()

        optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
        local_model.train()

        for _ in range(steps):
            optimizer.zero_grad()

            # Retain-set CE on ORIGINAL graph
            out = local_model(data_dev.x, data_dev.edge_index)
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

            # Negative KD on forget nodes
            nkd_loss = _negative_kd_loss(
                out, teacher_logits, forget_train_mask, temperature,
            )

            loss = ce_loss + lambda_nkd * nkd_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
            optimizer.step()

        unlearned_states.append(local_model.state_dict())
        client_weights.append(max(1, int(retain_train_mask.sum().item())))

    unlearned_global = copy.deepcopy(global_model)
    unlearned_global = fedavg_aggregate(unlearned_global, unlearned_states, client_weights)
    return unlearned_global
