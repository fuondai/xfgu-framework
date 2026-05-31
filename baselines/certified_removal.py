"""Certified Removal baseline (Guo et al., ICML 2020).

Reference:
    Baseline 7 in Table II.  Conjugate-gradient Newton update with
    damping lambda = 0.01, applied per client and aggregated via FedAvg
    (Section V-A).

Configuration keys (under ``certified_removal``):
    damping    : float – Hessian damping (default 0.01)
    scale      : float – Newton step scale factor (default 1.0)
    noise_sigma: float – optional certification noise (default 0.0)
    cg_max_iter: int   – CG max iterations (default 50)
    cg_tol     : float – CG convergence tolerance (default 1e-6)
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from federated.server import fedavg_aggregate
from graphs.influence_zone import get_forget_retain_masks


def _hessian_vector_product(model, data, mask, device, v_flat, damping):
    """
    Compute (H_retain + damping * I) * v via double-backpropagation,
    where H_retain is the Hessian of the retain-set loss w.r.t. parameters.

    This avoids forming H explicitly; cost is two forward + one backward pass.
    """
    model.train()
    model.to(device)
    data = data.to(device)

    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[mask], data.y[mask])

    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grads = torch.cat([g.reshape(-1) for g in grads])

    # Directional derivative: d/d(eps) [grad_flat(theta + eps*v)] at eps=0
    gv = (flat_grads * v_flat.detach()).sum()
    hvp = torch.autograd.grad(gv, params, retain_graph=False)
    flat_hvp = torch.cat([h.reshape(-1).detach() for h in hvp])

    return flat_hvp + damping * v_flat


def _conjugate_gradient(hvp_fn, b_flat, max_iter=50, tol=1e-6):
    """
    Solve (H + damping * I) x = b using the conjugate gradient method.
    For the small GCN used in experiments (13,120 params) this converges
    in tens of iterations, replacing the O(d^2) full Hessian inversion.
    """
    x = torch.zeros_like(b_flat)
    r = b_flat.clone()
    p = r.clone()
    r_dot = (r * r).sum()

    for _ in range(max_iter):
        Ap = hvp_fn(p)
        pAp = (p * Ap).sum()
        if pAp.abs().item() < 1e-12:
            break
        alpha = r_dot / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        r_dot_new = (r * r).sum()
        if r_dot_new.sqrt().item() < tol:
            break
        beta = r_dot_new / r_dot
        p = r + beta * p
        r_dot = r_dot_new

    return x


def certified_removal_unlearn(global_model, chains, forget_sets, cfg):
    """
    Certified data removal via a Newton update (Guo et al., ICML 2020).

    The Newton step H_retain^{-1} * grad_forget is computed via conjugate
    gradient rather than the diagonal approximation g/(lambda+eps), which
    was incorrect in the prior version and systematically overestimated
    parameter distance, disadvantaging this baseline unfairly.
    """
    device = cfg["device"]
    damping = cfg.get("certified_removal", {}).get("damping", 0.01)
    scale_factor = cfg.get("certified_removal", {}).get("scale", 1.0)
    noise_sigma = cfg.get("certified_removal", {}).get("noise_sigma", 0.0)
    cg_max_iter = cfg.get("certified_removal", {}).get("cg_max_iter", 50)
    cg_tol = cfg.get("certified_removal", {}).get("cg_tol", 1e-6)

    unlearned_states = []
    client_weights = []
    certification_radii = {}

    for chain_name, data in chains.items():
        forget_nodes = forget_sets.get(chain_name, [])
        if len(forget_nodes) == 0:
            local_model = copy.deepcopy(global_model)
            unlearned_states.append(local_model.state_dict())
            client_weights.append(data.train_mask.sum().item())
            certification_radii[chain_name] = 0.0
            continue

        local_model = copy.deepcopy(global_model)
        _, forget_train_mask, retain_train_mask = get_forget_retain_masks(data, forget_nodes)

        local_model.train()
        local_model.to(device)
        data_dev = data.to(device)

        # Step 1: Compute gradient of the forget-set loss
        out = local_model(data_dev.x, data_dev.edge_index)
        if forget_train_mask.sum() > 0:
            loss_forget = F.cross_entropy(out[forget_train_mask], data_dev.y[forget_train_mask])
        else:
            loss_forget = torch.tensor(0.0, device=device, requires_grad=False)

        local_model.zero_grad()
        if loss_forget.requires_grad:
            loss_forget.backward()

        params = [p for p in local_model.parameters() if p.requires_grad]
        forget_grad_flat = torch.cat([
            p.grad.reshape(-1).detach() if p.grad is not None
            else torch.zeros(p.numel(), device=device)
            for p in params
        ])

        n_total = data.train_mask.sum().float().item()
        n_forget = forget_train_mask.sum().float().item()
        n_retain = max(n_total - n_forget, 1.0)

        # Step 2: CG solve for (H_retain + damping*I)^{-1} * grad_forget
        # This is the exact Newton-step of certified removal (Guo et al. 2020)
        if retain_train_mask.sum() > 0 and forget_grad_flat.norm().item() > 1e-10:
            def hvp_fn(v):
                return _hessian_vector_product(
                    local_model, data_dev, retain_train_mask, device, v, damping
                )
            newton_step_flat = _conjugate_gradient(
                hvp_fn, forget_grad_flat, max_iter=cg_max_iter, tol=cg_tol
            )
        else:
            # Fall back to scaled gradient if retain set is empty
            newton_step_flat = forget_grad_flat / (damping + 1e-8)

        # Certification radius: ||H^{-1} grad|| / n_retain (approximate)
        cert_radius = forget_grad_flat.norm().item() / (damping * n_retain)
        certification_radii[chain_name] = cert_radius

        # Step 3: Apply Newton update
        with torch.no_grad():
            offset = 0
            for param in params:
                numel = param.numel()
                step = newton_step_flat[offset:offset + numel].reshape(param.shape)
                param.add_((scale_factor / n_retain) * step)
                if noise_sigma > 0:
                    param.add_(torch.randn_like(param) * noise_sigma)
                offset += numel

        unlearned_states.append(local_model.state_dict())
        client_weights.append(max(1, retain_train_mask.sum().item()))

    unlearned_global = copy.deepcopy(global_model)
    unlearned_global = fedavg_aggregate(unlearned_global, unlearned_states, client_weights)
    return unlearned_global, certification_radii
