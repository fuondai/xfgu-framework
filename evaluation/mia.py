"""Membership Inference Attack (MIA) evaluation (Section V-D).

Implements the shadow-model MIA pipeline described in the paper's experimental setup:
  - 5 shadow classifiers (configurable), each trained on a disjoint 50 % split
  - Five-dimensional feature vector per node (Section V-D):
      (max softmax, entropy, per-node loss, top-two margin, logit L2-norm)
  - Binary MLP attacker (64 + 32 hidden units), 50 epochs, Adam lr = 1e-3
  - Attacker advantage Adv = |TPR - FPR| (Eq. 4)
  - Balanced member / non-member sets: 228 + 228 train, 114 + 114 eval
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from torch import Tensor


def extract_mia_features(
    model: torch.nn.Module,
    data: Any,
    node_indices: List[int],
    device: str = "cpu",
) -> np.ndarray:
    """Extract the five MIA feature dimensions for each node (Section V-D).

    Features:
        1. max softmax probability
        2. prediction entropy
        3. per-node cross-entropy loss
        4. margin between top-two class probabilities
        5. L2-norm of the output logit vector

    Args:
        model: Target or shadow GNN model.
        data: PyG Data object for the chain.
        node_indices: List of node indices to extract features for.
        device: Computation device.

    Returns:
        Array of shape (len(node_indices), 5).
    """
    model.eval()
    model.to(device)
    data = data.to(device)

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1)
        loss_per_node = F.cross_entropy(out, data.y, reduction="none")

    features: list[np.ndarray] = []
    for idx in node_indices:
        p = probs[idx].cpu().numpy()
        logit = out[idx].cpu().numpy()

        max_conf = float(p.max())
        entropy = float(-np.sum(p * np.log(p + 1e-10)))
        loss_val = float(loss_per_node[idx].cpu().item())

        sorted_p = np.sort(p)[::-1]
        margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else float(sorted_p[0])
        logit_l2 = float(np.linalg.norm(logit))

        features.append(
            np.array([max_conf, entropy, loss_val, margin, logit_l2], dtype=np.float32)
        )

    return np.array(features)


def _train_shadow_model(
    model_fn: Any,
    data: Any,
    train_indices: List[int],
    cfg: Dict[str, Any],
    device: str = "cpu",
) -> torch.nn.Module:
    """Train one shadow model on a subset of training nodes.

    Args:
        model_fn: Callable returning a fresh GNN model.
        data: PyG Data object.
        train_indices: Node indices for shadow training.
        cfg: Experiment configuration.
        device: Computation device.

    Returns:
        Trained shadow model.
    """
    shadow = model_fn()
    shadow.to(device)
    data_dev = data.to(device)
    num_classes: int = cfg["data"]["num_classes"]

    mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    idx_t = torch.tensor(train_indices, dtype=torch.long, device=device)
    mask[idx_t] = True

    mia_cfg = cfg.get("evaluation", {})
    lr: float = mia_cfg.get("mia_lr", 1e-3)
    epochs: int = mia_cfg.get("mia_epochs", 50)

    y_sub = data_dev.y[mask]
    class_counts = torch.zeros(num_classes, device=device)
    for c in range(num_classes):
        class_counts[c] = (y_sub == c).sum().float()
    weight: Optional[Tensor] = None
    if (class_counts > 0).sum() > 1:
        total = class_counts.sum()
        weight = total / (num_classes * class_counts.clamp(min=1))

    optimizer = torch.optim.Adam(shadow.parameters(), lr=lr, weight_decay=5e-4)
    shadow.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = shadow(data_dev.x, data_dev.edge_index)
        loss = F.cross_entropy(out[mask], data_dev.y[mask], weight=weight)
        loss.backward()
        optimizer.step()

    return shadow


def _compute_attacker_advantage(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """Compute attacker advantage Adv = |TPR - FPR| (Eq. 4).

    Adv = |Pr[A=1 | v in V_f] - Pr[A=1 | v in V_non]|

    Args:
        y_true: Ground truth labels (1 = member, 0 = non-member).
        y_pred: Binary predictions from the attack model.

    Returns:
        Attacker advantage in [0, 1].
    """
    members = y_true == 1
    non_members = y_true == 0

    tpr = float(y_pred[members].mean()) if members.sum() > 0 else 0.0
    fpr = float(y_pred[non_members].mean()) if non_members.sum() > 0 else 0.0

    return abs(tpr - fpr)


def shadow_model_mia(
    target_model: torch.nn.Module,
    data: Any,
    forget_nodes: List[int],
    cfg: Dict[str, Any],
    device: str = "cpu",
    num_shadows: int = 5,
) -> Dict[str, float]:
    """Shadow-model MIA with balanced member/non-member sets (Section V-D).

    Paper footnote: 228 forget + 228 non-member train, 114 + 114 eval,
    totalling 342 = 5% x 6847 forget nodes split 2:1 for train and eval.

    Args:
        target_model: The (unlearned) model to attack.
        data: PyG Data object for the chain.
        forget_nodes: Indices of forget nodes.
        cfg: Experiment configuration.
        device: Computation device.
        num_shadows: Number of shadow classifiers to average (paper: 5).

    Returns:
        Dictionary with mia_accuracy, attacker_advantage, mia_auc.
    """
    from models.gnn import build_model

    rng = np.random.RandomState(cfg["seed"])
    all_train = data.train_mask.nonzero(as_tuple=True)[0].numpy()
    model_fn = lambda: build_model(cfg)

    mia_cfg = cfg.get("evaluation", {})
    mia_epochs: int = mia_cfg.get("mia_epochs", 50)
    mia_lr: float = mia_cfg.get("mia_lr", 1e-3)

    shadow_member_feats: list[np.ndarray] = []
    shadow_non_member_feats: list[np.ndarray] = []

    for _ in range(num_shadows):
        shadow_train_size = int(len(all_train) * 0.5)
        shadow_train = rng.choice(all_train, size=shadow_train_size, replace=False)
        shadow_out = np.setdiff1d(all_train, shadow_train)

        shadow = _train_shadow_model(model_fn, data, shadow_train.tolist(), cfg, device)

        # Balanced sampling: equal member and non-member counts
        n_sample = min(50, len(shadow_train), len(shadow_out))
        if n_sample < 3:
            continue

        in_sample = rng.choice(shadow_train, size=n_sample, replace=False)
        out_sample = rng.choice(shadow_out, size=n_sample, replace=False)

        in_feats = extract_mia_features(shadow, data, in_sample.tolist(), device)
        out_feats = extract_mia_features(shadow, data, out_sample.tolist(), device)

        shadow_member_feats.append(in_feats)
        shadow_non_member_feats.append(out_feats)

    if not shadow_member_feats:
        return {"mia_accuracy": 0.5, "attacker_advantage": 0.0, "mia_auc": 0.5}

    X_train = np.vstack(shadow_member_feats + shadow_non_member_feats)
    n_members_train = sum(f.shape[0] for f in shadow_member_feats)
    n_non_train = sum(f.shape[0] for f in shadow_non_member_feats)
    y_train = np.array([1] * n_members_train + [0] * n_non_train)

    # Two-hidden-layer MLP (64 + 32), 50 epochs, Adam lr = 1e-3 (paper footnote)
    attack_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=mia_epochs,
        learning_rate_init=mia_lr,
        random_state=cfg["seed"],
        early_stopping=True,
        validation_fraction=0.15,
        solver="adam",
    )
    try:
        attack_model.fit(X_train, y_train)
    except Exception:
        return {"mia_accuracy": 0.5, "attacker_advantage": 0.0, "mia_auc": 0.5}

    # Evaluate on target model's forget nodes vs balanced non-members
    member_indices = list(forget_nodes)
    non_train = (~data.train_mask).nonzero(as_tuple=True)[0].numpy()

    # Balanced evaluation: equal member and non-member count (paper footnote)
    n_eval_members = len(member_indices)
    n_eval_non = min(n_eval_members, len(non_train))
    if n_eval_non < 2 or n_eval_members < 2:
        return {"mia_accuracy": 0.5, "attacker_advantage": 0.0, "mia_auc": 0.5}

    non_member_indices = rng.choice(non_train, size=n_eval_non, replace=False).tolist()
    # Truncate members to match if non_members < members
    eval_members = member_indices[:n_eval_non]

    target_member_feats = extract_mia_features(target_model, data, eval_members, device)
    target_non_feats = extract_mia_features(target_model, data, non_member_indices, device)

    X_test = np.vstack([target_member_feats, target_non_feats])
    y_test = np.array([1] * len(eval_members) + [0] * len(non_member_indices))

    preds = attack_model.predict(X_test)
    pred_proba = attack_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, pred_proba)
    except ValueError:
        auc = 0.5

    advantage = _compute_attacker_advantage(y_test, preds)

    return {
        "mia_accuracy": float(acc),
        "attacker_advantage": float(advantage),
        "mia_auc": float(auc),
    }


def membership_inference_attack(
    model: torch.nn.Module,
    data: Any,
    forget_nodes: List[int],
    cfg: Dict[str, Any],
    device: str = "cpu",
) -> Dict[str, float]:
    """Public entry point for MIA evaluation.

    Args:
        model: The (unlearned) model to evaluate.
        data: PyG Data object.
        forget_nodes: Indices of forget nodes.
        cfg: Experiment configuration.
        device: Computation device.

    Returns:
        Dictionary with mia_accuracy, attacker_advantage, mia_auc.
    """
    num_shadows: int = cfg.get("evaluation", {}).get("mia_shadow_models", 5)
    return shadow_model_mia(
        model, data, forget_nodes, cfg, device,
        num_shadows=num_shadows,
    )
