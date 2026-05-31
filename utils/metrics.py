"""Evaluation metrics for utility and unlearning quality (Section V-D)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute accuracy, macro-F1, and AUC.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (for AUC).

    Returns:
        Dictionary with 'accuracy', 'f1', 'auc' keys.
    """
    acc = accuracy_score(y_true, y_pred)
    n_classes = len(np.unique(y_true))
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    auc = 0.0
    if y_prob is not None and n_classes > 1:
        try:
            if n_classes == 2:
                auc = roc_auc_score(y_true, y_prob)
            else:
                if y_prob.ndim == 1:
                    auc = 0.0
                else:
                    auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except ValueError:
            auc = 0.0
    return {"accuracy": acc, "f1": f1, "auc": auc}


def compute_parameter_distance(
    params_a: List[Any], params_b: List[Any]
) -> float:
    """Compute L2 distance between two parameter lists (||theta' - theta*||_2)."""
    dist = 0.0
    for pa, pb in zip(params_a, params_b):
        dist += ((pa - pb) ** 2).sum().item()
    return dist ** 0.5


def compute_prediction_disagreement(
    preds_a: List[Any], preds_b: List[Any]
) -> float:
    """Compute fraction of disagreeing predictions between two models."""
    return (np.array(preds_a) != np.array(preds_b)).mean()
