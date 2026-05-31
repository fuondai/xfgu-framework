"""Model evaluation: utility metrics and unlearning quality (Section V-D)."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from evaluation.mia import membership_inference_attack
from federated.client import client_evaluate
from utils.metrics import (
    compute_classification_metrics,
    compute_parameter_distance,
    compute_prediction_disagreement,
)


def evaluate_model(
    model: torch.nn.Module,
    chains: Dict[str, Any],
    cfg: Dict[str, Any],
    label: str = "model",
) -> Dict[str, Dict[str, float]]:
    """Evaluate classification metrics per chain and overall.

    Args:
        model: GNN model to evaluate.
        chains: Mapping from chain name to PyG Data objects.
        cfg: Experiment configuration.
        label: Optional label for logging.

    Returns:
        Dictionary mapping chain names (plus 'overall') to metric dicts.
    """
    device = cfg["device"]
    results = {}

    all_true, all_pred, all_prob = [], [], []

    for chain_name, data in chains.items():
        y_true, y_pred, y_prob = client_evaluate(model, data, data.test_mask, device)
        m = compute_classification_metrics(y_true, y_pred, y_prob)
        results[chain_name] = m
        all_true.extend(y_true)
        all_pred.extend(y_pred)
        all_prob.extend(y_prob)

    overall = compute_classification_metrics(
        np.array(all_true), np.array(all_pred), np.array(all_prob)
    )
    results["overall"] = overall
    return results


def evaluate_unlearning_quality(
    unlearned_model: torch.nn.Module,
    retrained_model: torch.nn.Module,
    original_model: torch.nn.Module,
    chains: Dict[str, Any],
    forget_sets: Dict[str, List[int]],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate unlearning quality: param distance, prediction disagreement, MIA.

    Args:
        unlearned_model: Model after unlearning.
        retrained_model: Gold-standard retrained model.
        original_model: Model before unlearning.
        chains: Mapping from chain name to PyG Data objects.
        forget_sets: Mapping from chain name to forget node indices.
        cfg: Experiment configuration.

    Returns:
        Dictionary of unlearning quality metrics.
    """
    device = cfg["device"]

    param_dist_to_retrain = compute_parameter_distance(
        list(unlearned_model.parameters()), list(retrained_model.parameters())
    )
    param_dist_to_original = compute_parameter_distance(
        list(unlearned_model.parameters()), list(original_model.parameters())
    )

    unlearned_preds = []
    retrained_preds = []
    for chain_name, data in chains.items():
        _, up, _ = client_evaluate(unlearned_model, data, data.test_mask, device)
        _, rp, _ = client_evaluate(retrained_model, data, data.test_mask, device)
        unlearned_preds.extend(up)
        retrained_preds.extend(rp)

    pred_disagree = compute_prediction_disagreement(unlearned_preds, retrained_preds)

    mia_results = {}
    for chain_name, data in chains.items():
        forget_nodes = forget_sets.get(chain_name, [])
        if len(forget_nodes) > 0:
            mia = membership_inference_attack(unlearned_model, data, forget_nodes, cfg, device)
            mia_results[chain_name] = mia

    return {
        "param_distance_to_retrain": param_dist_to_retrain,
        "param_distance_to_original": param_dist_to_original,
        "prediction_disagreement_with_retrain": pred_disagree,
        "mia_per_chain": mia_results,
    }
