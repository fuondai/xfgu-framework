"""Communication cost analysis (Section V-E, Table III)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import torch


def compute_model_size_bytes(model: torch.nn.Module) -> int:
    """Compute total model size in bytes."""
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.nelement() * param.element_size()
    for buf in model.buffers():
        total_bytes += buf.nelement() * buf.element_size()
    return total_bytes


def compute_model_size_params(model: torch.nn.Module) -> int:
    """Compute total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def analyze_communication_cost(
    model: torch.nn.Module,
    cfg: Dict[str, Any],
    timing_dict: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute per-method communication cost breakdown.

    Args:
        model: Trained GNN model.
        cfg: Experiment configuration.
        timing_dict: Optional wall-clock timings.

    Returns:
        Dictionary of communication cost metrics per method.
    """
    model_bytes = compute_model_size_bytes(model)
    model_params = compute_model_size_params(model)
    num_chains = cfg["data"]["num_chains"]
    num_rounds = cfg["federated"]["num_rounds"]
    num_shards = cfg["sisa"]["num_shards"]
    reverse_steps = cfg["unlearning"]["reverse_steps"]
    finetune_steps = cfg["unlearning"]["finetune_steps"]
    num_partitions = cfg.get("graph_eraser", {}).get("num_partitions", num_shards)

    results = {
        "model_size_bytes": model_bytes,
        "model_size_params": model_params,
        "model_size_mb": model_bytes / (1024 * 1024),
        "num_chains": num_chains,
    }

    methods = {}

    methods["FullRetrain"] = {
        "comm_rounds": num_rounds,
        "models_per_round": num_chains,
        "bytes_per_round": model_bytes * num_chains * 2,
        "total_bytes": model_bytes * num_chains * 2 * num_rounds,
    }

    methods["XFGU"] = {
        "comm_rounds": 1,
        "models_per_round": num_chains,
        "bytes_per_round": model_bytes * num_chains * 2,
        "total_bytes": model_bytes * num_chains * 2,
    }

    methods["SISA"] = {
        "comm_rounds": 1,
        "models_per_round": num_chains,
        "bytes_per_round": model_bytes * num_chains * 2,
        "total_bytes": model_bytes * num_chains * 2,
    }

    methods["NaiveFT"] = {
        "comm_rounds": 1,
        "models_per_round": num_chains,
        "bytes_per_round": model_bytes * num_chains * 2,
        "total_bytes": model_bytes * num_chains * 2,
    }

    methods["GraphEraser"] = {
        "comm_rounds": 1,
        "models_per_round": num_chains,
        "bytes_per_round": model_bytes * num_chains * 2,
        "total_bytes": model_bytes * num_chains * 2,
    }

    methods["CertRemoval"] = {
        "comm_rounds": 1,
        "models_per_round": num_chains,
        "bytes_per_round": model_bytes * num_chains * 2,
        "total_bytes": model_bytes * num_chains * 2,
    }

    retrain_total = methods["FullRetrain"]["total_bytes"]
    for name, m in methods.items():
        m["total_mb"] = m["total_bytes"] / (1024 * 1024)
        m["comm_reduction"] = retrain_total / max(m["total_bytes"], 1)

    if timing_dict:
        for name in methods:
            if name in timing_dict:
                methods[name]["wall_time_s"] = timing_dict[name]

    results["methods"] = methods
    return results


def print_communication_report(report: Dict[str, Any]) -> None:
    """Log communication cost report.

    Args:
        report: Output of analyze_communication_cost().
    """
    import logging

    _logger = logging.getLogger(__name__)
    _logger.info("=" * 90)
    _logger.info("COMMUNICATION COST ANALYSIS")
    _logger.info("=" * 90)
    _logger.info(
        "Model size: %s parameters (%.3f MB)",
        f"{report['model_size_params']:,}",
        report["model_size_mb"],
    )
    _logger.info("Number of chains: %d", report["num_chains"])

    header = (
        f"{'Method':<15} {'Rounds':<8} {'Bytes/Round':<15} "
        f"{'Total (MB)':<12} {'Reduction':<12} {'Time (s)':<10}"
    )
    _logger.info(header)
    _logger.info("-" * 82)

    for name, m in report["methods"].items():
        time_str = f"{m['wall_time_s']:.2f}" if "wall_time_s" in m else "-"
        _logger.info(
            "%s %s %s %s %s %s",
            f"{name:<15}",
            f"{m['comm_rounds']:<8}",
            f"{m['bytes_per_round']:>12,}  ",
            f"{m['total_mb']:>8.3f}   ",
            f"{m['comm_reduction']:>8.1f}x   ",
            f"{time_str:<10}",
        )


def save_communication_report(report, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "communication_cost.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path
