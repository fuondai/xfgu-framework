import sys
import os
import time
import json
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

from utils.config import load_config, get_output_dir
from utils.seed import set_seed
from data.generate_sc_graph import build_cross_chain_data
from models.gnn import build_model
from federated.server import federated_train
from unlearning.xfgu import xfgu_unlearn
from baselines.retrain import full_retrain
from evaluation.evaluate import evaluate_model, evaluate_unlearning_quality
from graphs.influence_zone import select_forget_nodes


def run_single_ablation(cfg, seed=42):
    set_seed(seed)
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy["seed"] = seed

    chains = build_cross_chain_data(cfg_copy)
    global_model = build_model(cfg_copy)
    trained_model, _ = federated_train(global_model, chains, cfg_copy)

    forget_sets = {}
    for name, data in chains.items():
        fn = select_forget_nodes(data, cfg_copy["unlearning"]["forget_ratio"], seed=seed)
        forget_sets[name] = fn

    t0 = time.time()
    xfgu_model, zone_sizes, dp_sigma = xfgu_unlearn(
        copy.deepcopy(trained_model), chains, forget_sets, cfg_copy
    )
    unlearn_time = time.time() - t0

    retrain_model = full_retrain(None, chains, forget_sets, cfg_copy)

    utility = evaluate_model(xfgu_model, chains, cfg_copy, "XFGU")
    uq = evaluate_unlearning_quality(
        xfgu_model, retrain_model, trained_model,
        chains, forget_sets, cfg_copy
    )

    mia_accs = []
    for cn, mia in uq["mia_per_chain"].items():
        mia_accs.append(mia["mia_accuracy"])

    return {
        "f1": utility["overall"]["f1"],
        "accuracy": utility["overall"]["accuracy"],
        "auc": utility["overall"]["auc"],
        "mia_accuracy": np.mean(mia_accs) if mia_accs else 0.5,
        "param_dist": uq["param_distance_to_retrain"],
        "time": unlearn_time,
        "zone_sizes": zone_sizes,
        "dp_sigma": dp_sigma,
    }


def ablation_lhop(base_cfg, output_dir):
    print("\n=== Ablation: L-hop ===")
    results = {}
    for l_hop in [1, 2, 3]:
        cfg = copy.deepcopy(base_cfg)
        cfg["unlearning"]["l_hop"] = l_hop
        print(f"  L-hop = {l_hop}")
        r = run_single_ablation(cfg)
        results[l_hop] = r
        print(f"    F1={r['f1']:.4f} MIA={r['mia_accuracy']:.4f} "
              f"Time={r['time']:.2f}s Zones={r['zone_sizes']}")

    with open(os.path.join(output_dir, "ablation_lhop.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def ablation_forget_ratio(base_cfg, output_dir):
    print("\n=== Ablation: Forget Ratio ===")
    results = {}
    for ratio in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]:
        cfg = copy.deepcopy(base_cfg)
        cfg["unlearning"]["forget_ratio"] = ratio
        print(f"  Forget ratio = {ratio}")
        r = run_single_ablation(cfg)
        results[str(ratio)] = r
        print(f"    F1={r['f1']:.4f} MIA={r['mia_accuracy']:.4f} Time={r['time']:.2f}s")

    with open(os.path.join(output_dir, "ablation_forget_ratio.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def ablation_num_chains(base_cfg, output_dir):
    print("\n=== Ablation: Number of Chains ===")
    chain_configs = {
        2: (["ETH", "BSC"], [800, 500]),
        3: (["ETH", "BSC", "Polygon"], [800, 500, 300]),
        5: (["ETH", "BSC", "Polygon", "Avalanche", "Fantom"], [800, 500, 300, 200, 200]),
        7: (["ETH", "BSC", "Polygon", "Avalanche", "Fantom", "Arbitrum", "Optimism"],
            [800, 500, 300, 200, 200, 200, 200]),
    }
    results = {}
    for k, (names, sizes) in chain_configs.items():
        cfg = copy.deepcopy(base_cfg)
        cfg["data"]["num_chains"] = k
        cfg["data"]["chain_names"] = names
        cfg["data"]["chain_sizes"] = sizes
        cfg["sisa"]["num_shards"] = min(k, cfg["sisa"]["num_shards"])
        print(f"  K = {k} chains")
        r = run_single_ablation(cfg)
        results[k] = r
        print(f"    F1={r['f1']:.4f} MIA={r['mia_accuracy']:.4f} Time={r['time']:.2f}s")

    with open(os.path.join(output_dir, "ablation_num_chains.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def ablation_reverse_steps(base_cfg, output_dir):
    print("\n=== Ablation: Reverse Steps ===")
    results = {}
    for steps in [1, 3, 5, 10, 20]:
        cfg = copy.deepcopy(base_cfg)
        cfg["unlearning"]["reverse_steps"] = steps
        print(f"  Reverse steps = {steps}")
        r = run_single_ablation(cfg)
        results[steps] = r
        print(f"    F1={r['f1']:.4f} MIA={r['mia_accuracy']:.4f} Time={r['time']:.2f}s")

    with open(os.path.join(output_dir, "ablation_reverse_steps.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def ablation_dp_epsilon(base_cfg, output_dir):
    print("\n=== Ablation: DP Epsilon ===")
    results = {}
    for eps in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
        cfg = copy.deepcopy(base_cfg)
        cfg["unlearning"]["dp_epsilon"] = eps
        print(f"  Epsilon = {eps}")
        r = run_single_ablation(cfg)
        results[str(eps)] = r
        print(f"    F1={r['f1']:.4f} MIA={r['mia_accuracy']:.4f} "
              f"DP_sigma={r['dp_sigma']:.4f}")

    with open(os.path.join(output_dir, "ablation_dp_epsilon.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    base_cfg = load_config(config_path)
    output_dir = get_output_dir("outputs", "ablation")

    all_results = {}
    all_results["lhop"] = ablation_lhop(base_cfg, output_dir)
    all_results["forget_ratio"] = ablation_forget_ratio(base_cfg, output_dir)
    all_results["num_chains"] = ablation_num_chains(base_cfg, output_dir)
    all_results["reverse_steps"] = ablation_reverse_steps(base_cfg, output_dir)
    all_results["dp_epsilon"] = ablation_dp_epsilon(base_cfg, output_dir)

    with open(os.path.join(output_dir, "all_ablation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAll ablation results saved to {output_dir}")


if __name__ == "__main__":
    main()
