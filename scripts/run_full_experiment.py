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
from baselines.sisa import sisa_unlearn
from baselines.naive_finetune import naive_finetune_unlearn
from baselines.graph_eraser import graph_eraser_unlearn
from baselines.page_fgu import page_fgu_unlearn
from baselines.certified_removal import certified_removal_unlearn
from baselines.gnndelete import gnndelete_unlearn
from baselines.erase_rectify import erase_rectify_unlearn
from baselines.page import page_unlearn
from evaluation.evaluate import evaluate_model, evaluate_unlearning_quality
from evaluation.plotting import (
    plot_utility_comparison, plot_mia_comparison,
    plot_efficiency, plot_training_curve,
)
from graphs.influence_zone import select_forget_nodes
from evaluation.communication import analyze_communication_cost, print_communication_report, save_communication_report


ALL_METHODS = ["Original", "XFGU", "FullRetrain", "SISA",
               "NaiveFT", "GraphEraser", "PAGE-FGU", "CertRemoval",
               "GNNDelete", "EraseRectify", "PAGE"]
UNLEARN_METHODS = ["XFGU", "SISA", "NaiveFT",
                   "GraphEraser", "PAGE-FGU", "CertRemoval",
                   "GNNDelete", "EraseRectify", "PAGE"]
TIMED_METHODS = ["XFGU", "FullRetrain", "SISA", "NaiveFT",
                 "GraphEraser", "PAGE-FGU", "CertRemoval",
                 "GNNDelete", "EraseRectify", "PAGE"]


def run_single_seed(cfg, seed, output_dir):
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"Running experiment with seed={seed}")
    print(f"{'='*60}")

    cfg_copy = copy.deepcopy(cfg)
    cfg_copy["seed"] = seed

    print("\n[1/6] Building cross-chain smart contract graphs...")
    chains = build_cross_chain_data(cfg_copy)
    for name, data in chains.items():
        n_vuln = (data.y > 0).sum().item()
        n_train = data.train_mask.sum().item()
        print(f"  {name}: {data.num_nodes} nodes, {data.edge_index.size(1)} edges, "
              f"{n_vuln} vulnerable, {n_train} training")

    print("\n[2/6] Federated GNN training...")
    global_model = build_model(cfg_copy)
    t0 = time.time()
    trained_model, history = federated_train(global_model, chains, cfg_copy)
    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.1f}s")

    print("\n[3/6] Selecting forget nodes and computing influence zones...")
    forget_sets = {}
    for name, data in chains.items():
        fn = select_forget_nodes(data, cfg_copy["unlearning"]["forget_ratio"], seed=seed)
        forget_sets[name] = fn
        print(f"  {name}: {len(fn)} nodes to forget")

    print("\n[4/6] Running unlearning methods...")
    timing = {}

    print("  -> XFGU unlearning...")
    t0 = time.time()
    xfgu_model, zone_sizes, dp_sigma = xfgu_unlearn(
        copy.deepcopy(trained_model), chains, forget_sets, cfg_copy
    )
    timing["XFGU"] = time.time() - t0
    print(f"     Done in {timing['XFGU']:.2f}s | Zones: {zone_sizes} | DP sigma: {dp_sigma:.4f}")

    print("  -> Full retrain baseline...")
    t0 = time.time()
    retrain_model = full_retrain(None, chains, forget_sets, cfg_copy)
    timing["FullRetrain"] = time.time() - t0
    print(f"     Done in {timing['FullRetrain']:.2f}s")

    print("  -> SISA baseline...")
    t0 = time.time()
    sisa_model = sisa_unlearn(chains, forget_sets, cfg_copy)
    timing["SISA"] = time.time() - t0
    print(f"     Done in {timing['SISA']:.2f}s")

    print("  -> Naive fine-tune baseline...")
    t0 = time.time()
    naive_model = naive_finetune_unlearn(
        copy.deepcopy(trained_model), chains, forget_sets, cfg_copy
    )
    timing["NaiveFT"] = time.time() - t0
    print(f"     Done in {timing['NaiveFT']:.2f}s")

    print("  -> GraphEraser baseline...")
    t0 = time.time()
    ge_model = graph_eraser_unlearn(chains, forget_sets, cfg_copy)
    timing["GraphEraser"] = time.time() - t0
    print(f"     Done in {timing['GraphEraser']:.2f}s")

    print("  -> PAGE-FGU baseline...")
    t0 = time.time()
    page_model = page_fgu_unlearn(
        copy.deepcopy(trained_model), chains, forget_sets, cfg_copy
    )
    timing["PAGE-FGU"] = time.time() - t0
    print(f"     Done in {timing['PAGE-FGU']:.2f}s")

    print("  -> Certified removal baseline...")
    t0 = time.time()
    cert_model, cert_radii = certified_removal_unlearn(
        copy.deepcopy(trained_model), chains, forget_sets, cfg_copy
    )
    timing["CertRemoval"] = time.time() - t0
    print(f"     Done in {timing['CertRemoval']:.2f}s | Cert radii: {cert_radii}")

    print("  -> GNNDelete baseline...")
    t0 = time.time()
    gnndelete_model = gnndelete_unlearn(
        copy.deepcopy(trained_model), chains, forget_sets, cfg_copy
    )
    timing["GNNDelete"] = time.time() - t0
    print(f"     Done in {timing['GNNDelete']:.2f}s")

    print("  -> EraseRectify baseline...")
    t0 = time.time()
    erase_rectify_model = erase_rectify_unlearn(
        copy.deepcopy(trained_model), chains, forget_sets, cfg_copy
    )
    timing["EraseRectify"] = time.time() - t0
    print(f"     Done in {timing['EraseRectify']:.2f}s")

    print("  -> PAGE baseline...")
    t0 = time.time()
    page_baseline_model = page_unlearn(
        copy.deepcopy(trained_model), chains, forget_sets, cfg_copy
    )
    timing["PAGE"] = time.time() - t0
    print(f"     Done in {timing['PAGE']:.2f}s")

    print("\n[5/6] Evaluating all methods...")
    models = {
        "Original": trained_model,
        "XFGU": xfgu_model,
        "FullRetrain": retrain_model,
        "SISA": sisa_model,
        "NaiveFT": naive_model,
        "GraphEraser": ge_model,
        "PAGE-FGU": page_model,
        "CertRemoval": cert_model,
        "GNNDelete": gnndelete_model,
        "EraseRectify": erase_rectify_model,
        "PAGE": page_baseline_model,
    }

    utility_results = {}
    for method_name, model in models.items():
        utility_results[method_name] = evaluate_model(model, chains, cfg_copy, method_name)

    unlearn_quality = {}
    for method_name in UNLEARN_METHODS:
        uq = evaluate_unlearning_quality(
            models[method_name], retrain_model, trained_model,
            chains, forget_sets, cfg_copy
        )
        unlearn_quality[method_name] = uq

    print("\n[6/6] Saving checkpoints and generating plots...")
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    for method_name, model in models.items():
        ckpt_path = os.path.join(checkpoints_dir, f"{method_name}.pt")
        torch.save(model.state_dict(), ckpt_path)
    print(f"  Checkpoints saved to {checkpoints_dir}")

    plot_training_curve(history, output_dir)
    plot_utility_comparison(utility_results, output_dir)

    mia_dict = {}
    for method_name in UNLEARN_METHODS:
        mia_dict[method_name] = unlearn_quality[method_name]["mia_per_chain"]
    if mia_dict:
        plot_mia_comparison(mia_dict, output_dir)

    plot_efficiency(timing, output_dir)

    return {
        "seed": seed,
        "utility": utility_results,
        "unlearn_quality": unlearn_quality,
        "timing": timing,
        "zone_sizes": zone_sizes,
        "dp_sigma": dp_sigma,
        "train_time": train_time,
        "cert_radii": {k: float(v) for k, v in cert_radii.items()},
    }


def aggregate_results(all_results):
    agg = {}

    for method in ALL_METHODS:
        f1s, accs, aucs = [], [], []
        for r in all_results:
            if method in r["utility"]:
                f1s.append(r["utility"][method]["overall"]["f1"])
                accs.append(r["utility"][method]["overall"]["accuracy"])
                aucs.append(r["utility"][method]["overall"]["auc"])
        if f1s:
            agg[method] = {
                "f1_mean": np.mean(f1s), "f1_std": np.std(f1s),
                "acc_mean": np.mean(accs), "acc_std": np.std(accs),
                "auc_mean": np.mean(aucs), "auc_std": np.std(aucs),
            }

    for method in UNLEARN_METHODS:
        mia_accs, param_dists, advs = [], [], []
        for r in all_results:
            if method in r["unlearn_quality"]:
                for chain_name, mia in r["unlearn_quality"][method]["mia_per_chain"].items():
                    mia_accs.append(mia["mia_accuracy"])
                    advs.append(mia.get("attacker_advantage", 0.0))
                param_dists.append(r["unlearn_quality"][method]["param_distance_to_retrain"])
        if method in agg:
            if mia_accs:
                agg[method]["mia_mean"] = np.mean(mia_accs)
                agg[method]["mia_std"] = np.std(mia_accs)
            if advs:
                agg[method]["adv_mean"] = np.mean(advs)
                agg[method]["adv_std"] = np.std(advs)
            if param_dists:
                agg[method]["param_dist_mean"] = np.mean(param_dists)
                agg[method]["param_dist_std"] = np.std(param_dists)

    timing_agg = {}
    for method in TIMED_METHODS:
        times = [r["timing"][method] for r in all_results if method in r["timing"]]
        if times:
            timing_agg[method] = {"mean": np.mean(times), "std": np.std(times)}

    return agg, timing_agg


def print_final_results(agg, timing_agg):
    print("\n" + "=" * 80)
    print("FINAL AGGREGATED RESULTS (mean +/- std over seeds)")
    print("=" * 80)

    print("\n--- Utility Metrics (Test Set) ---")
    print(f"{'Method':<15} {'Accuracy':<20} {'F1-Score':<20} {'AUC-ROC':<20}")
    print("-" * 75)
    for method in ALL_METHODS:
        if method in agg:
            m = agg[method]
            print(f"{method:<15} "
                  f"{m['acc_mean']:.4f} +/- {m['acc_std']:.4f}   "
                  f"{m['f1_mean']:.4f} +/- {m['f1_std']:.4f}   "
                  f"{m['auc_mean']:.4f} +/- {m['auc_std']:.4f}")

    print("\n--- Unlearning Quality ---")
    print(f"{'Method':<15} {'MIA Acc':<15} {'Attacker Adv':<15} {'Param Dist':<20} {'Speedup':<10}")
    print("-" * 75)

    retrain_time = timing_agg.get("FullRetrain", {}).get("mean", 1.0)
    for method in UNLEARN_METHODS:
        if method in agg and "mia_mean" in agg[method]:
            m = agg[method]
            adv_str = f"{m.get('adv_mean', 0):.4f}+/-{m.get('adv_std', 0):.4f}" if "adv_mean" in m else "-"
            speedup = retrain_time / timing_agg.get(method, {}).get("mean", 1.0) if method in timing_agg else 0
            print(f"{method:<15} "
                  f"{m['mia_mean']:.4f}+/-{m['mia_std']:.4f}  "
                  f"{adv_str:<15}  "
                  f"{m['param_dist_mean']:.2f}+/-{m['param_dist_std']:.2f}        "
                  f"{speedup:.1f}x")

    print("\n--- Efficiency ---")
    print(f"{'Method':<15} {'Time (seconds)':<25}")
    print("-" * 40)
    for method in TIMED_METHODS:
        if method in timing_agg:
            t = timing_agg[method]
            print(f"{method:<15} {t['mean']:.2f} +/- {t['std']:.2f}")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    cfg = load_config(config_path)

    base_output = get_output_dir("outputs", "xfgu_exp")
    seeds = cfg["evaluation"]["seeds"]

    all_results = []
    for seed in seeds:
        seed_dir = get_output_dir(base_output, f"seed_{seed}")
        result = run_single_seed(cfg, seed, seed_dir)
        all_results.append(result)

    agg, timing_agg = aggregate_results(all_results)
    print_final_results(agg, timing_agg)

    final_output = {
        "aggregated": {},
        "timing": {},
        "per_seed": [],
    }
    for method, vals in agg.items():
        final_output["aggregated"][method] = {k: float(v) for k, v in vals.items()}
    for method, vals in timing_agg.items():
        final_output["timing"][method] = {k: float(v) for k, v in vals.items()}

    for r in all_results:
        seed_data = {"seed": r["seed"], "timing": r["timing"]}
        if "cert_radii" in r:
            seed_data["cert_radii"] = r["cert_radii"]
        final_output["per_seed"].append(seed_data)

    results_path = os.path.join(base_output, "final_results.json")
    with open(results_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    model = build_model(cfg)
    flat_timing = {m: t["mean"] for m, t in timing_agg.items()}
    comm_report = analyze_communication_cost(model, cfg, flat_timing)
    print_communication_report(comm_report)
    save_communication_report(comm_report, base_output)

    return agg, timing_agg


if __name__ == "__main__":
    main()
