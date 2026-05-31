import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json

from utils.config import load_config
from evaluation.plot_generators import (
    plot_main_results_table,
    plot_ablation_line,
    plot_dp_pareto,
    plot_communication_cost,
    plot_influence_zone_visualization,
    plot_architecture_diagram,
    plot_architecture_comparison,
    plot_radar_chart,
)
from evaluation.plot_model_analysis import (
    plot_weight_heatmap,
    plot_forgetting_effectiveness,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "xfgu_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def main():
    print("Generating plots...")

    cfg = load_config(os.path.join(PROJECT_DIR, "configs", "default.yaml"))

    exp_results = load_json(os.path.join(PROJECT_DIR, "outputs", "xfgu_exp", "final_results.json"))
    gat_results = load_json(os.path.join(PROJECT_DIR, "outputs", "xfgu_gat_exp", "final_results.json"))
    ablation_dir = os.path.join(PROJECT_DIR, "outputs", "ablation")
    abl_lhop = load_json(os.path.join(ablation_dir, "ablation_lhop.json"))
    abl_forget = load_json(os.path.join(ablation_dir, "ablation_forget_ratio.json"))
    abl_chains = load_json(os.path.join(ablation_dir, "ablation_num_chains.json"))
    abl_steps = load_json(os.path.join(ablation_dir, "ablation_reverse_steps.json"))
    abl_dp = load_json(os.path.join(ablation_dir, "ablation_dp_epsilon.json"))
    comm_data = load_json(
        os.path.join(PROJECT_DIR, "outputs", "xfgu_exp", "communication_cost.json")
    )

    plot_main_results_table(exp_results, OUTPUT_DIR)
    print("  [1/13] Results table")

    plot_influence_zone_visualization(cfg, OUTPUT_DIR)
    print("  [2/13] Influence zone visualization")

    plot_architecture_diagram(OUTPUT_DIR)
    print("  [3/13] Architecture diagram")

    plot_ablation_line(abl_lhop, "L-hop", "Ablation: L-hop Variation", "ablation_lhop.png", OUTPUT_DIR)
    print("  [4/13] Ablation L-hop")

    plot_ablation_line(abl_forget, "Forget Ratio", "Ablation: Forget Ratio Variation",
                       "ablation_forget_ratio.png", OUTPUT_DIR)
    print("  [5/13] Ablation forget ratio")

    plot_ablation_line(abl_chains, "K (chains)", "Ablation: Number of Chains",
                       "ablation_num_chains.png", OUTPUT_DIR)
    print("  [6/13] Ablation num chains")

    plot_ablation_line(abl_steps, "Reverse Steps", "Ablation: Reverse Gradient Steps",
                       "ablation_reverse_steps.png", OUTPUT_DIR)
    print("  [7/13] Ablation reverse steps")

    plot_dp_pareto(abl_dp, OUTPUT_DIR)
    print("  [8/13] DP Pareto front")

    plot_communication_cost(comm_data, OUTPUT_DIR)
    print("  [9/13] Communication cost")

    plot_weight_heatmap(cfg, OUTPUT_DIR)
    print("  [10/13] Weight heatmap")

    plot_forgetting_effectiveness(cfg, OUTPUT_DIR)
    print("  [11/13] Forgetting effectiveness")

    if gat_results is None:
        print("  [12/13] Architecture comparison (skipped - no GAT results)")
    else:
        plot_architecture_comparison(exp_results, gat_results, OUTPUT_DIR)
        print("  [12/13] Architecture comparison (GCN vs GAT)")

    plot_radar_chart(exp_results, OUTPUT_DIR)
    print("  [13/13] Radar chart")

    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

