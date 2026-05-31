import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


def plot_main_results_table(results, output_dir):
    if results is None:
        return

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.axis("off")

    agg = results.get("aggregated", {})
    timing = results.get("timing", {})

    columns = ["Method", "Accuracy", "F1-Score", "AUC-ROC", "MIA Acc", "Time (s)", "Speedup"]
    data_rows = []

    retrain_time = timing.get("FullRetrain", {}).get("mean", 1.0)

    method_configs = [
        ("Original", "Original (Pre-Unlearn)"),
        ("XFGU", "XFGU (Ours)"),
        ("FullRetrain", "Full Retrain"),
        ("SISA", "SISA-Graph"),
        ("NaiveFT", "Naive Fine-tune"),
        ("GraphEraser", "GraphEraser"),
        ("PAGE-FGU", "PAGE-FGU"),
        ("CertRemoval", "Certified Removal"),
    ]

    for key, label in method_configs:
        m = agg.get(key, {})
        if not m:
            continue
        acc_str = f"{m.get('acc_mean', 0):.3f} +/- {m.get('acc_std', 0):.3f}"
        f1_str = f"{m.get('f1_mean', 0):.3f} +/- {m.get('f1_std', 0):.3f}"
        auc_str = f"{m.get('auc_mean', 0):.3f} +/- {m.get('auc_std', 0):.3f}"
        mia_str = (f"{m.get('mia_mean', 0):.3f} +/- {m.get('mia_std', 0):.3f}"
                   if "mia_mean" in m else "-")
        t = timing.get(key, {})
        t_str = f"{t.get('mean', 0):.2f} +/- {t.get('std', 0):.2f}" if t else "-"
        speedup = f"{retrain_time / t['mean']:.1f}x" if t and t.get("mean", 0) > 0 else "-"
        data_rows.append([label, acc_str, f1_str, auc_str, mia_str, t_str, speedup])

    cell_colors = [
        ["#E8F5E9"] * len(columns) if "Ours" in row[0]
        else ["#FFF3E0"] * len(columns) if "Full Retrain" in row[0]
        else ["#FFFFFF"] * len(columns)
        for row in data_rows
    ]

    table = ax.table(cellText=data_rows, colLabels=columns, loc="center",
                     cellColours=cell_colors, colColours=["#1565C0"] * len(columns))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_edgecolor("#0D47A1")
        else:
            cell.set_edgecolor("#BBDEFB")

    ax.set_title("XFGU: Experimental Results",
                 fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "results_table.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close()


def plot_ablation_line(data, x_label, title, filename, output_dir):
    if data is None:
        return

    keys = sorted(data.keys(), key=lambda k: float(k))
    x_vals = [float(k) for k in keys]
    f1_vals = [data[k]["f1"] for k in keys]
    mia_vals = [data[k]["mia_accuracy"] for k in keys]
    time_vals = [data[k]["time"] for k in keys]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(x_vals, f1_vals, "o-", color="#2E7D32", linewidth=2, markersize=8)
    axes[0].set_xlabel(x_label, fontsize=11)
    axes[0].set_ylabel("F1-Score", fontsize=11)
    axes[0].set_title("Utility (F1-Score)", fontsize=12, fontweight="bold")
    axes[0].grid(alpha=0.3)

    axes[1].plot(x_vals, mia_vals, "s-", color="#D32F2F", linewidth=2, markersize=8)
    axes[1].axhline(y=0.5, color="gray", linestyle=":", linewidth=1)
    axes[1].set_xlabel(x_label, fontsize=11)
    axes[1].set_ylabel("MIA Accuracy", fontsize=11)
    axes[1].set_title("Privacy (MIA Accuracy)", fontsize=12, fontweight="bold")
    axes[1].grid(alpha=0.3)

    axes[2].plot(x_vals, time_vals, "^-", color="#1565C0", linewidth=2, markersize=8)
    axes[2].set_xlabel(x_label, fontsize=11)
    axes[2].set_ylabel("Time (seconds)", fontsize=11)
    axes[2].set_title("Efficiency", fontsize=12, fontweight="bold")
    axes[2].grid(alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close()


def plot_dp_pareto(data, output_dir):
    if data is None:
        return

    keys = sorted(data.keys(), key=lambda k: float(k))
    epsilons = [float(k) for k in keys]
    f1_vals = [data[k]["f1"] for k in keys]
    mia_vals = [data[k]["mia_accuracy"] for k in keys]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    c1 = "#2E7D32"
    ax1.set_xlabel("Privacy Budget (epsilon)", fontsize=12)
    ax1.set_ylabel("F1-Score (Utility)", fontsize=12, color=c1)
    line1 = ax1.plot(epsilons, f1_vals, "o-", color=c1, linewidth=2.5, markersize=8, label="F1-Score")
    ax1.tick_params(axis="y", labelcolor=c1)

    ax2 = ax1.twinx()
    c2 = "#D32F2F"
    ax2.set_ylabel("MIA Accuracy (Privacy Risk)", fontsize=12, color=c2)
    line2 = ax2.plot(epsilons, mia_vals, "s--", color=c2, linewidth=2.5, markersize=8,
                     label="MIA Accuracy")
    ax2.tick_params(axis="y", labelcolor=c2)
    ax2.axhline(y=0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=11)
    ax1.set_title("Privacy-Utility Trade-off: DP Budget Analysis", fontsize=13, fontweight="bold")
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dp_pareto_front.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close()


def plot_communication_cost(comm_data, output_dir):
    if comm_data is None:
        return

    methods_data = comm_data.get("methods", {})
    methods = list(methods_data.keys())
    total_mb = [methods_data[m]["total_mb"] for m in methods]
    reductions = [methods_data[m]["comm_reduction"] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    bars = axes[0].bar(methods, total_mb, color=colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, total_mb):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].set_ylabel("Total Communication (MB)")
    axes[0].set_title("Communication Cost per Method", fontsize=12, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", alpha=0.3)

    bars2 = axes[1].bar(methods, reductions, color=colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars2, reductions):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{v:.1f}x", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1].set_ylabel("Communication Reduction (vs Retrain)")
    axes[1].set_title("Communication Savings", fontsize=12, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "communication_cost.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close()


def plot_influence_zone_visualization(cfg, output_dir):
    from data.generate_sc_graph import build_cross_chain_data
    from graphs.influence_zone import select_forget_nodes, get_influence_zone
    from utils.seed import set_seed

    set_seed(42)
    chains = build_cross_chain_data(cfg)
    chain_name = cfg["data"]["chain_names"][0]
    data = chains[chain_name]

    forget_nodes = select_forget_nodes(data, 0.05, seed=42)[:5]
    influence_nodes, _, _ = get_influence_zone(data.edge_index, forget_nodes, 2, data.num_nodes)

    n_vis = min(150, data.num_nodes)
    edge_index = data.edge_index.numpy()
    G = nx.Graph()
    for i in range(edge_index.shape[1]):
        s, d = int(edge_index[0, i]), int(edge_index[1, i])
        if s < n_vis and d < n_vis:
            G.add_edge(s, d)
    for n in range(n_vis):
        if n not in G:
            G.add_node(n)

    pos = nx.spring_layout(G, seed=42, k=0.3)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    forget_set = set(forget_nodes)
    influence_set = set(influence_nodes.numpy().tolist())

    for ax_idx, (ax, title) in enumerate(zip(axes, ["Before Unlearning", "After XFGU Unlearning"])):
        node_colors, node_sizes = [], []
        for n in G.nodes():
            if n in forget_set:
                node_colors.append("#D32F2F" if ax_idx == 0 else "#E0E0E0")
                node_sizes.append(120 if ax_idx == 0 else 40)
            elif n in influence_set:
                node_colors.append("#FF8F00" if ax_idx == 0 else "#4CAF50")
                node_sizes.append(60)
            elif data.y[n].item() > 0:
                node_colors.append("#7B1FA2")
                node_sizes.append(40)
            else:
                node_colors.append("#1565C0")
                node_sizes.append(30)

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, edge_color="#9E9E9E", width=0.5)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.8)

        legend_elements = [
            mpatches.Patch(color="#D32F2F", label="Forget Nodes"),
            mpatches.Patch(color="#FF8F00", label="Influence Zone (2-hop)"),
            mpatches.Patch(color="#7B1FA2", label="Vulnerable Contracts"),
            mpatches.Patch(color="#1565C0", label="Safe Contracts"),
        ]
        if ax_idx == 1:
            legend_elements[0] = mpatches.Patch(color="#E0E0E0", label="Forgotten")
            legend_elements[1] = mpatches.Patch(color="#4CAF50", label="Influence Zone (Repaired)")
        ax.legend(handles=legend_elements, loc="lower left", fontsize=8, framealpha=0.9)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")

    fig.suptitle(f"XFGU: Influence Zone on {chain_name} Chain",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "influence_zone_viz.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close()


def plot_architecture_diagram(output_dir):
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    chain_configs = [
        (1.5, 7, "ETH Chain", "#1565C0"),
        (6, 7, "BSC Chain", "#2E7D32"),
        (10.5, 7, "Polygon Chain", "#E65100"),
    ]
    for x, y, label, color in chain_configs:
        rect = mpatches.FancyBboxPatch((x, y), 3, 1.8, boxstyle="round,pad=0.1",
                                       facecolor=color, alpha=0.2,
                                       edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 1.5, y + 0.9, label, ha="center", va="center", fontsize=10, fontweight="bold")

    box_y = 4.5
    steps = [
        (0.5, box_y, "1. Influence\nZone Discovery\n(L-hop BFS)", "#7B1FA2"),
        (4, box_y, "2. Reverse\nGradient Update\n(on forget set)", "#C62828"),
        (7.5, box_y, "3. DP Noise\nCalibration\n(Gaussian)", "#00695C"),
        (11, box_y, "4. Retain-set\nFine-tuning\n(utility recovery)", "#1565C0"),
    ]
    for x, y, label, color in steps:
        rect = mpatches.FancyBboxPatch((x, y), 3, 1.5, boxstyle="round,pad=0.1",
                                       facecolor=color, alpha=0.15,
                                       edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 1.5, y + 0.75, label, ha="center", va="center", fontsize=9, fontweight="bold")

    for i in range(len(steps) - 1):
        x1 = steps[i][0] + 3
        x2 = steps[i + 1][0]
        y_mid = box_y + 0.75
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    agg_rect = mpatches.FancyBboxPatch((5, 1.5), 6, 1.5, boxstyle="round,pad=0.15",
                                       facecolor="#FF8F00", alpha=0.2,
                                       edgecolor="#FF8F00", linewidth=2)
    ax.add_patch(agg_rect)
    ax.text(8, 2.25,
            "5. Cross-Chain FedAvg Aggregation\n(Weighted by retain-set size)",
            ha="center", va="center", fontsize=11, fontweight="bold")

    for x_start in [3, 7.5, 12]:
        ax.annotate("", xy=(8, 3.0), xytext=(x_start, box_y),
                    arrowprops=dict(arrowstyle="->", color="#FF8F00", lw=1.5, linestyle="--"))

    ax.text(8, 0.3, "Global Unlearned Model", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#D32F2F",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFCDD2", edgecolor="#D32F2F"))
    ax.annotate("", xy=(8, 0.7), xytext=(8, 1.5),
                arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=2))
    ax.set_title(
        "XFGU Architecture: Cross-Chain Federated Graph Unlearning Pipeline",
        fontsize=14, fontweight="bold", pad=10,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "architecture_diagram.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close()


def plot_architecture_comparison(gcn_results, gat_results, output_dir):
    if gcn_results is None or gat_results is None:
        return

    methods = ["XFGU", "FullRetrain", "NaiveFT", "PAGE-FGU", "CertRemoval"]
    labels = ["XFGU\n(Ours)", "FullRetrain", "NaiveFT", "PAGE-FGU", "CertRemoval"]
    metrics = ["f1_mean", "acc_mean", "mia_mean"]
    metric_labels = ["F1-Score", "Accuracy", "MIA Accuracy"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    x = np.arange(len(methods))
    width = 0.35

    for ax_idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        gcn_vals, gat_vals = [], []
        gcn_errs, gat_errs = [], []
        for m in methods:
            g = gcn_results.get("aggregated", {}).get(m, {})
            gcn_vals.append(g.get(metric, 0))
            gcn_errs.append(g.get(metric.replace("mean", "std"), 0))
            a = gat_results.get("aggregated", {}).get(m, {})
            gat_vals.append(a.get(metric, 0))
            gat_errs.append(a.get(metric.replace("mean", "std"), 0))

        axes[ax_idx].bar(x - width / 2, gcn_vals, width, yerr=gcn_errs,
                         label="GCN", color="#1565C0", alpha=0.8,
                         capsize=3, edgecolor="black", linewidth=0.5)
        axes[ax_idx].bar(x + width / 2, gat_vals, width, yerr=gat_errs,
                         label="GAT", color="#D32F2F", alpha=0.8,
                         capsize=3, edgecolor="black", linewidth=0.5)

        if metric == "mia_mean":
            axes[ax_idx].axhline(y=0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)
            axes[ax_idx].annotate("Ideal (random)", xy=(0, 0.5), fontsize=8, color="gray")

        axes[ax_idx].set_ylabel(metric_label, fontsize=11)
        axes[ax_idx].set_title(metric_label, fontsize=12, fontweight="bold")
        axes[ax_idx].set_xticks(x)
        axes[ax_idx].set_xticklabels(labels, fontsize=8)
        axes[ax_idx].legend(fontsize=10)
        axes[ax_idx].grid(axis="y", alpha=0.3)

    fig.suptitle("Architecture Robustness: GCN vs GAT Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "architecture_comparison.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close()


def plot_radar_chart(results, output_dir):
    if results is None:
        return

    agg = results.get("aggregated", {})
    timing = results.get("timing", {})

    methods_to_plot = ["XFGU", "FullRetrain", "NaiveFT", "PAGE-FGU", "CertRemoval"]
    colors = ["#2E7D32", "#1565C0", "#FF8F00", "#7B1FA2", "#D32F2F"]
    retrain_time = timing.get("FullRetrain", {}).get("mean", 1.0)

    categories = ["F1-Score", "Accuracy", "AUC-ROC", "Privacy\n(1-MIA)", "Speedup\n(norm)"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for method, color in zip(methods_to_plot, colors):
        m = agg.get(method, {})
        if not m:
            continue
        t = timing.get(method, {}).get("mean", retrain_time)
        speedup_norm = min(1.0, (retrain_time / t) / 15.0)
        mia_val = 1.0 - m.get("mia_mean", 0.5) if "mia_mean" in m else 0.5

        values = [
            m.get("f1_mean", 0), m.get("acc_mean", 0), m.get("auc_mean", 0),
            mia_val, speedup_norm,
        ]
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=2, label=method, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title("XFGU: Multi-Dimensional Performance Radar",
                 fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radar_chart.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close()
