import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_utility_comparison(results_dict, output_dir):
    methods = list(results_dict.keys())
    chains = [k for k in results_dict[methods[0]] if k != "overall"]
    chains.append("overall")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, metric in enumerate(["accuracy", "f1", "auc"]):
        ax = axes[idx]
        x = np.arange(len(chains))
        width = 0.8 / len(methods)

        for i, method in enumerate(methods):
            vals = [results_dict[method].get(c, {}).get(metric, 0) for c in chains]
            ax.bar(x + i * width, vals, width, label=method, alpha=0.85)

        ax.set_xlabel("Chain")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} Comparison")
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(chains, rotation=15)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "utility_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_mia_comparison(mia_dict, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    methods = list(mia_dict.keys())
    chain_names = set()
    for m in methods:
        chain_names.update(mia_dict[m].keys())
    chain_names = sorted(chain_names)

    x = np.arange(len(chain_names))
    width = 0.8 / len(methods)

    for i, method in enumerate(methods):
        vals = [mia_dict[method].get(c, {}).get("mia_accuracy", 0.5) for c in chain_names]
        ax.bar(x + i * width, vals, width, label=method, alpha=0.85)

    ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1.5, label="Random Guess (0.5)")
    ax.set_xlabel("Chain")
    ax.set_ylabel("MIA Accuracy")
    ax.set_title("Membership Inference Attack Accuracy (lower = better unlearning)")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(chain_names)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "mia_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_efficiency(timing_dict, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = list(timing_dict.keys())
    times = [timing_dict[m] for m in methods]

    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    bars = ax.bar(methods, times, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{t:.1f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Wall-clock Time (seconds)")
    ax.set_title("Unlearning Efficiency Comparison")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "efficiency_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_training_curve(history, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(history["rounds"], history["train_loss"], "b-", linewidth=1.5)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Training Loss")
    ax.set_title("Federated Training Loss")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for name in history["val_metrics"]:
        f1s = [m["f1"] for m in history["val_metrics"][name]]
        ax.plot(history["rounds"], f1s, label=name, linewidth=1.5)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Validation F1")
    ax.set_title("Per-Chain Validation F1")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path
