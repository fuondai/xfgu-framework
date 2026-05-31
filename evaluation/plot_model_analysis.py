import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_weight_heatmap(cfg, output_dir):
    from data.generate_sc_graph import build_cross_chain_data
    from models.gnn import build_model
    from federated.server import federated_train
    from graphs.influence_zone import select_forget_nodes
    from unlearning.xfgu import xfgu_unlearn
    from utils.seed import set_seed

    set_seed(42)
    chains = build_cross_chain_data(cfg)
    global_model = build_model(cfg)
    trained_model, _ = federated_train(global_model, chains, cfg)

    forget_sets = {name: select_forget_nodes(data, 0.05, seed=42)
                   for name, data in chains.items()}
    unlearned_model, _, _ = xfgu_unlearn(
        copy.deepcopy(trained_model), chains, forget_sets, cfg
    )

    orig_state = trained_model.state_dict()
    unl_state = unlearned_model.state_dict()
    weight_key = "convs.0.lin.weight"
    orig_w = orig_state[weight_key].cpu().numpy()
    unl_w = unl_state[weight_key].cpu().numpy()
    diff_w = np.abs(orig_w - unl_w)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    vmax = max(np.abs(orig_w).max(), np.abs(unl_w).max())
    slice_size = min(48, orig_w.shape[0], orig_w.shape[1])

    im0 = axes[0].imshow(orig_w[:slice_size, :slice_size], cmap="RdBu_r",
                         aspect="auto", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Before Unlearning\n(Original Weights)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Output Dimension")
    axes[0].set_ylabel("Input Dimension")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(unl_w[:slice_size, :slice_size], cmap="RdBu_r",
                         aspect="auto", vmin=-vmax, vmax=vmax)
    axes[1].set_title("After XFGU Unlearning\n(Modified Weights)",
                      fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Output Dimension")
    axes[1].set_ylabel("Input Dimension")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff_w[:slice_size, :slice_size], cmap="hot", aspect="auto")
    axes[2].set_title("Weight Difference\n(|Before - After|)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Output Dimension")
    axes[2].set_ylabel("Input Dimension")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle("GNN Weight Heatmap: Effect of XFGU Unlearning on Layer 1",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "weight_heatmap.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close()


def plot_forgetting_effectiveness(cfg, output_dir):
    from data.generate_sc_graph import build_cross_chain_data
    from models.gnn import build_model
    from federated.server import federated_train
    from graphs.influence_zone import select_forget_nodes
    from unlearning.xfgu import xfgu_unlearn
    from utils.seed import set_seed

    set_seed(42)
    chains = build_cross_chain_data(cfg)
    global_model = build_model(cfg)
    trained_model, _ = federated_train(global_model, chains, cfg)

    forget_sets = {name: select_forget_nodes(data, 0.05, seed=42)
                   for name, data in chains.items()}
    unlearned_model, _, _ = xfgu_unlearn(
        copy.deepcopy(trained_model), chains, forget_sets, cfg
    )

    chain_name = list(chains.keys())[0]
    data = chains[chain_name].to(cfg["device"])
    fn = forget_sets[chain_name]

    trained_model.eval()
    unlearned_model.eval()
    with torch.no_grad():
        out_orig = trained_model(data.x, data.edge_index)
        out_unl = unlearned_model(data.x, data.edge_index)
        conf_orig = F.softmax(out_orig, dim=1).max(dim=1)[0].cpu().numpy()
        conf_unl = F.softmax(out_unl, dim=1).max(dim=1)[0].cpu().numpy()

    forget_mask = np.zeros(data.num_nodes, dtype=bool)
    forget_mask[fn] = True
    retain_mask = data.train_mask.cpu().numpy() & ~forget_mask

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(conf_orig[forget_mask], bins=20, alpha=0.7, label="Before Unlearning",
                 color="#D32F2F", edgecolor="black", linewidth=0.5)
    axes[0].hist(conf_unl[forget_mask], bins=20, alpha=0.7, label="After XFGU",
                 color="#4CAF50", edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("Prediction Confidence", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title("Confidence on Forget Nodes", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(conf_orig[retain_mask], bins=20, alpha=0.7, label="Before Unlearning",
                 color="#1565C0", edgecolor="black", linewidth=0.5)
    axes[1].hist(conf_unl[retain_mask], bins=20, alpha=0.7, label="After XFGU",
                 color="#FF8F00", edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Prediction Confidence", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title("Confidence on Retain Nodes", fontsize=12, fontweight="bold")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"Forgetting Effectiveness: Confidence Distribution ({chain_name} Chain)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "forgetting_effectiveness.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close()
