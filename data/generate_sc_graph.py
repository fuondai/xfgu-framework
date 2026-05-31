"""Cross-chain edge construction and stratified splitting (Section III-B)."""

import numpy as np
import torch


def add_cross_chain_edges(chains, cross_chain_ratio, seed=42, tau_cross=0.70):
    """Add cross-chain proxy edges based on cosine similarity (Definition 1).

    For each pair of chains, compute pairwise cosine similarity between node
    features and add proxy edges where similarity exceeds tau_cross.
    The number of edges is capped at cross_chain_ratio * min(n_a, n_b).

    Args:
        chains: Dictionary mapping chain names to PyG Data objects.
        cross_chain_ratio: Maximum fraction of cross-chain edges relative
            to the smaller chain.
        seed: Random seed for tie-breaking.
        tau_cross: Cosine similarity threshold (paper: 0.70, Section III-B).
    """
    rng = np.random.RandomState(seed)
    chain_names = list(chains.keys())

    for i, name_a in enumerate(chain_names):
        for j, name_b in enumerate(chain_names):
            if i >= j:
                continue
            data_a = chains[name_a]
            data_b = chains[name_b]
            n_a = data_a.num_nodes
            n_b = data_b.num_nodes
            n_cross = max(1, int(min(n_a, n_b) * cross_chain_ratio))

            # L2-normalise features for cosine similarity
            x_a = data_a.x / (data_a.x.norm(dim=1, keepdim=True) + 1e-8)
            x_b = data_b.x / (data_b.x.norm(dim=1, keepdim=True) + 1e-8)

            # Compute cosine similarity matrix (n_a x n_b)
            sim = torch.mm(x_a, x_b.t())

            # Find pairs exceeding threshold
            above = (sim >= tau_cross).nonzero(as_tuple=False)

            if len(above) > 0:
                # Sort by similarity descending, take top n_cross
                sim_vals = sim[above[:, 0], above[:, 1]]
                order = sim_vals.argsort(descending=True)
                selected = above[order[:n_cross]]
                src_a = selected[:, 0].numpy()
                src_b = selected[:, 1].numpy()
            else:
                # Fall back to the top n_cross pairs by similarity
                flat_idx = sim.flatten().argsort(descending=True)[:n_cross]
                src_a = (flat_idx // n_b).numpy()
                src_b = (flat_idx % n_b).numpy()

            # Add proxy edges within each chain to represent cross-chain links
            ei_a = chains[name_a].edge_index
            proxy_src_a = rng.choice(n_a, size=len(src_a), replace=True)
            proxy_a = torch.tensor(
                [src_a.tolist() + proxy_src_a.tolist(),
                 proxy_src_a.tolist() + src_a.tolist()],
                dtype=torch.long,
            )
            chains[name_a].edge_index = torch.cat([ei_a, proxy_a], dim=1)

            ei_b = chains[name_b].edge_index
            proxy_src_b = rng.choice(n_b, size=len(src_b), replace=True)
            proxy_b = torch.tensor(
                [src_b.tolist() + proxy_src_b.tolist(),
                 proxy_src_b.tolist() + src_b.tolist()],
                dtype=torch.long,
            )
            chains[name_b].edge_index = torch.cat([ei_b, proxy_b], dim=1)

            chains[name_a].cross_chain_edges = getattr(chains[name_a], "cross_chain_edges", [])
            chains[name_a].cross_chain_edges.append((name_b, src_a, src_b))
            chains[name_b].cross_chain_edges = getattr(chains[name_b], "cross_chain_edges", [])
            chains[name_b].cross_chain_edges.append((name_a, src_b, src_a))

    return chains


def split_data(data, train_ratio, val_ratio, seed=42):
    """Stratified per-class train/val/test split (Section III-B)."""
    rng = np.random.RandomState(seed + 7919)
    n = data.num_nodes
    labels = data.y.numpy()

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    for c in np.unique(labels):
        c_idx = np.where(labels == c)[0]
        rng.shuffle(c_idx)
        nc = len(c_idx)
        nt = int(nc * train_ratio)
        nv = int(nc * val_ratio)
        train_mask[c_idx[:nt]] = True
        val_mask[c_idx[nt:nt + nv]] = True
        test_mask[c_idx[nt + nv:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def build_cross_chain_data(cfg):
    """Build the cross-chain graph dataset from the SmartBugs corpus."""
    from data.dataset import build_cross_chain_dataset
    return build_cross_chain_dataset(cfg)
