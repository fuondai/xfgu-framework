"""Dataset loading and preprocessing (Section III-B).

Loads Solidity contracts from a SmartBugs export (a CSV placed under archive/)
when available, otherwise from the curated .sol corpus under
datasets/smartbugs-curated/dataset/. Per-chain graphs are assembled by
stratified sampling without replacement so that the class balance matches the
benchmark prevalence reported in Table II.
"""

import logging
import os

import numpy as np
import pandas as pd

from data.contract_graph_builder import (
    extract_solidity_features,
    detect_vulnerability_type,
    build_graph_from_contracts,
)
from data.generate_sc_graph import split_data, add_cross_chain_edges

logger = logging.getLogger(__name__)


ARCHIVE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "archive")
CURATED_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "datasets", "smartbugs-curated", "dataset"
)


def _resolve_archive_csv(cfg):
    """Return the absolute path to the configured archive CSV, or None."""
    name = cfg["data"].get("archive_csv")
    if not name:
        return None
    path = os.path.abspath(os.path.join(ARCHIVE_DIR, name))
    return path if os.path.exists(path) else None


def parse_archive_contracts(csv_path, feature_dim=64):
    """Parse a SmartBugs CSV into feature vectors and binary labels.

    The CSV must provide a source-code column and a label column. A contract is
    labelled benign when its label is the dataset's clean tag and vulnerable
    otherwise. Rows tagged ``unknow`` or with negligible source are dropped.
    """
    df = pd.read_csv(csv_path)
    src_col = next(c for c in df.columns if c.lower() in ("source_code", "source", "code"))
    label_col = next(c for c in df.columns if c.lower() in ("labels", "label", "class", "target"))
    df = df.dropna(subset=[src_col, label_col])

    features_list = []
    labels_list = []
    for source, raw_label in zip(df[src_col].astype(str), df[label_col].astype(str)):
        tag = raw_label.strip().lower()
        if tag in ("unknow", "unknown", "nan", ""):
            continue
        if len(source.strip()) < 20:
            continue
        label = 0 if tag in ("clean", "safe", "0", "none") else 1
        features_list.append(extract_solidity_features(source, feature_dim=feature_dim))
        labels_list.append(label)

    features = np.asarray(features_list, dtype=np.float32)
    labels = np.asarray(labels_list, dtype=np.int64)
    logger.info(
        "Loaded %d contracts from %s (%d benign, %d vulnerable)",
        len(labels), os.path.basename(csv_path),
        int((labels == 0).sum()), int((labels == 1).sum()),
    )
    return features, labels


def parse_curated_contracts(dataset_root, feature_dim=64):
    """Parse the curated .sol corpus into feature vectors and binary labels.

    Walks <dataset_root>/<category>/*.sol, derives the vulnerability category
    from the directory name, and maps it to a binary benign/vulnerable label.
    """
    dataset_root = os.path.abspath(dataset_root)
    features_list = []
    labels_list = []

    for category in sorted(os.listdir(dataset_root)):
        cat_dir = os.path.join(dataset_root, category)
        if not os.path.isdir(cat_dir):
            continue
        for fname in sorted(os.listdir(cat_dir)):
            if not fname.endswith(".sol"):
                continue
            with open(os.path.join(cat_dir, fname), "r", errors="ignore") as f:
                source = f.read()
            if len(source.strip()) < 20:
                continue
            vuln_type = detect_vulnerability_type(source, directory_name=category)
            label = 0 if vuln_type == 0 else 1
            features_list.append(extract_solidity_features(source, feature_dim=feature_dim))
            labels_list.append(label)

    features = np.asarray(features_list, dtype=np.float32)
    labels = np.asarray(labels_list, dtype=np.int64)
    logger.info(
        "Loaded %d contracts from curated corpus (%d benign, %d vulnerable)",
        len(labels), int((labels == 0).sum()), int((labels == 1).sum()),
    )
    return features, labels


def _per_chain_targets(chain_sizes, chain_vuln_counts, n_vuln_avail, n_benign_avail):
    """Resolve per-chain (vulnerable, benign) draw counts from the available pools.

    The configured Table II counts are honoured when both class pools are large
    enough. Otherwise every chain is scaled down by the single largest factor
    that the scarcer pool permits, so the per-chain prevalence is preserved.
    """
    req_vuln = sum(chain_vuln_counts)
    req_benign = sum(s - v for s, v in zip(chain_sizes, chain_vuln_counts))

    scale = 1.0
    if req_vuln > n_vuln_avail:
        scale = min(scale, n_vuln_avail / req_vuln)
    if req_benign > n_benign_avail:
        scale = min(scale, n_benign_avail / req_benign)

    targets = []
    for size, vuln in zip(chain_sizes, chain_vuln_counts):
        v = int(round(vuln * scale))
        b = int(round((size - vuln) * scale))
        targets.append((v, b))

    if scale < 1.0:
        logger.info(
            "Available pools (%d vulnerable, %d benign) below Table II totals "
            "(%d, %d); scaling all chains by %.3f to preserve prevalence",
            n_vuln_avail, n_benign_avail, req_vuln, req_benign, scale,
        )
    return targets


def build_cross_chain_dataset(cfg):
    """Build per-chain graphs from the SmartBugs corpus (Section III-B)."""
    data_cfg = cfg["data"]
    feature_dim = data_cfg["feature_dim"]
    chain_names = data_cfg["chain_names"]
    chain_sizes = data_cfg["chain_sizes"]
    train_ratio = data_cfg["train_ratio"]
    val_ratio = data_cfg["val_ratio"]
    cross_chain_ratio = data_cfg.get("cross_chain_ratio", 0.0)
    base_seed = cfg["seed"]
    edge_threshold = data_cfg.get("edge_similarity_threshold", 0.6)
    max_neighbors = data_cfg.get("max_neighbors", 3)

    csv_path = _resolve_archive_csv(cfg)
    if csv_path is not None:
        features, labels = parse_archive_contracts(csv_path, feature_dim=feature_dim)
    else:
        features, labels = parse_curated_contracts(CURATED_ROOT, feature_dim=feature_dim)

    # Per-chain vulnerable counts default to the overall prevalence when Table II
    # counts are not supplied explicitly.
    vuln_ratio = data_cfg.get("vuln_ratio", 0.386)
    chain_vuln_counts = data_cfg.get(
        "chain_vuln_counts", [int(round(s * vuln_ratio)) for s in chain_sizes]
    )

    rng = np.random.RandomState(base_seed)
    vuln_pool = rng.permutation(np.where(labels == 1)[0]).tolist()
    benign_pool = rng.permutation(np.where(labels == 0)[0]).tolist()

    targets = _per_chain_targets(
        chain_sizes, chain_vuln_counts, len(vuln_pool), len(benign_pool)
    )

    chains = {}
    v_ptr = b_ptr = 0
    for i, name in enumerate(chain_names):
        n_vuln, n_benign = targets[i]
        v_idx = vuln_pool[v_ptr:v_ptr + n_vuln]
        b_idx = benign_pool[b_ptr:b_ptr + n_benign]
        v_ptr += n_vuln
        b_ptr += n_benign

        sel = np.array(v_idx + b_idx, dtype=np.int64)
        rng.shuffle(sel)
        chain_feat = features[sel]
        chain_labels = labels[sel]

        data = build_graph_from_contracts(
            chain_feat, chain_labels, threshold=edge_threshold, max_neighbors=max_neighbors
        )
        data = split_data(data, train_ratio, val_ratio, seed=base_seed + i)
        data.chain_name = name

        n_v = int((data.y == 1).sum().item())
        logger.info(
            "%s: %d nodes, %d edges, %d vulnerable (%.1f%%)",
            name, data.num_nodes, data.edge_index.size(1),
            n_v, 100.0 * n_v / max(1, data.num_nodes),
        )
        chains[name] = data

    if cross_chain_ratio > 0:
        tau_cross = data_cfg.get("cross_chain_threshold", 0.70)
        chains = add_cross_chain_edges(chains, cross_chain_ratio, seed=base_seed, tau_cross=tau_cross)

    return chains
