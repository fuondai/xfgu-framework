# XFGU: Influence-Zone Differential Privacy for Cross-Chain Federated Graph Unlearning

A research implementation of the XFGU framework for privacy-preserving cross-chain
federated graph unlearning on smart contract vulnerability detection graphs.

## Project Structure

```
XFGU/
├── configs/
│   ├── default.yaml              # Paper-faithful 5-chain experiment (GCN)
│   ├── gat.yaml                  # GAT backbone variant
│   └── graphsage.yaml            # GraphSAGE backbone variant
├── data/
│   ├── generate_sc_graph.py      # Stratified splitting + cross-chain edges (Section III-B)
│   ├── dataset.py                # SmartBugs loader (curated corpus / CSV export)
│   └── contract_graph_builder.py # Solidity feature extraction + graph construction
├── graphs/
│   └── influence_zone.py         # L-hop influence zone discovery
├── models/
│   └── gnn.py                    # GCN, GAT, and GraphSAGE architectures
├── federated/
│   ├── client.py                 # Local training and evaluation
│   └── server.py                 # FedAvg aggregation and training loop
├── unlearning/
│   └── xfgu.py                   # Core XFGU algorithm
├── baselines/
│   ├── retrain.py                # Full retraining baseline
│   ├── sisa.py                   # SISA-Graph baseline
│   ├── naive_finetune.py         # Naive fine-tuning baseline
│   ├── graph_eraser.py           # GraphEraser baseline
│   ├── page_fgu.py               # PAGE-FGU baseline
│   ├── page.py                   # PAGE baseline
│   ├── gnndelete.py              # GNNDelete baseline
│   ├── erase_rectify.py          # Erase-Rectify baseline
│   └── certified_removal.py      # Certified removal baseline
├── evaluation/
│   ├── evaluate.py               # Utility and unlearning quality metrics
│   ├── mia.py                    # Membership Inference Attack
│   ├── communication.py          # Communication cost analysis
│   ├── plotting.py               # Experiment result plots
│   ├── plot_generators.py        # Static result plots
│   └── plot_model_analysis.py    # Model-training analysis plots
├── utils/
│   ├── config.py                 # Config loader
│   ├── metrics.py                # Macro-F1, AUC, parameter distance
│   └── seed.py                   # Reproducibility
├── scripts/
│   ├── run_full_experiment.py    # Main experiment runner
│   ├── run_ablation.py           # Ablation study runner
│   └── generate_all_plots.py     # Visualization suite
├── tests/
│   ├── test_data.py              # Data pipeline unit tests
│   ├── test_model.py             # GNN model unit tests
│   └── test_integration.py       # End-to-end pipeline test on a small subset
├── requirements.txt              # Pinned dependencies
├── setup.sh                      # One-shot environment setup
└── README.md
```

## Data

The benchmark is built from Solidity smart contracts labelled by vulnerability
category.

- **SmartBugs export (preferred)**: a labelled CSV placed under `archive/` and
  named by `data.archive_csv` in the config (default `smartbug-scrawid-irfuzz.csv`,
  columns `source_code`, `label`). A contract is benign when its label is the clean
  tag and vulnerable otherwise.
- **Curated corpus (fallback)**: contracts under
  `datasets/smartbugs-curated/dataset/`, organised by vulnerability category, used
  automatically when the configured CSV is absent.

Per-chain graphs are assembled by stratified sampling without replacement so that
the class balance matches the benchmark prevalence in Table II: each chain draws its
configured `chain_vuln_counts[i]` vulnerable contracts and the remaining
`chain_sizes[i] - chain_vuln_counts[i]` benign contracts from disjoint pools. With
the default CSV this reproduces the Table II distribution (6,847 contracts, 38.6%
overall vulnerability prevalence, per-chain 32.8%-41.1%). If a pool is too small to
meet those counts, every chain is scaled down by a single factor that preserves the
prevalence.

To reproduce the full 5-chain, 6,847-contract scale, the referenced SmartBugs CSV
must be present under `archive/`. The curated corpus alone is smaller and will be
partitioned at its available size.

## Environment Setup

```bash
bash setup.sh
source .venv/bin/activate
```

Or manually:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Experiments

Main experiment (GCN, 5 chains, 5 seeds):
```bash
python scripts/run_full_experiment.py configs/default.yaml
```

GAT / GraphSAGE backbone variants:
```bash
python scripts/run_full_experiment.py configs/gat.yaml
python scripts/run_full_experiment.py configs/graphsage.yaml
```

Ablation studies:
```bash
python scripts/run_ablation.py configs/default.yaml
```

Generate figures:
```bash
python scripts/generate_all_plots.py
```

Outputs are written to `outputs/xfgu_exp/` and `outputs/xfgu_plots/`.

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests cover Solidity feature extraction, stratified splitting, cross-chain edge
construction, GNN forward passes, and an end-to-end run of the data pipeline
(training, XFGU unlearning, and evaluation) on a small subset.

## Algorithm Overview

XFGU performs cross-chain federated graph unlearning in 5 steps:

1. **Influence Zone Discovery**: BFS-based L-hop neighborhood extraction for forget nodes
2. **Reverse Gradient Update**: Gradient ascent on forget set to erase learned information
3. **DP Noise Calibration**: Gaussian mechanism with gradient clipping for (epsilon, delta)-DP
4. **Retain-set Fine-tuning**: Utility recovery via fine-tuning on retained data
5. **Cross-chain FedAvg Aggregation**: Weighted model averaging across chains

## Configuration

Edit `configs/default.yaml` to customize:
- Chain sizes, feature dimensions, cross-chain edge thresholds
- GNN architecture (GCN/GAT/GraphSAGE), layers, hidden dimensions
- Federated rounds, local epochs, learning rates
- Unlearning parameters: L-hop, reverse steps, DP budget
- SISA shards, evaluation seeds
