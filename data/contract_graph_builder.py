"""Solidity feature extraction and graph construction utilities."""

import re
import os
import numpy as np
import torch
from torch_geometric.data import Data


VULN_CATEGORIES = {
    "reentrancy": 1,
    "access_control": 2,
    "arithmetic": 3,
    "unchecked_low_level_calls": 4,
    "time_manipulation": 5,
    "denial_of_service": 5,
    "bad_randomness": 5,
    "front_running": 5,
    "short_addresses": 5,
}

VULN_NAMES = ["safe", "reentrancy", "access_control", "arithmetic",
              "unchecked_low_level", "other"]


def extract_solidity_features(source_code, feature_dim=64):
    features = np.zeros(feature_dim, dtype=np.float32)

    functions = re.findall(r'function\s+\w+', source_code)
    features[0] = len(functions)

    modifiers = re.findall(r'modifier\s+\w+', source_code)
    features[1] = len(modifiers)

    state_vars = re.findall(r'(uint|int|address|bool|string|bytes|mapping)\s', source_code)
    features[2] = len(state_vars)

    features[3] = source_code.count('.call')
    features[4] = source_code.count('.delegatecall')
    features[5] = source_code.count('.send(')
    features[6] = source_code.count('.transfer(')
    features[7] = source_code.count('selfdestruct')

    features[8] = len(re.findall(r'for\s*\(', source_code))
    features[9] = len(re.findall(r'while\s*\(', source_code))

    features[10] = source_code.count('require(')
    features[11] = source_code.count('assert(')
    features[12] = source_code.count('revert(')

    features[13] = source_code.count('msg.sender')
    features[14] = source_code.count('msg.value')
    features[15] = source_code.count('tx.origin')
    features[16] = source_code.count('block.timestamp')
    features[17] = source_code.count('block.number')

    features[18] = source_code.count('onlyOwner')
    features[19] = len(re.findall(r'public\s', source_code))
    features[20] = len(re.findall(r'external\s', source_code))
    features[21] = len(re.findall(r'internal\s', source_code))
    features[22] = len(re.findall(r'private\s', source_code))

    features[23] = len(re.findall(r'emit\s+\w+', source_code))
    features[24] = len(re.findall(r'event\s+\w+', source_code))

    features[25] = source_code.count('SafeMath')
    features[26] = len(re.findall(r'[\+\-\*\/]\s*=', source_code))
    features[27] = len(re.findall(r'overflow|underflow', source_code, re.IGNORECASE))

    features[28] = len(re.findall(r'mapping\s*\(', source_code))
    features[29] = source_code.count('balance')
    features[30] = source_code.count('withdraw')
    features[31] = source_code.count('deposit')

    call_before_state = 0
    lines = source_code.split('\n')
    in_function = False
    saw_call = False
    for line in lines:
        stripped = line.strip()
        if re.match(r'function\s+', stripped):
            in_function = True
            saw_call = False
        if in_function:
            if '.call' in stripped or '.send(' in stripped:
                saw_call = True
            if saw_call and ('=' in stripped and '.call' not in stripped):
                call_before_state += 1
                saw_call = False
    features[32] = call_before_state

    features[33] = len(source_code)
    features[34] = len(lines)
    features[35] = len(re.findall(r'interface\s+\w+', source_code))
    features[36] = len(re.findall(r'library\s+\w+', source_code))
    features[37] = len(re.findall(r'contract\s+\w+', source_code))
    features[38] = len(re.findall(r'struct\s+\w+', source_code))
    features[39] = len(re.findall(r'enum\s+\w+', source_code))

    imports = re.findall(r'import\s+', source_code)
    features[40] = len(imports)
    features[41] = len(re.findall(r'pragma\s+solidity', source_code))

    features[42] = source_code.count('payable')
    features[43] = source_code.count('view ')
    features[44] = source_code.count('pure ')
    features[45] = len(re.findall(r'returns?\s*\(', source_code))

    features[46] = len(re.findall(r'new\s+\w+', source_code))
    features[47] = source_code.count('abi.encode')
    features[48] = source_code.count('keccak256')
    features[49] = source_code.count('sha3')

    if feature_dim > 50:
        func_bodies = re.findall(r'function\s+\w+[^{]*\{([^}]*)\}', source_code, re.DOTALL)
        total_complexity = 0
        for body in func_bodies:
            total_complexity += body.count('if') + body.count('for') + body.count('while')
        features[50] = total_complexity
        features[51] = len(func_bodies)
        if len(func_bodies) > 0:
            features[52] = total_complexity / len(func_bodies)
        features[53] = max(len(b) for b in func_bodies) if func_bodies else 0

    norms = np.linalg.norm(features)
    if norms > 0:
        features = features / norms

    return features


def detect_vulnerability_type(source_code, directory_name=""):
    dir_lower = directory_name.lower().replace("_", " ").replace("-", " ")
    for vuln_name, vuln_id in VULN_CATEGORIES.items():
        if vuln_name.replace("_", " ") in dir_lower:
            return vuln_id

    annotations = re.findall(r'//\s*<yes>\s*<report>\s*(\w+)', source_code, re.IGNORECASE)
    for ann in annotations:
        ann_lower = ann.lower()
        for vuln_name, vuln_id in VULN_CATEGORIES.items():
            if vuln_name.replace("_", "") in ann_lower.replace("_", ""):
                return vuln_id

    return 0


def build_similarity_edges(features_matrix, threshold=0.7, max_neighbors=3):
    n = features_matrix.shape[0]
    norms = np.linalg.norm(features_matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = features_matrix / norms

    src, dst = [], []
    batch_size = 500
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        block = normalized[start:end] @ normalized.T
        for local_i in range(end - start):
            global_i = start + local_i
            sims = block[local_i].copy()
            sims[global_i] = -1
            top_indices = np.argsort(sims)[-max_neighbors:]
            for j in top_indices:
                if sims[j] > threshold:
                    src.append(global_i)
                    dst.append(j)

    if len(src) < n:
        rng = np.random.RandomState(42)
        for i in range(n):
            has_edge = any(s == i for s in src)
            if not has_edge:
                j = rng.randint(0, n)
                while j == i:
                    j = rng.randint(0, n)
                src.append(i)
                dst.append(j)

    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    return edge_index


def build_graph_from_contracts(features, labels, threshold=0.5, max_neighbors=3):
    edge_index = build_similarity_edges(features, threshold=threshold, max_neighbors=max_neighbors)
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_nodes = len(labels)
    return data
