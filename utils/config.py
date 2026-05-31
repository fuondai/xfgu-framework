"""Configuration loading and merging utilities."""

import yaml
import os


def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_output_dir(base="outputs", tag="default"):
    d = os.path.join(base, tag)
    os.makedirs(d, exist_ok=True)
    return d
