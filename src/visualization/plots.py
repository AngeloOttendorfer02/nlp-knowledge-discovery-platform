"""Reusable plotting helpers for reports and notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence


def save_bar_chart(values: Mapping[str, float], output_path: str | Path, title: str = "") -> Path:
    """Save a simple bar chart and return the output path."""
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(values.keys()), list(values.values()))
    ax.set_ylabel("Value")
    if title:
        ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_scatter(x: Sequence[float], y: Sequence[float], output_path: str | Path, title: str = "") -> Path:
    """Save a simple scatter plot and return the output path."""
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=20)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
