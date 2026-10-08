"""Generate figures from the CSV and from results/ (no hard-coded numbers).

    python scripts/plot_results.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uavids.models import MODEL_NAMES  # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def dataset_overview():
    df = pd.read_csv(ROOT / "data" / "UAVIDS-2025.csv")
    counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(counts.index, counts.values, color="#69b3a2", edgecolor="#444", width=0.55)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / len(df):.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, counts.max() * 1.2)
    ax.set_ylabel("Flows")
    ax.set_title(f"UAVIDS-2025 class distribution (N = {len(df):,})")
    ax.grid(axis="y", ls="--", alpha=.5)
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(FIG / "dataset_overview.png", dpi=200)
    plt.close(fig)


def confusion_grid(seed=42):
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    for ax, name in zip(axes.ravel(), MODEL_NAMES):
        p = ROOT / "results" / "confusion" / f"{name}_seed{seed}.json"
        if not p.exists():
            ax.axis("off")
            continue
        d = json.loads(p.read_text())
        cm = np.array(d["confusion_matrix"])
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        short = [c.replace(" Traffic", "").replace(" Attack", "") for c in d["classes"]]
        ax.set_xticks(range(len(short)), short, rotation=45, ha="right")
        ax.set_yticks(range(len(short)), short)
        for i in range(len(short)):
            for j in range(len(short)):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                        color="white" if norm[i, j] > .5 else "black", fontsize=8)
        ax.set_title(name)
    fig.suptitle("Row-normalised confusion matrices (20% stratified test split)")
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrices.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    dataset_overview()
    confusion_grid()
    print("figures written to", FIG)
