"""Train and evaluate the benchmark models on UAVIDS-2025.

    python -m uavids.run                       # all ten models, seed 42
    python -m uavids.run --models RF XGBoost   # a subset
    python -m uavids.run --seeds 42 43 44      # repeat with other splits/inits
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

from .data import DEFAULT_CSV, prepare
from .models import HPARAMS, MODEL_NAMES, build


def evaluate(name, seed, data):
    X_tr, X_te, y_tr, y_te, classes, _ = data
    model = build(name, seed)
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = model.predict_proba(X_te)
    infer_ms = (time.perf_counter() - t0) / len(X_te) * 1000
    pred = proba.argmax(1)

    macro = precision_recall_fscore_support(y_te, pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_te, pred, average="weighted", zero_division=0)
    row = {
        "model": name, "seed": seed,
        "accuracy": accuracy_score(y_te, pred),
        "precision_macro": macro[0], "recall_macro": macro[1], "f1_macro": macro[2],
        "precision_weighted": weighted[0], "recall_weighted": weighted[1], "f1_weighted": weighted[2],
        "auc_roc_ovr_macro": roc_auc_score(y_te, proba, multi_class="ovr", average="macro"),
        "train_time_s": train_s, "inference_ms_per_flow": infer_ms,
    }
    extra = {
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        "per_class": classification_report(y_te, pred, target_names=classes,
                                           output_dict=True, zero_division=0),
    }
    return row, extra


def worker(args):
    """Train/evaluate ONE model in this process and dump its result as JSON."""
    data = prepare(args.csv, seed=args.seed)
    row, extra = evaluate(args.worker, args.seed, data)
    out = Path(args.out) / "runs"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.worker}_seed{args.seed}.json").write_text(
        json.dumps({"row": row, "classes": data[4], **extra}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--out", default="results")
    ap.add_argument("--worker", choices=MODEL_NAMES, help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=42, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        return worker(args)

    # Every model runs in its own process: on macOS torch and xgboost/lightgbm
    # ship conflicting OpenMP runtimes and crash/deadlock if loaded together.
    out = Path(args.out)
    (out / "confusion").mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        for name in args.models:
            subprocess.run([sys.executable, "-W", "ignore", "-m", "uavids.run", "--worker", name,
                            "--seed", str(seed), "--csv", args.csv, "--out", args.out], check=True)
            r = json.loads((out / "runs" / f"{name}_seed{seed}.json").read_text())
            row = r["row"]
            rows.append(row)
            (out / "confusion" / f"{name}_seed{seed}.json").write_text(json.dumps(
                {"classes": r["classes"], "confusion_matrix": r["confusion_matrix"],
                 "per_class": r["per_class"]}, indent=2))
            print(f"[seed {seed}] {name:9s} acc={row['accuracy']:.4f} f1_macro={row['f1_macro']:.4f} "
                  f"auc={row['auc_roc_ovr_macro']:.4f} train={row['train_time_s']:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "results_all_runs.csv", index=False)
    if len(args.seeds) > 1:
        summary = df.drop(columns="seed").groupby("model", sort=False).agg(["mean", "std"])
    else:
        summary = df.drop(columns="seed").set_index("model")
    summary.to_csv(out / "results_summary.csv")
    (out / "hyperparameters.json").write_text(json.dumps(HPARAMS, indent=2))
    print("\nSaved to", out.resolve())


if __name__ == "__main__":
    main()
