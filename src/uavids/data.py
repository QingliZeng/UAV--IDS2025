"""Data loading and preprocessing for UAVIDS-2025.

Protocol (paper, Sec. IV-A): numerical features are normalised and the data is
split 80:20 with stratification.  Normalisation statistics are fitted on the
training portion only, so no test information leaks into preprocessing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DEFAULT_CSV = Path(__file__).resolve().parents[2] / "data" / "UAVIDS-2025.csv"
LABEL = "label"

# Columns that are identifiers / constants and are NOT used as features.
#  - FlowID   : the CSV is ordered by class, so FlowID alone predicts the label
#               (RF reaches ~99.98 % accuracy with it).  Never use it.
#  - SrcAddr, DstAddr : simulation node addresses (scenario specific).
#  - SrcPort, DstPort : only 2 distinct values each in the whole dataset.
#  - Protocol : constant ("UDP") for every record.
DROP_COLUMNS = ["FlowID", "SrcAddr", "DstAddr", "SrcPort", "DstPort", "Protocol"]


def load_raw(csv_path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DROP_COLUMNS + [LABEL]]


def prepare(csv_path: str | Path = DEFAULT_CSV, test_size: float = 0.2, seed: int = 42):
    """Return (X_train, X_test, y_train, y_test, class_names, feature_names).

    X are standardised float32 arrays, y are integer-encoded labels.
    """
    df = load_raw(csv_path)
    feats = feature_columns(df)
    le = LabelEncoder().fit(df[LABEL])
    y = le.transform(df[LABEL])
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[feats].to_numpy(dtype=np.float64), y,
        test_size=test_size, stratify=y, random_state=seed,
    )
    scaler = StandardScaler().fit(X_tr)
    return (
        scaler.transform(X_tr).astype(np.float32),
        scaler.transform(X_te).astype(np.float32),
        y_tr, y_te, list(le.classes_), feats,
    )
