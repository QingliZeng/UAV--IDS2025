"""The ten models benchmarked on UAVIDS-2025.

Classical models follow scikit-learn / XGBoost / LightGBM APIs.  The three deep
models (DNN, CNN, LSTM) live in ``deep.py``: small PyTorch networks wrapped in a scikit-learn
style ``fit`` / ``predict_proba`` interface so every model is evaluated by the
same code path.

All hyper-parameters are collected in ``hparams.HPARAMS`` so they are easy to audit.
"""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from .hparams import HPARAMS


# --------------------------------------------------------------------------- #
def build(name: str, seed: int = 42):
    h = HPARAMS
    if name == "LR":
        return LogisticRegression(random_state=seed, **h["LR"])
    if name == "SVM":
        return SVC(random_state=seed, **h["SVM"])
    if name == "DT":
        return DecisionTreeClassifier(random_state=seed, **h["DT"])
    if name == "KNN":
        return KNeighborsClassifier(n_jobs=-1, **h["KNN"])
    if name == "RF":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **h["RF"])
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=seed, n_jobs=-1, eval_metric="mlogloss", **h["XGBoost"])
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, **h["LightGBM"])
    if name in ("DNN", "CNN", "LSTM"):
        from .deep import TorchClassifier  # imported lazily, see deep.py
        return TorchClassifier(name, seed=seed)
    raise KeyError(name)


MODEL_NAMES = ["LR", "SVM", "DT", "RF", "KNN", "XGBoost", "LightGBM", "DNN", "CNN", "LSTM"]
