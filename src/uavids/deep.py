"""PyTorch models (DNN / CNN / LSTM) with a scikit-learn style interface.

Imported lazily by ``models.build`` only for the deep models: on macOS, torch and
xgboost/lightgbm ship different OpenMP runtimes that crash or deadlock when
loaded into the same process, so ``run.py`` runs every model in its own process.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .hparams import HPARAMS

# --------------------------------------------------------------------------- #
class _DNN(nn.Module):
    def __init__(self, n_feat, n_cls, p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_cls),
        )

    def forward(self, x):
        return self.net(x)


class _CNN(nn.Module):
    """1-D CNN over the feature vector (features treated as a length-F signal)."""

    def __init__(self, n_feat, n_cls, p):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(p),
            nn.Linear(64 * (n_feat // 2), 64), nn.ReLU(),
            nn.Linear(64, n_cls),
        )

    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1)))


class _LSTM(nn.Module):
    """LSTM that reads the F features as a length-F sequence of scalars."""

    def __init__(self, n_feat, n_cls, p):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(64, n_cls))

    def forward(self, x):
        _, (h, _) = self.lstm(x.unsqueeze(-1))
        return self.head(h[-1])


_ARCH = {"DNN": _DNN, "CNN": _CNN, "LSTM": _LSTM}


class TorchClassifier:
    def __init__(self, arch: str, seed: int = 42, **hp):
        self.arch, self.seed, self.hp = arch, seed, {**HPARAMS["deep"], **hp}
        self.history_: list[dict] = []

    def fit(self, X, y):
        hp = self.hp
        torch.manual_seed(self.seed)
        rng = np.random.RandomState(self.seed)
        n_cls = int(y.max()) + 1
        idx = rng.permutation(len(X))
        n_val = int(len(X) * hp["val_fraction"])
        va, tr = idx[:n_val], idx[n_val:]
        Xt, yt = torch.from_numpy(X[tr]), torch.from_numpy(y[tr]).long()
        Xv, yv = torch.from_numpy(X[va]), torch.from_numpy(y[va]).long()

        self.model = _ARCH[self.arch](X.shape[1], n_cls, hp["dropout"])
        opt = torch.optim.Adam(self.model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        loss_fn = nn.CrossEntropyLoss()
        best, best_state, bad = float("inf"), None, 0
        for ep in range(hp["epochs"]):
            self.model.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), hp["batch_size"]):
                b = perm[i:i + hp["batch_size"]]
                opt.zero_grad()
                loss_fn(self.model(Xt[b]), yt[b]).backward()
                opt.step()
            self.model.eval()
            with torch.no_grad():
                vl = loss_fn(self.model(Xv), yv).item()
            self.history_.append({"epoch": ep + 1, "val_loss": vl})
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= hp["patience"]:
                    break
        self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                out.append(torch.softmax(self.model(torch.from_numpy(X[i:i + 4096])), 1).numpy())
        return np.vstack(out)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


