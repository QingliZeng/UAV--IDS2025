"""All model hyper-parameters in one place (easy to audit)."""

HPARAMS = {
    "LR": dict(max_iter=1000),
    "SVM": dict(kernel="rbf", C=1.0, gamma="scale", probability=True),
    "DT": dict(),
    "KNN": dict(n_neighbors=5),
    "RF": dict(n_estimators=100),
    "XGBoost": dict(n_estimators=100, max_depth=6, learning_rate=0.3),
    "LightGBM": dict(n_estimators=100, learning_rate=0.1, num_leaves=31),
    # deep models
    "deep": dict(epochs=50, batch_size=256, lr=1e-3, weight_decay=0.0,
                 val_fraction=0.1, patience=5, dropout=0.2),
}
