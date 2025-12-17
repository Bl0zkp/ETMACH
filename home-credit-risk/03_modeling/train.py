from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


ARTIFACTS_DIR = Path("artifacts")
PROCESSED_PATH = Path("data/processed/dataset.parquet")


def main() -> None:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"No existe {PROCESSED_PATH}. Ejecuta primero: python 02_data_preparation/build_dataset.py"
        )

    df = pd.read_parquet(PROCESSED_PATH)

    if "TARGET" not in df.columns:
        raise ValueError("No existe TARGET en el dataset. Necesitas application_train con TARGET para entrenar.")

    y = df["TARGET"].astype(int)
    X = df.drop(columns=["TARGET"])

    # Separar columnas
    id_col = "SK_ID_CURR"
    if id_col in X.columns:
        X = X.drop(columns=[id_col])

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop",
    )

    # Modelo baseline sólido + class_weight para desbalance
    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=None,
        class_weight="balanced",
        solver="lbfgs",
    )

    model = Pipeline(steps=[("pre", pre), ("clf", clf)])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)[:, 1]

    roc = roc_auc_score(y_val, proba)
    pr = average_precision_score(y_val, proba)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS_DIR / "model.joblib")

    metrics = {"roc_auc": float(roc), "pr_auc": float(pr)}
    (ARTIFACTS_DIR / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("OK: modelo guardado en artifacts/model.joblib")
    print("Métricas validación:", metrics)


if __name__ == "__main__":
    main()
