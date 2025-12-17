from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix


ARTIFACTS_DIR = Path("artifacts")
DATASET_PATH = Path("data/processed/dataset.parquet")


def decision_from_prob(p: float, approve_th: float = 0.20, reject_th: float = 0.50) -> str:
    if p < approve_th:
        return "APROBAR"
    if p >= reject_th:
        return "RECHAZAR"
    return "REVISIÓN MANUAL"


def main() -> None:
    model_path = ARTIFACTS_DIR / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("No existe artifacts/model.joblib. Ejecuta: python 03_modeling/train.py")

    df = pd.read_parquet(DATASET_PATH)
    if "TARGET" not in df.columns:
        raise ValueError("No existe TARGET. No puedo evaluar sin etiqueta.")

    y = df["TARGET"].astype(int)
    X = df.drop(columns=["TARGET"])
    if "SK_ID_CURR" in X.columns:
        X = X.drop(columns=["SK_ID_CURR"])

    model = joblib.load(model_path)
    proba = model.predict_proba(X)[:, 1]

    roc = roc_auc_score(y, proba)
    pr = average_precision_score(y, proba)

    # Umbrales negocio (puedes ajustarlos)
    approve_th = 0.20
    reject_th = 0.50

    decisions = [decision_from_prob(float(p), approve_th, reject_th) for p in proba]
    y_hat = (proba >= reject_th).astype(int)  # "RECHAZAR" como positivo (default alto)

    cm = confusion_matrix(y, y_hat).tolist()

    report = {
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "approve_threshold": approve_th,
        "reject_threshold": reject_th,
        "confusion_matrix_at_reject_threshold": cm,
        "decision_counts": pd.Series(decisions).value_counts().to_dict(),
    }

    (ARTIFACTS_DIR / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ARTIFACTS_DIR / "thresholds.json").write_text(
        json.dumps({"approve_th": approve_th, "reject_th": reject_th}, indent=2),
        encoding="utf-8",
    )

    print("OK: reporte guardado en artifacts/evaluation_report.json")
    print(report)


if __name__ == "__main__":
    main()
