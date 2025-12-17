from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.json"
THRESH_PATH = ARTIFACTS_DIR / "thresholds.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def decision_from_prob(p: float, approve_th: float, reject_th: float) -> str:
    if p < approve_th:
        return "APROBAR"
    if p >= reject_th:
        return "RECHAZAR"
    return "REVISIÓN MANUAL"


app = FastAPI(title="Home Credit Risk API", version="1.0")


class EvaluateRiskRequest(BaseModel):
    # Recibe cualquier set de campos del solicitante en JSON
    data: Dict[str, Any]


@app.on_event("startup")
def _load_artifacts():
    global model, schema, thresholds
    if not MODEL_PATH.exists():
        raise RuntimeError("No existe artifacts/model.joblib. Entrena primero el modelo.")
    if not SCHEMA_PATH.exists():
        raise RuntimeError("No existe artifacts/feature_schema.json. Construye dataset primero.")

    model = joblib.load(MODEL_PATH)
    schema = load_json(SCHEMA_PATH)

    if THRESH_PATH.exists():
        thresholds = load_json(THRESH_PATH)
    else:
        thresholds = {"approve_th": 0.20, "reject_th": 0.50}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/evaluate_risk")
def evaluate_risk(req: EvaluateRiskRequest):
    try:
        feature_cols = schema["feature_cols"]

        # Construir df de 1 fila
        x = pd.DataFrame([req.data])

        # Agregar columnas faltantes
        for c in feature_cols:
            if c not in x.columns:
                x[c] = pd.NA

        # Eliminar columnas extra
        x = x[feature_cols]

        # 🔑 CLAVE: convertir pd.NA a np.nan (sklearn-friendly)
        x = x.replace({pd.NA: np.nan})

        proba = float(model.predict_proba(x)[:, 1][0])
        decision = decision_from_prob(
            proba,
            thresholds["approve_th"],
            thresholds["reject_th"],
        )

        return {
            "probabilidad_incumplimiento": proba,
            "decision_sugerida": decision,
            "umbral_aprobar": thresholds["approve_th"],
            "umbral_rechazar": thresholds["reject_th"],
        }

    except Exception as e:
        return {
            "error": "Error al evaluar el riesgo",
            "detalle": str(e),
        }
