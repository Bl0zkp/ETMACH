from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_io import load_raw_tables, save_parquet, ensure_dir
from features_previous import build_prev_map, previous_application_features
from features_installments import installments_features
from features_pos_cash import pos_cash_features
from features_credit_card import credit_card_features
from features_bureau import bureau_features


ARTIFACTS_DIR = Path("artifacts")
PROCESSED_DIR = Path("data/processed")


def main() -> None:
    tables = load_raw_tables("data/raw")

    app = tables["application"].copy()
    prev = tables["previous_application"]
    bureau = tables["bureau"]
    bureau_balance = tables["bureau_balance"]
    cc = tables["credit_card_balance"]
    inst = tables["installments_payments"]
    pos = tables["pos_cash_balance"]

    if "SK_ID_CURR" not in app.columns:
        raise ValueError("application_.parquet debe tener SK_ID_CURR.")

    # Base de clientes
    base = app[["SK_ID_CURR"]].drop_duplicates().copy()

    # Features
    prev_map = build_prev_map(prev)

    f_prev = previous_application_features(prev)
    f_inst = installments_features(inst, prev_map)
    f_pos = pos_cash_features(pos, prev_map)
    f_cc = credit_card_features(cc, prev_map)
    f_bureau = bureau_features(bureau, bureau_balance)

    # Join
    df = base.merge(app, on="SK_ID_CURR", how="left")
    for feat in [f_prev, f_inst, f_pos, f_cc, f_bureau]:
        df = df.merge(feat, on="SK_ID_CURR", how="left")

    # Guardar dataset procesado
    ensure_dir(PROCESSED_DIR)
    out_path = PROCESSED_DIR / "dataset.parquet"
    save_parquet(df, out_path)

    # Guardar schema de features (para API)
    ensure_dir(ARTIFACTS_DIR)
    target_col = "TARGET" if "TARGET" in df.columns else None
    feature_cols = [c for c in df.columns if c not in ["SK_ID_CURR", target_col] and c != "TARGET"]

    schema = {
        "id_col": "SK_ID_CURR",
        "target_col": "TARGET" if "TARGET" in df.columns else None,
        "feature_cols": feature_cols,
    }
    (ARTIFACTS_DIR / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    print(f"OK: dataset guardado en {out_path}")
    print(f"OK: schema guardado en {ARTIFACTS_DIR / 'feature_schema.json'}")
    if "TARGET" not in df.columns:
        print("AVISO: No encontré columna TARGET en application_.parquet (¿dataset sin etiqueta?).")


if __name__ == "__main__":
    main()
