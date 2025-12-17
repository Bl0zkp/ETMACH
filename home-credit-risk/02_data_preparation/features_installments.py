from __future__ import annotations

import numpy as np
import pandas as pd


def installments_features(installments: pd.DataFrame, prev_map: pd.DataFrame) -> pd.DataFrame:
    """
    installments_payments -> features por SK_ID_CURR.
    Usa previous_application como puente.
    """
    df = installments.copy()

    if "SK_ID_PREV" not in df.columns:
        raise ValueError("installments_payments debe tener SK_ID_PREV.")

    df = df.merge(prev_map, on="SK_ID_PREV", how="left", suffixes=("", "_prev"))

    # Si por alguna razón quedó con sufijo, lo normalizamos
    if "SK_ID_CURR" not in df.columns:
        if "SK_ID_CURR_prev" in df.columns:
            df["SK_ID_CURR"] = df["SK_ID_CURR_prev"]
        else:
            raise ValueError(
                f"No se pudo generar SK_ID_CURR. Columnas disponibles post-merge: {list(df.columns)}"
            )


    df = df[df["SK_ID_CURR"].notna()].copy()

    # ----- FEATURES -----
    if "AMT_PAYMENT" in df.columns and "AMT_INSTALMENT" in df.columns:
        df["inst_pay_ratio"] = df["AMT_PAYMENT"] / df["AMT_INSTALMENT"].replace(0, pd.NA)
        df["inst_pay_diff"] = df["AMT_PAYMENT"] - df["AMT_INSTALMENT"]

    if "DAYS_ENTRY_PAYMENT" in df.columns and "DAYS_INSTALMENT" in df.columns:
        df["inst_days_late"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
        df["inst_is_late"] = (df["inst_days_late"] > 0).astype("int8")

    grp = df.groupby("SK_ID_CURR", observed=True)

    out = pd.DataFrame(index=grp.size().index)
    out["inst_count"] = grp.size()

    for c in [
        "AMT_PAYMENT",
        "AMT_INSTALMENT",
        "inst_pay_ratio",
        "inst_pay_diff",
        "inst_days_late",
        "inst_is_late",
    ]:
        if c in df.columns:
            s = grp[c]
            out[f"inst_{c}_mean"] = s.mean()
            out[f"inst_{c}_max"] = s.max()
            out[f"inst_{c}_sum"] = s.sum()

    return out.reset_index()
