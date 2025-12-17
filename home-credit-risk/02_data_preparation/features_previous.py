from __future__ import annotations

import numpy as np
import pandas as pd


def build_prev_map(previous_application: pd.DataFrame) -> pd.DataFrame:
    """
    Construye un mapa robusto SK_ID_PREV -> SK_ID_CURR
    """
    required = {"SK_ID_PREV", "SK_ID_CURR"}
    if not required.issubset(previous_application.columns):
        raise ValueError(
            "previous_application debe contener SK_ID_PREV y SK_ID_CURR"
        )

    prev_map = (
        previous_application[["SK_ID_PREV", "SK_ID_CURR"]]
        .dropna()
        .drop_duplicates()
    )

    return prev_map



def previous_application_features(previous_application: pd.DataFrame) -> pd.DataFrame:
    """Agregaciones por cliente desde previous_application."""
    df = previous_application.copy()

    if "SK_ID_CURR" not in df.columns:
        raise ValueError("previous_application debe tener SK_ID_CURR.")

    # Contadores simples por cliente
    grp = df.groupby("SK_ID_CURR", observed=True)

    out = pd.DataFrame(index=grp.size().index)
    out["prev_app_count"] = grp.size()

    # Ejemplos robustos (si existen columnas típicas)
    num_cols = [c for c in df.columns if c.startswith("AMT_") or c.startswith("DAYS_")]
    for c in num_cols:
        s = grp[c]
        out[f"prev_{c}_mean"] = s.mean()
        out[f"prev_{c}_max"] = s.max()
        out[f"prev_{c}_min"] = s.min()

    # Si existe NAME_CONTRACT_STATUS, hacemos conteos por estado
    if "NAME_CONTRACT_STATUS" in df.columns:
        counts = (
            df.pivot_table(
                index="SK_ID_CURR",
                columns="NAME_CONTRACT_STATUS",
                values="SK_ID_PREV" if "SK_ID_PREV" in df.columns else df.columns[0],
                aggfunc="count",
                fill_value=0,
            )
        )
        counts.columns = [f"prev_status__{str(c).lower().replace(' ', '_')}" for c in counts.columns]
        out = out.join(counts, how="left")

    out = out.reset_index()
    return out
