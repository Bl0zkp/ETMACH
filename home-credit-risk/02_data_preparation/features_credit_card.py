from __future__ import annotations

import pandas as pd


def credit_card_features(credit_card: pd.DataFrame, prev_map: pd.DataFrame) -> pd.DataFrame:
    """
    credit_card_balance -> features por SK_ID_CURR (via SK_ID_PREV).
    """
    df = credit_card.copy()

    if "SK_ID_PREV" not in df.columns:
        raise ValueError("credit_card_balance debe tener SK_ID_PREV.")

    df = df.merge(prev_map, on="SK_ID_PREV", how="left", suffixes=("", "_prev"))

    # Normalizar SK_ID_CURR
    if "SK_ID_CURR" not in df.columns:
        if "SK_ID_CURR_prev" in df.columns:
            df["SK_ID_CURR"] = df["SK_ID_CURR_prev"]
        else:
            raise ValueError(
                f"No se pudo generar SK_ID_CURR en CC. Columnas post-merge: {list(df.columns)}"
            )

    df = df[df["SK_ID_CURR"].notna()].copy()

    grp = df.groupby("SK_ID_CURR", observed=True)
    out = pd.DataFrame(index=grp.size().index)
    out["cc_count"] = grp.size()

    # Columnas numéricas típicas
    num_like = [c for c in df.columns if c.startswith("AMT_") or c.startswith("CNT_") or c in ["MONTHS_BALANCE"]]
    for c in num_like:
        s = grp[c]
        out[f"cc_{c}_mean"] = s.mean()
        out[f"cc_{c}_max"] = s.max()
        out[f"cc_{c}_min"] = s.min()

    return out.reset_index()
