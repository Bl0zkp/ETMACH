from __future__ import annotations
import pandas as pd


def pos_cash_features(pos_cash: pd.DataFrame, prev_map: pd.DataFrame) -> pd.DataFrame:
    df = pos_cash.copy()

    if "SK_ID_PREV" not in df.columns:
        raise ValueError("POS_CASH_balance debe tener SK_ID_PREV.")

    df = df.merge(prev_map, on="SK_ID_PREV", how="left", suffixes=("", "_prev"))

    # Normalizar SK_ID_CURR
    if "SK_ID_CURR" not in df.columns:
        if "SK_ID_CURR_prev" in df.columns:
            df["SK_ID_CURR"] = df["SK_ID_CURR_prev"]
        else:
            raise ValueError(
                f"No se pudo generar SK_ID_CURR en POS. Columnas post-merge: {list(df.columns)}"
            )

    df = df[df["SK_ID_CURR"].notna()].copy()

    grp = df.groupby("SK_ID_CURR", observed=True)
    out = pd.DataFrame(index=grp.size().index)
    out["pos_count"] = grp.size()

    for c in ["MONTHS_BALANCE", "SK_DPD", "SK_DPD_DEF"]:
        if c in df.columns:
            s = grp[c]
            out[f"pos_{c}_mean"] = s.mean()
            out[f"pos_{c}_max"] = s.max()
            out[f"pos_{c}_min"] = s.min()

    if "NAME_CONTRACT_STATUS" in df.columns:
        counts = (
            df.pivot_table(
                index="SK_ID_CURR",
                columns="NAME_CONTRACT_STATUS",
                values="SK_ID_PREV",
                aggfunc="count",
                fill_value=0,
            )
        )
        counts.columns = [f"pos_status__{str(c).lower().replace(' ', '_')}" for c in counts.columns]
        out = out.join(counts, how="left")

    return out.reset_index()

