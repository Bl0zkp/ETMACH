from __future__ import annotations

import pandas as pd


def bureau_features(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """
    bureau + bureau_balance:
    - bureau_balance agrega por SK_ID_BUREAU
    - se une a bureau
    - se agrega por SK_ID_CURR
    """
    b = bureau.copy()
    bb = bureau_balance.copy()

    if "SK_ID_BUREAU" not in b.columns or "SK_ID_CURR" not in b.columns:
        raise ValueError("bureau debe tener SK_ID_BUREAU y SK_ID_CURR.")
    if "SK_ID_BUREAU" not in bb.columns:
        raise ValueError("bureau_balance debe tener SK_ID_BUREAU.")

    # 1) Agregar bureau_balance por bureau
    bb_grp = bb.groupby("SK_ID_BUREAU", observed=True)
    bb_feat = pd.DataFrame(index=bb_grp.size().index)
    bb_feat["bb_count"] = bb_grp.size()

    if "MONTHS_BALANCE" in bb.columns:
        s = bb_grp["MONTHS_BALANCE"]
        bb_feat["bb_months_min"] = s.min()
        bb_feat["bb_months_max"] = s.max()

    if "STATUS" in bb.columns:
        # Conteos de estados
        counts = (
            bb.pivot_table(
                index="SK_ID_BUREAU",
                columns="STATUS",
                values="MONTHS_BALANCE" if "MONTHS_BALANCE" in bb.columns else bb.columns[0],
                aggfunc="count",
                fill_value=0,
            )
        )
        counts.columns = [f"bb_status__{str(c).lower()}" for c in counts.columns]
        bb_feat = bb_feat.join(counts, how="left")

    bb_feat = bb_feat.reset_index()

    # 2) Unir a bureau
    b2 = b.merge(bb_feat, on="SK_ID_BUREAU", how="left")

    # 3) Agregar por cliente
    grp = b2.groupby("SK_ID_CURR", observed=True)
    out = pd.DataFrame(index=grp.size().index)
    out["bureau_count"] = grp.size()

   # Selección segura: SOLO columnas numéricas reales
    candidate_cols = [c for c in b2.columns if c.startswith(("AMT_", "DAYS_", "CREDIT_"))]
    candidate_cols += [c for c in ["bb_count", "bb_months_min", "bb_months_max"] if c in b2.columns]
    candidate_cols = list(dict.fromkeys(candidate_cols))

    num_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(b2[c])]

    for c in num_cols:
        s = grp[c]
        out[f"bureau_{c}_mean"] = s.mean()
        out[f"bureau_{c}_max"] = s.max()
        out[f"bureau_{c}_min"] = s.min()


    # Si existe CREDIT_ACTIVE, conteos
    if "CREDIT_ACTIVE" in b2.columns:
        counts = (
            b2.pivot_table(
                index="SK_ID_CURR",
                columns="CREDIT_ACTIVE",
                values="SK_ID_BUREAU",
                aggfunc="count",
                fill_value=0,
            )
        )
        counts.columns = [f"bureau_active__{str(c).lower().replace(' ', '_')}" for c in counts.columns]
        out = out.join(counts, how="left")

    out = out.reset_index()
    return out
