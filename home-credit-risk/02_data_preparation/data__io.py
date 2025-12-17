from __future__ import annotations

from pathlib import Path
import pandas as pd


RAW_DIR_DEFAULT = Path("data/raw")
PROCESSED_DIR_DEFAULT = Path("data/processed")


def read_parquet(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path.resolve()}")
    return pd.read_parquet(path)


def load_raw_tables(raw_dir: str | Path = RAW_DIR_DEFAULT) -> dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir)

    files = {
        "application": raw_dir / "application_.parquet",
        "previous_application": raw_dir / "previous_application.parquet",
        "bureau": raw_dir / "bureau.parquet",
        "bureau_balance": raw_dir / "bureau_balance.parquet",
        "credit_card_balance": raw_dir / "credit_card_balance.parquet",
        "installments_payments": raw_dir / "installments_payments.parquet",
        "pos_cash_balance": raw_dir / "POS_CASH_balance.parquet",
        # "columns_desc": raw_dir / "HomeCredit_columns_description.parquet",  # opcional
    }

    tables: dict[str, pd.DataFrame] = {}
    missing = []
    for k, p in files.items():
        if p.exists():
            tables[k] = read_parquet(p)
        else:
            missing.append(str(p))
    if missing:
        raise FileNotFoundError(
            "Faltan archivos en data/raw/. Revisa estos:\n" + "\n".join(missing)
        )

    return tables


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)
