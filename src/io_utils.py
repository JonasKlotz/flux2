from pathlib import Path
from typing import Union, Optional

import pandas as pd


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_df_if_exists(path: Union[str, Path]) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        if p.suffix.lower() in [".parquet"]:
            return pd.read_parquet(p)
        if p.suffix.lower() in [".csv"]:
            return pd.read_csv(p)
        if p.suffix.lower() in [".pkl", ".pickle"]:
            return pd.read_pickle(p)
        raise ValueError(f"Unsupported dataframe format: {p.suffix}")
    except Exception as e:
        raise RuntimeError(f"Failed to load cached dataframe at {p}: {e}") from e


def save_df(df: pd.DataFrame, path: Union[str, Path]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".parquet":
        df.to_parquet(p, index=False)
    elif p.suffix.lower() == ".csv":
        df.to_csv(p, index=False)
    elif p.suffix.lower() in [".pkl", ".pickle"]:
        df.to_pickle(p)
    else:
        raise ValueError(f"Unsupported dataframe format: {p.suffix}")
