"""Dataset loading, validation and splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


class DataValidationError(RuntimeError):
    """Raised when the incoming dataset does not match the expected contract."""


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the RiskRadar dataset and assert its schema."""
    path = Path(path) if path else C.RAW_DATASET
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the CSV in the data/ directory."
        )

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in [*C.RAW_FEATURE_COLUMNS, C.TARGET] if c not in df.columns]
    if missing:
        raise DataValidationError(f"Dataset is missing required columns: {missing}")

    unknown_targets = set(df[C.TARGET].unique()) - set(C.CLASS_ORDER)
    if unknown_targets:
        raise DataValidationError(f"Unexpected target labels: {unknown_targets}")

    return df


def data_quality_report(df: pd.DataFrame) -> dict:
    """Summarise the health of the dataset — surfaced in the UI and the report."""
    numeric = df.select_dtypes(include=[np.number])
    counts = df[C.TARGET].value_counts()
    return {
        "rows": len(df),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "constant_columns": [c for c in df.columns if df[c].nunique() <= 1],
        "class_distribution": {k: int(v) for k, v in counts.items()},
        "imbalance_ratio": round(float(counts.max() / counts.min()), 3),
        "numeric_columns": int(numeric.shape[1]),
        "categorical_columns": int(df.shape[1] - numeric.shape[1]),
    }


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Separate raw features from the encoded target.

    ``Area_ID`` is deliberately dropped: it is unique per row and would let a
    tree memorise individual records instead of learning generalisable structure.
    """
    X = df[C.RAW_FEATURE_COLUMNS].copy()
    y = df[C.TARGET].map(C.CLASS_TO_INT).to_numpy()
    return X, y


def stratified_split(
    X: pd.DataFrame, y: np.ndarray, test_size: float | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Stratified hold-out split — preserves the class ratio in both halves."""
    return train_test_split(
        X,
        y,
        test_size=test_size if test_size is not None else C.TEST_SIZE,
        random_state=C.RANDOM_STATE,
        stratify=y,
    )
