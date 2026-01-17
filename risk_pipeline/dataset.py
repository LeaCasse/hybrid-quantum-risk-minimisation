"""Dataset utilities (windowing + leak-free temporal split).

The paper/notebook use daily time series per site with a fixed history window H (e.g., 6).
This module provides:

- load_time_series_csv: read a per-site CSV with a date column.
- make_supervised_windows: convert a multivariate time series into (X,y) windows.
- temporal_split: split windows by timestamp boundaries (no leakage).

Key rule (ACM-grade rigor):
  * Do NOT random-shuffle time series.
  * Fit scalers on TRAIN ONLY.
  * Ensure validation/test windows do not overlap the training period.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowedDataset:
    X: np.ndarray  # shape (N, H, F)
    y: np.ndarray  # shape (N,)
    t: np.ndarray  # shape (N,) timestamps corresponding to the target time (t+1)
    feature_cols: List[str]
    target_col: str


def load_time_series_csv(
    path: str,
    date_col: str = "date",
    sort: bool = True,
) -> pd.DataFrame:
    """Load a site CSV.

    Expected columns include a date column plus features and target.
    The function returns a DataFrame indexed by pandas.DatetimeIndex.
    """
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Missing date column '{date_col}' in {path}")
    df[date_col] = pd.to_datetime(df[date_col])
    if sort:
        df = df.sort_values(date_col)
    df = df.set_index(date_col)
    # Drop duplicate timestamps deterministically (keep last)
    df = df[~df.index.duplicated(keep="last")]
    return df


def make_supervised_windows(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str,
    history: int,
    horizon: int = 1,
    dropna: bool = True,
) -> WindowedDataset:
    """Create windowed samples.

    For each time t, we take features from [t-history+1 ... t] and predict y at t+horizon.

    Shapes:
      X: (N, history, F)
      y: (N,)
      t: (N,) target timestamps
    """
    feature_cols = list(feature_cols)
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found")
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"feature_col '{c}' not found")

    data = df[feature_cols + [target_col]].copy()
    if dropna:
        data = data.dropna(axis=0, how="any")

    values = data.values
    ts = data.index.to_numpy()

    H = int(history)
    h = int(horizon)
    if H <= 0 or h <= 0:
        raise ValueError("history and horizon must be positive")

    # last usable index i is such that i-horizon is within range and i-history+1 >= 0
    # We'll build target at i (timestamp ts[i]) and window ending at i-h
    # Simpler: choose window_end = k (ending index for history), target_index = k + h
    X_list = []
    y_list = []
    t_list = []

    for k in range(H - 1, len(values) - h):
        window = values[k - H + 1 : k + 1, : len(feature_cols)]
        target = values[k + h, len(feature_cols)]
        X_list.append(window)
        y_list.append(target)
        t_list.append(ts[k + h])

    X = np.asarray(X_list, dtype=float)
    y = np.asarray(y_list, dtype=float)
    t = np.asarray(t_list)

    return WindowedDataset(X=X, y=y, t=t, feature_cols=feature_cols, target_col=target_col)


def temporal_split(
    ds: WindowedDataset,
    train_end: str,
    val_end: Optional[str] = None,
) -> Tuple[WindowedDataset, WindowedDataset, WindowedDataset]:
    """Split dataset by time boundaries.

    Parameters
    ----------
    train_end : str
        Last timestamp (inclusive) for training targets.
    val_end : str, optional
        Last timestamp (inclusive) for validation targets.
        If None, validation set is empty and everything after train_end is test.

    Returns
    -------
    train, val, test : WindowedDataset

    Notes
    -----
    The timestamps ds.t correspond to the prediction target time.
    Splitting by ds.t avoids lookahead leakage.
    """
    t = pd.to_datetime(ds.t)
    train_end_ts = pd.to_datetime(train_end)
    val_end_ts = pd.to_datetime(val_end) if val_end is not None else None

    train_mask = t <= train_end_ts
    if val_end_ts is None:
        val_mask = np.zeros_like(train_mask, dtype=bool)
        test_mask = ~train_mask
    else:
        val_mask = (t > train_end_ts) & (t <= val_end_ts)
        test_mask = t > val_end_ts

    def _subset(mask: np.ndarray) -> WindowedDataset:
        return WindowedDataset(
            X=ds.X[mask],
            y=ds.y[mask],
            t=ds.t[mask],
            feature_cols=ds.feature_cols,
            target_col=ds.target_col,
        )

    return _subset(train_mask), _subset(val_mask), _subset(test_mask)
