"""Risk scoring utilities.

The paper maps physical river level predictions to a bounded flood-risk score via
historical thresholds per day (or per season) such as p5 and p95:

risk(h) = 0                      if h <= p5
        = (h - p5) / (p95 - p5)  if p5 < h < p95
        = 1                      if h >= p95

This module implements that transformation, ensuring *test-only* evaluation
(no leakage from training data into risk maps used for decision-making).

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskBandConfig:
    low_col: str = "p5"
    high_col: str = "p95"
    clip: bool = True


def band_risk(
    h: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    clip: bool = True,
) -> np.ndarray:
    """Compute band-based risk in [0,1]."""
    h = np.asarray(h, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    denom = np.where(np.abs(high - low) < 1e-12, 1.0, (high - low))
    r = (h - low) / denom
    if clip:
        r = np.clip(r, 0.0, 1.0)
    return r


def compute_daily_risk_for_site(
    preds_physical: pd.Series,
    hist_thresholds: pd.DataFrame,
    cfg: RiskBandConfig = RiskBandConfig(),
) -> pd.Series:
    """Compute daily risk for a single site.

    Parameters
    ----------
    preds_physical : pd.Series
        Predicted river levels in physical units, indexed by date.
    hist_thresholds : pd.DataFrame
        DataFrame indexed by date with columns cfg.low_col, cfg.high_col.
        These thresholds must be derived from historical data (not using the
        prediction period itself).

    Returns
    -------
    pd.Series
        Risk score per date in [0,1].
    """
    if not isinstance(preds_physical.index, pd.DatetimeIndex):
        preds_physical = preds_physical.copy()
        preds_physical.index = pd.to_datetime(preds_physical.index)

    ht = hist_thresholds.copy()
    if not isinstance(ht.index, pd.DatetimeIndex):
        ht.index = pd.to_datetime(ht.index)

    # Align on dates (inner join)
    joined = pd.concat([preds_physical.rename("pred"), ht[[cfg.low_col, cfg.high_col]]], axis=1, join="inner")
    if joined.empty:
        raise ValueError("No overlapping dates between predictions and thresholds")

    r = band_risk(joined["pred"].values, joined[cfg.low_col].values, joined[cfg.high_col].values, clip=cfg.clip)
    return pd.Series(r, index=joined.index, name="risk")


def aggregate_site_risk(
    daily_risk: pd.Series,
    start: Optional[str] = None,
    end: Optional[str] = None,
    agg: str = "sum",
) -> float:
    """Aggregate daily risk across a period (e.g., hydrological year).

    agg:
      - 'sum' : paper's default for annual cumulative exposure
      - 'mean': average daily risk
      - 'max' : worst-day risk
    """
    s = daily_risk
    if start is not None:
        s = s[s.index >= pd.to_datetime(start)]
    if end is not None:
        s = s[s.index <= pd.to_datetime(end)]

    if s.empty:
        raise ValueError("No data in the selected aggregation interval")

    agg = agg.lower().strip()
    if agg == "sum":
        return float(s.sum())
    if agg == "mean":
        return float(s.mean())
    if agg == "max":
        return float(s.max())
    raise ValueError(f"Unknown agg: {agg}")


def risk_scores_test_only(
    preds_df: pd.DataFrame,
    thresholds_by_site: Dict[str, pd.DataFrame],
    site_col: str = "site",
    date_col: str = "date",
    pred_col: str = "y_pred",
    split_col: str = "split",
    test_value: str = "test",
    cfg: RiskBandConfig = RiskBandConfig(),
    agg: str = "sum",
    agg_start: Optional[str] = None,
    agg_end: Optional[str] = None,
) -> Dict[str, float]:
    """Compute per-site risk scores using *test split only*.

    preds_df is expected to contain per-date predictions for all sites with a split label.
    This function enforces that only rows with split==test_value contribute.

    Returns dict site -> aggregated risk.
    """
    required = {site_col, date_col, pred_col, split_col}
    missing = required - set(preds_df.columns)
    if missing:
        raise ValueError(f"preds_df missing columns: {sorted(missing)}")

    df = preds_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[df[split_col] == test_value]

    scores: Dict[str, float] = {}
    for site, g in df.groupby(site_col):
        if site not in thresholds_by_site:
            raise ValueError(f"Missing thresholds for site '{site}'")
        s = pd.Series(g[pred_col].values, index=g[date_col].values).sort_index()
        daily = compute_daily_risk_for_site(s, thresholds_by_site[site], cfg=cfg)
        scores[site] = aggregate_site_risk(daily, start=agg_start, end=agg_end, agg=agg)

    return scores
