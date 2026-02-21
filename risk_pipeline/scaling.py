"""Scaling utilities.

This repository uses bounded quantum outputs (e.g., <Z> in [-1,1]) for QRU models.
To train stably and to keep units consistent end-to-end (forecast -> risk -> QUBO/QAOA),
we enforce an explicit scaling contract:

- Fit scaler on TRAIN ONLY (per-site).
- Transform both inputs and targets with that scaler.
- Inference produces predictions in the scaled space.
- inverse_transform brings predictions back to physical units for risk scoring.

Two common choices are supported:
  1) Standardization (mean/std)   : recommended for stability.
  2) Min-max to [-1,1] (pm1)     : naturally matches <Z> bounds.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


Array = np.ndarray


def _as_2d(x: Array) -> Array:
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {x.shape}")
    return x


@dataclass
class StandardScaler:
    """Feature-wise standardization."""

    mean_: Optional[Array] = None
    scale_: Optional[Array] = None
    eps: float = 1e-12

    def fit(self, x: Array) -> "StandardScaler":
        x2 = _as_2d(x)
        mean = x2.mean(axis=0)
        std = x2.std(axis=0)
        std = np.where(std < self.eps, 1.0, std)
        self.mean_ = mean
        self.scale_ = std
        return self

    def transform(self, x: Array) -> Array:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler not fit. Call fit() first.")
        x2 = _as_2d(x)
        z = (x2 - self.mean_) / self.scale_
        return z if np.asarray(x).ndim == 2 else z.reshape(-1)

    def inverse_transform(self, z: Array) -> Array:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler not fit. Call fit() first.")
        z2 = _as_2d(z)
        x = z2 * self.scale_ + self.mean_
        return x if np.asarray(z).ndim == 2 else x.reshape(-1)

    def to_dict(self) -> Dict[str, Any]:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler not fit. Cannot serialize.")
        return {
            "type": "standard",
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "eps": float(self.eps),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "StandardScaler":
        if d.get("type") != "standard":
            raise ValueError(f"Expected scaler type 'standard', got {d.get('type')}")
        s = StandardScaler(eps=float(d.get("eps", 1e-12)))
        s.mean_ = np.asarray(d["mean"], dtype=float)
        s.scale_ = np.asarray(d["scale"], dtype=float)
        return s


@dataclass
class MinMaxPM1Scaler:
    """Feature-wise min-max scaling to [-1, 1]."""

    min_: Optional[Array] = None
    max_: Optional[Array] = None
    eps: float = 1e-12

    def fit(self, x: Array) -> "MinMaxPM1Scaler":
        x2 = _as_2d(x)
        mn = x2.min(axis=0)
        mx = x2.max(axis=0)
        # Avoid zero range
        mx = np.where(np.abs(mx - mn) < self.eps, mn + 1.0, mx)
        self.min_ = mn
        self.max_ = mx
        return self

    def transform(self, x: Array) -> Array:
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Scaler not fit. Call fit() first.")
        x2 = _as_2d(x)
        z01 = (x2 - self.min_) / (self.max_ - self.min_)
        zpm1 = 2.0 * z01 - 1.0
        return zpm1 if np.asarray(x).ndim == 2 else zpm1.reshape(-1)

    def inverse_transform(self, zpm1: Array) -> Array:
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Scaler not fit. Call fit() first.")
        z2 = _as_2d(zpm1)
        z01 = (z2 + 1.0) / 2.0
        x = z01 * (self.max_ - self.min_) + self.min_
        return x if np.asarray(zpm1).ndim == 2 else x.reshape(-1)

    def to_dict(self) -> Dict[str, Any]:
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Scaler not fit. Cannot serialize.")
        return {
            "type": "minmax_pm1",
            "min": self.min_.tolist(),
            "max": self.max_.tolist(),
            "eps": float(self.eps),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MinMaxPM1Scaler":
        if d.get("type") != "minmax_pm1":
            raise ValueError(f"Expected scaler type 'minmax_pm1', got {d.get('type')}")
        s = MinMaxPM1Scaler(eps=float(d.get("eps", 1e-12)))
        s.min_ = np.asarray(d["min"], dtype=float)
        s.max_ = np.asarray(d["max"], dtype=float)
        return s


def make_scaler(kind: str) -> StandardScaler | MinMaxPM1Scaler:
    kind = kind.lower().strip()
    if kind in {"standard", "zscore", "z"}:
        return StandardScaler()
    if kind in {"minmax_pm1", "pm1", "minmax"}:
        return MinMaxPM1Scaler()
    raise ValueError(f"Unknown scaler kind: {kind}")


def fit_transform_inverse_check(
    scaler: StandardScaler | MinMaxPM1Scaler,
    x: Array,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> Tuple[Array, bool]:
    """Utility used by tests: verifies inverse_transform(transform(x)) == x."""
    scaler.fit(x)
    z = scaler.transform(x)
    x_hat = scaler.inverse_transform(z)
    ok = np.allclose(np.asarray(x), np.asarray(x_hat), atol=atol, rtol=rtol)
    return x_hat, bool(ok)
