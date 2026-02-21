#!/usr/bin/env python3
"""Train the 1-qubit QRU forecaster (simulator).

This script is intentionally small and delegates the heavy lifting to risk_pipeline.
It produces:
- artifacts/qru_<run_name>_params.npz  (trained parameters + config)
- artifacts/qru_<run_name>_metrics.json (training curves)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np

try:
    import yaml  # type: ignore
except Exception as e:
    raise SystemExit("pyyaml is required for this script. pip install pyyaml") from e

from risk_pipeline.dataset import load_time_series_csv, make_supervised_windows, temporal_split
from risk_pipeline.scaling import MinMaxPM1Scaler, StandardScaler
from risk_pipeline.qru_jax import QRUConfig, train_qru, predict_qru


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config under configs/")
    ap.add_argument("--csv", required=True, help="CSV time series for ONE site (date, rainfall, river_level, ...)")
    ap.add_argument("--out", default="artifacts", help="Output directory for artifacts")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = int(cfg.get("history", 6))
    horizon = int(cfg.get("horizon", 1))
    target = str(cfg.get("target", "river_level"))
    feature_cols = cfg.get("features", ["rainfall", "river_level"])
    feature_cols = list(feature_cols)

    df = load_time_series_csv(args.csv)
    X, y, t = make_supervised_windows(df, feature_cols=feature_cols, target_col=target, history=history, horizon=horizon)

    # temporal split
    (X_tr, y_tr, _), (X_va, y_va, _), (X_te, y_te, _) = temporal_split(X, y, t, train_frac=0.7, val_frac=0.15)

    # scaling (fit on train only)
    scaler_name = str(cfg.get("scaler", "minmax_pm1")).lower()
    if scaler_name in ("minmax_pm1", "pm1", "minmax"):
        x_scaler = MinMaxPM1Scaler()
        y_scaler = MinMaxPM1Scaler()
    elif scaler_name in ("standard", "zscore"):
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler: {scaler_name}")

    # flatten X to fit scaler feature-wise, then reshape
    X_tr_f = X_tr.reshape(len(X_tr), -1)
    X_va_f = X_va.reshape(len(X_va), -1)
    X_te_f = X_te.reshape(len(X_te), -1)

    X_tr_s = x_scaler.fit_transform(X_tr_f).reshape(X_tr.shape)
    X_va_s = x_scaler.transform(X_va_f).reshape(X_va.shape)
    X_te_s = x_scaler.transform(X_te_f).reshape(X_te.shape)

    y_tr_s = y_scaler.fit_transform(y_tr.reshape(-1, 1)).reshape(-1)
    y_va_s = y_scaler.transform(y_va.reshape(-1, 1)).reshape(-1)
    y_te_s = y_scaler.transform(y_te.reshape(-1, 1)).reshape(-1)

    # QRU config
    qcfg = QRUConfig(
        history=history,
        n_features=len(feature_cols),
        layers=int(cfg.get("layers", 2)),
        learning_rate=float(cfg.get("learning_rate", 1e-2)),
        epochs=int(cfg.get("epochs", 500)),
        patience=int(cfg.get("patience", 50)),
        batch_size=int(cfg.get("batch_size", 64)),
        seed=int(cfg.get("seed", 0)),
    )

    params, metrics = train_qru(X_tr_s, y_tr_s, X_va_s, y_va_s, qcfg)

    # evaluate
    yhat_te_s = predict_qru(X_te_s, params, qcfg)
    yhat_te = y_scaler.inverse_transform(yhat_te_s.reshape(-1, 1)).reshape(-1)

    mse = float(np.mean((yhat_te - y_te) ** 2))

    run_name = str(cfg.get("run_name", "qru_run"))
    np.savez_compressed(
        out_dir / f"qru_{run_name}_params.npz",
        params=params,
        qru_config=asdict(qcfg),
        feature_cols=np.array(feature_cols, dtype=object),
        x_scaler=x_scaler.to_dict(),
        y_scaler=y_scaler.to_dict(),
        test_mse=mse,
    )
    (out_dir / f"qru_{run_name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved: {out_dir / f'qru_{run_name}_params.npz'}")
    print(f"Test MSE (original scale): {mse:.6g}")


if __name__ == "__main__":
    main()
