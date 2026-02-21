#!/usr/bin/env python3
"""Build the coupling matrix C = α C_hydro + β C_geo + γ C_soc.

This implements the definition used in the paper. It does *not* require QRU/QAOA.
Inputs (CSV):
- sites.csv with columns: site_id, lat, lon, social_index (optional)
- hydro_edges.csv (optional) with columns: upstream_site_id, downstream_site_id

Outputs:
- artifacts/C_matrix_<run_name>.npz with C and components.

Notes:
- C_hydro here is a binary adjacency-based coupling. If you have catchment labels instead,
  replace the implementation accordingly.
- C_geo uses kernel 1/(1 + distance_km).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from geopy.distance import geodesic  # type: ignore
except Exception:
    geodesic = None


def haversine_km(lat1, lon1, lat2, lon2):
    import math
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", required=True, help="sites.csv")
    ap.add_argument("--hydro_edges", default=None, help="hydro_edges.csv (optional)")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=0.2)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--run_name", default="paper")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sites = pd.read_csv(args.sites)
    site_ids = sites["site_id"].astype(str).tolist()
    n = len(site_ids)

    # Geo distances
    lat = sites["lat"].to_numpy(float)
    lon = sites["lon"].to_numpy(float)
    D = np.zeros((n, n), float)
    for i in range(n):
        for j in range(i+1, n):
            if geodesic is not None:
                dist = geodesic((lat[i], lon[i]), (lat[j], lon[j])).km
            else:
                dist = haversine_km(lat[i], lon[i], lat[j], lon[j])
            D[i, j] = D[j, i] = dist

    C_geo = 1.0 / (1.0 + D)
    np.fill_diagonal(C_geo, 0.0)
    # normalize to [0,1]
    if C_geo.max() > 0:
        C_geo = C_geo / C_geo.max()

    # Hydro coupling
    C_hydro = np.zeros((n, n), float)
    if args.hydro_edges is not None:
        edges = pd.read_csv(args.hydro_edges)
        idx = {sid: i for i, sid in enumerate(site_ids)}
        for _, row in edges.iterrows():
            u = idx.get(str(row["upstream_site_id"]))
            v = idx.get(str(row["downstream_site_id"]))
            if u is None or v is None:
                continue
            C_hydro[u, v] = 1.0
            C_hydro[v, u] = 1.0
    # already in [0,1]

    # Social coupling
    if "social_index" in sites.columns:
        s = sites["social_index"].to_numpy(float)
    else:
        s = np.zeros(n, float)
    C_soc = (s.reshape(-1, 1) + s.reshape(1, -1)) / 2.0
    np.fill_diagonal(C_soc, 0.0)
    if C_soc.max() > 0:
        C_soc = C_soc / C_soc.max()

    C = args.alpha * C_hydro + args.beta * C_geo + args.gamma * C_soc
    if C.max() > 0:
        C = C / C.max()

    np.savez_compressed(
        out / f"C_matrix_{args.run_name}.npz",
        C=C, C_hydro=C_hydro, C_geo=C_geo, C_soc=C_soc,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        site_ids=np.array(site_ids, dtype=object),
    )
    print("Saved", out / f"C_matrix_{args.run_name}.npz")


if __name__ == "__main__":
    main()
