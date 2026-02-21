import os
import json
import numpy as np
import pandas as pd
from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    Batch,
    EstimatorV2,
    EstimatorOptions,
)
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.compiler import transpile

# --- Config ---
DATA_CSV    = "Waihou_River_Te_Aroha_ADCP.csv"
OUTPUT_DIR  = "output_Waihou_River_Te_Aroha_ADCP_qpu_batch_opt"
THETA_FILE  = os.path.join(OUTPUT_DIR, "qru_thetas_trained.npy")
PRED_CSV    = os.path.join(OUTPUT_DIR, "predictions_last30_test_qpu_batch.csv")
APIKEY_FILE = "apikey.json"

WINDOW_SIZE    = 6
NB_REUPLOADING = 4
SHOTS          = 1024

# 1. IBMQ Service & backend
with open(APIKEY_FILE, "r") as f:
    token = json.load(f)["apikey"]
service = QiskitRuntimeService(channel="ibm_cloud", token=token)
backend = service.least_busy(simulator=False, operational=True)

# 2. Load trained thetas
if not os.path.exists(THETA_FILE):
    raise FileNotFoundError(f"Thetas file not found: {THETA_FILE}")
thetas = np.load(THETA_FILE)

# 3. Read & split dataset
df = pd.read_csv(
    DATA_CSV,
    usecols=["date", "rainfall", "river_level"],
    parse_dates=["date"],
)
n_total = len(df) - WINDOW_SIZE
n_train = int(0.8 * n_total)
n_test  = n_total - n_train
n_last  = int(0.3 * n_test)

start = WINDOW_SIZE + n_train + (n_test - n_last)
end   = WINDOW_SIZE + n_total  # exclu

# 4. Build raw circuits + collect metadata
raw_circuits = []
dates       = []
true_vals   = []

for idx in range(start, end):
    row = df.iloc[idx]
    dates.append(row.date.date().isoformat())
    true_vals.append(row.river_level)

    window = df.iloc[idx - WINDOW_SIZE : idx][["rainfall", "river_level"]].values
    qc = QuantumCircuit(1)
    for l in range(NB_REUPLOADING):
        for i, (r, lvl) in enumerate(window):
            th_r, th_z, th_l = thetas[l, i]
            qc.rx(th_r * r,   0)
            qc.rz(th_z,       0)
            qc.ry(th_l * lvl, 0)
    raw_circuits.append(qc)

# 5. Transpile *avant* soumission (optimization_level=0 pour aller vite)
transpiled = transpile(
    raw_circuits,
    backend=backend,
    optimization_level=0
)

# 6. Prepare PUBs (circuit, observable)
z_op = SparsePauliOp("Z")
pubs = [(qc, z_op.apply_layout(qc.layout)) for qc in transpiled]

# 7. Configure EstimatorOptions V2 (sans « transpilation » !)
opts = EstimatorOptions()
opts.default_shots      = SHOTS
opts.optimization_level = 0   # pour le *runtime* primitive, pas la transpilation
opts.resilience_level   = 0   # désactive la mitigation

# 8. Submit en batch
with Batch(backend=backend) as batch:
    estimator = EstimatorV2(mode=batch, options=opts)
    job       = estimator.run(pubs)
    results   = job.result()

# 9. Write CSV
os.makedirs(OUTPUT_DIR, exist_ok=True)
records = [
    {
        "date":       dates[i],
        "river_true": float(true_vals[i]),
        "river_pred": float(results[i].data.evs[0]),
    }
    for i in range(len(dates))
]
pd.DataFrame.from_records(records).to_csv(PRED_CSV, index=False)
