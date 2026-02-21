import json
import os
import pandas as pd
import pennylane as qml
import pennylane.numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService

# --- Configuration des chemins et hyperparamètres ---
DATA_CSV       = "Waihou_River_Te_Aroha_ADCP.csv"
OUTPUT_DIR     = "output_Waihou_River_Te_Aroha_ADCP_qpu"
WINDOW_SIZE    = 6
NB_REUPLOADING = 4

# --- 1. Chargement de la clé API IBM Quantum ---
with open("apikey.json", "r") as f:
    token = json.load(f)["apikey"]

service = QiskitRuntimeService(
    #channel="ibm_quantum_platform",
    channel="ibm_cloud",
    token=token
)

# Sélection du QPU réel le moins occupé
backend = service.least_busy(simulator=False, operational=True)

# --- 2. Définition du device PennyLane pour IBMQ ---
dev = qml.device(
    "qiskit.remote",
    wires=1,
    backend=backend,
    #provider="ibm-q",
    shots=1024,
    optimization_level=1,
)

# --- 3. Définition du circuit QRU (idem entraînement) ---
def qru_circuit(features, thetas):
    for l in range(NB_REUPLOADING):
        for i in range(WINDOW_SIZE):
            r, lvl           = features[i]
            th_r, th_z, th_l = thetas[l, i]
            qml.RX(th_r * r,   wires=0)
            qml.RZ(th_z,       wires=0)
            qml.RY(th_l * lvl, wires=0)
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev)
def qnode(features, thetas):
    return qru_circuit(features, thetas)

# --- 4. Chargement des paramètres entraînés ---
thetas_path = os.path.join(OUTPUT_DIR, "qru_thetas_trained.npy")
thetas = np.load(thetas_path)

# --- 5. Reconstruction des fenêtres et du jeu de test ---
df = pd.read_csv(DATA_CSV, parse_dates=["date"])

X_list, y_list, dates = [], [], []
for i in range(len(df) - WINDOW_SIZE):
    window = df.iloc[i : i + WINDOW_SIZE][["rainfall", "river_level"]].values
    X_list.append(window)
    y_list.append(df.iloc[i + WINDOW_SIZE]["river_level"])
    dates.append(df.iloc[i + WINDOW_SIZE]["date"])

X     = np.array(X_list)        # (n_samples, WINDOW_SIZE, 2)
y     = np.array(y_list)        # (n_samples,)
dates = pd.to_datetime(dates)   # (n_samples,)

n   = X.shape[0]
v80 = int(0.8 * n)

X_test     = X[v80:]
y_test     = y[v80:]
dates_test = dates[v80:]

# --- 6. Fonction de prédiction par indice dans le jeu de test ---
def predict_for_index(idx):
    """
    Prédit le niveau d'eau pour l'indice idx (0-based) dans X_test.
    Affiche la date, la valeur réelle et la prédiction QPU.
    """
    if idx < 0 or idx >= len(X_test):
        raise IndexError(f"Index {idx} hors de la plage du jeu de test [0, {len(X_test)-1}]")
    features = X_test[idx]
    pred     = qnode(features, thetas)
    true_val = float(y_test[idx])
    date     = dates_test[idx]

    print(f"Index        : {idx}")
    print(f"Date         : {date.date()}")
    print(f"Niveau réel  : {true_val:.4f}")
    print(f"Prédiction QPU: {pred:.4f}")
    return pred

# --- 7. Exemple d'utilisation ---
if __name__ == "__main__":
    # Pour prédire l'élément d'indice 5 dans le jeu de test :
    predict_for_index(5)
    # Remplacez 5 par l'indice souhaité (entre 0 et len(X_test)-1).