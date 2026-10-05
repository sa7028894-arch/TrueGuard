import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA

print("[*] Initializing Quantum Anomaly Detection Model...")

np.random.seed(42)
X = np.random.randn(100, 2)
y = np.array([1 if np.linalg.norm(x) > 1.5 else 0 for x in X])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

num_features = 2 
feature_map = ZFeatureMap(feature_dimension=num_features, reps=1)
ansatz = RealAmplitudes(num_qubits=num_features, reps=1)

print("[*] Training VQC (Variational Quantum Classifier) using Qiskit...")
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=COBYLA(maxiter=30),
)

vqc.fit(X_train, y_train)

y_pred = vqc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"[+] Training Complete!")
print(f"[+] Quantum Model Test Accuracy: {accuracy * 100:.2f}%")
