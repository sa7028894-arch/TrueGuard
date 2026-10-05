import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Qiskit imports
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA

print("[*] Initializing Quantum Anomaly Detection Model...")

# 1. Generate a small synthetic dataset (2 features for fast simulation on laptop)
np.random.seed(42)
X = np.random.randn(100, 2)
# Create a simple non-linear boundary for anomalies (e.g., norm > 1.5)
y = np.array([1 if np.linalg.norm(x) > 1.5 else 0 for x in X])

# Preprocess and scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 2. Configure Quantum Components
num_features = 2  # Matches number of qubits
feature_map = ZFeatureMap(feature_dimension=num_features, reps=1)
ansatz = RealAmplitudes(num_qubits=num_features, reps=1)

# 3. Build the Variational Quantum Classifier (VQC)
print("[*] Training VQC (Variational Quantum Classifier) using Qiskit...")
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=COBYLA(maxiter=30),
)

# Train the model
vqc.fit(X_train, y_train)

# 4. Evaluate
y_pred = vqc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"[+] Training Complete!")
print(f"[+] Quantum Model Test Accuracy: {accuracy * 100:.2f}%")