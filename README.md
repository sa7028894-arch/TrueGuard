\# TrueGuard: Quantum-Enhanced Anomaly Detection


A hybrid quantum-classical machine learning project developed for \*\*Qiskit Fall Fest 2026 @ MPSTME\*\*. TrueGuard demonstrates how parameterised quantum circuits can be leveraged for classification and anomaly detection tasks.


\## 🚀 Overview

Traditional machine learning algorithms often struggle with complex, non-linear decision boundaries in high-dimensional feature spaces. TrueGuard implements a \*\*Variational Quantum Classifier (VQC)\*\* using Qiskit, mapping classical data into a higher-dimensional Hilbert space via quantum feature maps (`ZFeatureMap`) and optimizing a parameterised quantum circuit (`RealAmplitudes`) via classical optimizers (`COBYLA`).



\## 🛠️ Tech Stack \& Dependencies

\* \*\*Python\*\*

\* \*\*Qiskit\*\* (Quantum computing framework)

\* \*\*Qiskit Machine Learning\*\* (Hybrid quantum-classical algorithms)

\* \*\*Qiskit Algorithms\*\* (Optimizers)

\* \*\*Scikit-Learn\*\* (Data preprocessing \& performance metrics)



\## 📂 Repository Structure

```text

TrueGuard/

│

├── src/

│   └── model.py       # Core Qiskit VQC training and evaluation script

│

└── README.md          # Project documentation


⚙️ Installation \& Quickstart

Clone the repository:

Bash

git clone \[https://github.com/sa7028894-arch/TrueGuard.git](https://github.com/sa7028894-arch/TrueGuard.git)

cd TrueGuard

Install required dependencies:

Bash

pip install qiskit qiskit-machine-learning qiskit-algorithms scikit-learn numpy

Run the quantum model:

Bash

python src/model.py

📊 Results \& Performance

Model Architecture: Variational Quantum Classifier (VQC)

Feature Dimension: 2 Qubits

Optimizer: COBYLA

Baseline Performance: Validated via proof-of-concept synthetic anomaly generation achieving 75.00% test accuracy.

💡 Future Scope \& Enhancements

Scaling feature dimensions using Principal Component Analysis (PCA) for real-world cybersecurity intrusion logs or financial datasets.

Integrating realistic hardware noise models and error mitigation techniques to analyze performance degradation on simulated quantum processors.

