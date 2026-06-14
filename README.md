# Explainable Classical and Quantum Machine Learning for Higgs Boson Classification

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

## 📌 Overview
This project applies Classical Machine Learning, Deep Learning, and Explainable AI (XAI) to classify Higgs Boson signal events from background noise. Using the ATLAS Higgs Boson dataset, multiple architectures are evaluated to maximize classification accuracy. Furthermore, SHAP and LIME are utilized to extract the most critical physical variables, establishing a streamlined feature set for ongoing research into **Quantum Machine Learning (QML)** applications in high-energy physics.

---

## 📊 Dataset & Preprocessing

**Source:** [ATLAS Higgs Boson Machine Learning Challenge](https://www.kaggle.com/c/higgs-boson/data)
* **Samples:** ~818,000 | **Features:** 30 | **Class Imbalance:** 65.8% Background / 34.2% Signal
* **Data Cleaning:** Removed non-informative tracking variables (`EventId`, `KaggleSet`, `KaggleWeight`).
* **Imputation:** Missing sensor values (`-999`) were imputed using strict median imputation derived *only* from the training set to prevent data leakage.
* **Scaling:** Unscaled pipelines for tree-based ensembles; Standardized pipelines for neural networks.

---

## ⚙️ Modeling Pipeline

We trained and evaluated a diverse suite of tabular models to establish a robust baseline:
* **Classical Ensembles:** Random Forest, XGBoost, LightGBM
* **Deep Learning:** Multi-Layer Perceptron (MLP), TabNet, TabTransformer
* **Baseline:** Logistic Regression

### 🏆 Final Model Performance (Test Set)
**LightGBM** and **XGBoost** achieved state-of-the-art performance. The tuned **XGBoost** model was selected for the final XAI pipeline due to its balance of high ROC-AUC and interpretability.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | **0.8431** | 0.7906 | **0.7355** | **0.7621** | **0.9119** |
| **XGBoost (Selected)** | 0.8427 | 0.7905 | 0.7343 | 0.7614 | 0.9105 |
| TabNet | 0.8420 | **0.8086** | 0.7041 | 0.7528 | 0.9108 |
| TabTransformer | 0.8418 | 0.7895 | 0.7320 | 0.7597 | 0.9104 |
| Logistic Regression | 0.7492 | 0.6674 | 0.5299 | 0.5907 | 0.8129 |

*(Note: The XGBoost model demonstrated excellent generalization, with a Validation ROC-AUC of 0.9151 and Test ROC-AUC of 0.9116).*

---

## 🧠 Explainability (XAI)
To break the "black-box" nature of the final model, **SHAP** (global) and **LIME** (local) were utilized. Both frameworks independently verified the same physical mass reconstruction and kinematic variables as the primary drivers of prediction:

**Top 5 Predictive Features:**
1. `DER_mass_MMC`
2. `DER_mass_transverse_met_lep`
3. `DER_mass_vis`
4. `PRI_tau_pt`
5. `DER_deltar_tau_lep`

---

## ⚛️ Ongoing Research: Quantum ML (QML)
The classical ML pipeline serves as the foundation for the next phase of this project. Using the highly informative feature subsets identified by SHAP, current active research focuses on:
* **Quantum Support Vector Machines (QSVM):** Evaluating quantum kernels on dimensionally-reduced feature spaces using `Qiskit`.
* **Variational Quantum Classifiers (VQC):** Benchmarking parameterized quantum circuits against classical baselines.

---

## 💻 Tech Stack
* **Core:** `Python`, `NumPy`, `Pandas`
* **Modeling:** `Scikit-Learn`, `XGBoost`, `LightGBM`, `PyTorch`, `TabNet`
* **Explainability:** `SHAP`, `LIME`
* **Quantum:** `Qiskit`

---

## 📂 Repository Structure

```text
Higgs_Boson_Project/
├── data/               # Ignored: Raw and processed datasets
├── models/             # Ignored: Saved model artifacts (.pkl, .pt)
├── notebooks/          # Jupyter notebooks for EDA, tuning, and XAI
├── results/            # model outputs(graphs, plots, XAI graphs and tables)
├── src/                # Modular Python scripts (data_loader.py, etc.)
├── requirements.txt    # Reproducible Python environment
├── setup.py            # Local package build configuration
└── README.md           # Project documentation

## Author: Aather Nabi
## License: This project is intended for academic and research purposes.