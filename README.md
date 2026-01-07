
# Higgs Boson Event Classification

This project aims to classify Higgs boson production events (signal) from background events using machine learning techniques. It utilizes the **HIGGS dataset**, a classic dataset in high-energy physics, to train and evaluate models like Random Forest and XGBoost.

## 📂 Project Structure

```
ML_Project/
├── data/               # Contains the dataset (HIGGS.csv.gz)
├── models/             # Saved trained models and objects (pkl files)
├── notebooks/          # Jupyter notebooks for EDA and modeling
├── results/            # Directory for analysis results and plots
├── venv/               # Python virtual environment
└── README.md           # Project documentation
```

## 📊 Dataset

The dataset used is the **HIGGS dataset**, which contains 11 million Monte Carlo simulations of kinematic features produced by the ATLAS experiments.
- **Size:** 11 million instances (project currently uses a subset or full set depending on memory).
- **Features:** 28 kinematic features (21 low-level features like momentum, pseudorapidity, etc., and 7 high-level derived features).
- **Target:** Trace (1 for signal, 0 for background).

## 🛠️ Installation

1. **Clone the repository** (if applicable) or navigate to the project directory.

2. **Set up the environment:**
   It is recommended to use the provided virtual environment `venv` or create a new one.
   ```bash
   # Activate existing venv (Windows)
   .\venv\Scripts\activate
   ```

3. **Install dependencies:**
   Ensure you have the required Python packages installed.
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```

## 🚀 Usage

### Data Exploration
Open the `notebooks/01_data_exploration.ipynb` notebook to see the Exploratory Data Analysis (EDA). This notebook covers:
- Loading the data
- Viewing feature distributions
- Correlation analysis
- Signal vs. Background comparison

### Models
The project includes trained models saved in the `models/` directory:
- `higgs_random_forest.pkl`: A trained Random Forest Classifier.
- `higgs_xgboost.pkl`: A trained XGBoost Classifier.
- `standard_scaler.pkl`: The scaler used to normalize the data before training.

To use these models in a script:
```python
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('models/higgs_xgboost.pkl')
scaler = joblib.load('models/standard_scaler.pkl')

# Example prediction (assuming 'X_new' is your dataframe of features)
# X_scaled = scaler.transform(X_new)
# predictions = model.predict(X_scaled)
```

## 📈 Results
The models are evaluated based on AUROC (Area Under the Receiver Operating Characteristic curve) and Accuracy.
*(Detailed metrics and plots can be generated using the notebooks)*.

## 🔗 References
- [UCI Machine Learning Repository - HIGGS](https://archive.ics.uci.edu/ml/datasets/HIGGS)
- [CERN Open Data Portal](http://opendata.cern.ch/)
