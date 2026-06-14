import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _prepare_base_data():

    # Load dataset
    df = pd.read_csv(
        "../data/atlas-higgs-challenge-2014-v2.csv.gz",
        compression="gzip"
    )

    # Drop unnecessary columns
    df.drop(
        columns=["EventId", "KaggleSet", "KaggleWeight"],
        inplace=True
    )

    # Encode labels
    df["Label"] = df["Label"].map({
        "s": 1,
        "b": 0
    })

    # Features and target
    X = df.drop(columns=["Label", "Weight"])
    y = df["Label"]

    # Replace missing values
    X.replace(-999, np.nan, inplace=True)

    # First split (70% train, 30% temp)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42
    )

    # Second split (15% validation, 15% test)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42
    )

    # Median imputation using TRAIN statistics only
    median_values = X_train.median()
    
    X_train = X_train.fillna(median_values)
    X_val = X_val.fillna(median_values)
    X_test = X_test.fillna(median_values)

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


def load_data_unscaled():

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = _prepare_base_data()

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        X_train.columns.tolist()
    )


def load_data_scaled():

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = _prepare_base_data()

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )

    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        X_train.columns.tolist()
    )