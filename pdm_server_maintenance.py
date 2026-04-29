from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

try:
    import requests
except ImportError:
    requests = None

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
except ImportError:
    tf = None


DATASET_URL = (
    "https://raw.githubusercontent.com/westermo/test-system-performance-dataset/main/data/system-1.csv"
)
DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "system-1.csv"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def download_dataset(destination: Path, url: str = DATASET_URL) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        logging.info("Dataset already exists at %s", destination)
        return destination

    logging.info("Downloading dataset from %s", url)
    if requests is not None:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        destination.write_bytes(response.content)
    else:
        import urllib.request

        urllib.request.urlretrieve(url, str(destination))

    logging.info("Dataset downloaded to %s", destination)
    return destination


def load_dataset(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        download_dataset(path)

    df = pd.read_csv(path)
    logging.info("Loaded dataset with %d rows and %d columns", len(df), df.shape[1])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cpu_total"] = df["cpu-user"] + df["cpu-system"] + df["cpu-iowait"]
    df["mem_usage_percent"] = (
        df["sys-mem-total"] - df["sys-mem-available"]
    ) / df["sys-mem-total"]
    df["swap_usage_percent"] = (
        df["sys-mem-swap-total"] - df["sys-mem-swap-free"]
    ) / df["sys-mem-swap-total"]
    df["disk_total_bytes"] = df["disk-bytes-read"] + df["disk-bytes-written"]
    df["disk_total_ops"] = df["disk-io-read"] + df["disk-io-write"]
    df["load_avg"] = (df["load-1m"] + df["load-5m"] + df["load-15m"]) / 3.0
    return df


def drop_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    redundant = [
        "sys-mem-total",
        "sys-mem-swap-total",
        "load-1m",
        "load-5m",
        "load-15m",
        "disk-bytes-read",
        "disk-bytes-written",
        "disk-io-read",
        "disk-io-write",
    ]
    return df.drop(columns=[col for col in redundant if col in df.columns])


def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    start_time = pd.Timestamp("2024-01-01")
    df["datetime"] = pd.to_datetime(df[timestamp_col], unit="s", origin=start_time)
    df["hour_of_day"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    return df


def normalize_features(df: pd.DataFrame, feature_columns: Iterable[str]) -> tuple[pd.DataFrame, MinMaxScaler]:
    df = df.copy()
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns].astype(float))
    return df, scaler


def create_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "server-up",
    window_size: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y, timestamps = [], [], []
    values = df[feature_columns].values
    target = df[target_column].values
    time_values = df["datetime"].values

    for index in range(len(df) - window_size):
        X.append(values[index : index + window_size])
        y.append(target[index + window_size])
        timestamps.append(time_values[index + window_size])

    return np.array(X), np.array(y), np.array(timestamps)


def build_recurrent_model(
    input_shape: tuple[int, int],
    model_type: str = "gru",
    hidden_units: int = 64,
    dropout_rate: float = 0.2,
) -> Sequential:
    if tf is None:
        raise ImportError("TensorFlow is required to build the recurrent model.")

    model = Sequential()
    if model_type.lower() == "gru":
        model.add(GRU(hidden_units, input_shape=input_shape, return_sequences=False))
    else:
        model.add(LSTM(hidden_units, input_shape=input_shape, return_sequences=False))

    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_recurrent_model(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 30,
    batch_size: int = 32,
) -> Sequential:
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )
    return model


def train_isolation_forest(df: pd.DataFrame, feature_columns: Iterable[str], contamination: float = 0.05) -> pd.DataFrame:
    model = IsolationForest(random_state=42, contamination=contamination)
    features = df[list(feature_columns)].astype(float)
    df["iforest_anomaly_score"] = -model.score_samples(features)
    df["iforest_anomaly"] = model.predict(features) == -1
    return df


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    accuracy = accuracy_score(y_true, y_pred)
    logging.info("Test accuracy: %.4f", accuracy)
    logging.info("Confusion matrix:\n%s", confusion_matrix(y_true, y_pred))
    logging.info("Classification report:\n%s", classification_report(y_true, y_pred, zero_division=0))


def plot_time_series(df: pd.DataFrame, timestamp_col: str = "datetime", value_col: str = "cpu_total") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df[timestamp_col], df[value_col], label=value_col)
    ax.set_title(f"{value_col} over time")
    ax.set_xlabel("Time")
    ax.set_ylabel(value_col)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_anomaly_scores(df: pd.DataFrame, timestamp_col: str = "datetime") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    if "iforest_anomaly_score" in df.columns:
        ax.plot(df[timestamp_col], df["iforest_anomaly_score"], label="anomaly score")
    if "server-up" in df.columns:
        ax.plot(df[timestamp_col], df["server-up"], label="server-up", alpha=0.4)
    ax.set_title("Anomaly Score and Server Status")
    ax.set_xlabel("Time")
    ax.legend()
    fig.tight_layout()
    return fig


def build_pipeline(dataset_path: Path, window_size: int = 20, model_type: str = "gru") -> dict:
    df = load_dataset(dataset_path)
    df = engineer_features(df)
    df = add_time_features(df)
    df = drop_redundant_columns(df)

    feature_columns = [
        "load_avg",
        "sys-mem-swap-free",
        "sys-mem-free",
        "sys-mem-cache",
        "sys-mem-buffered",
        "sys-mem-available",
        "sys-fork-rate",
        "sys-interrupt-rate",
        "sys-context-switch-rate",
        "sys-thermal",
        "disk-io-time",
        "disk_total_bytes",
        "disk_total_ops",
        "cpu_total",
        "cpu-iowait",
        "cpu-system",
        "cpu-user",
        "mem_usage_percent",
        "swap_usage_percent",
        "hour_of_day",
        "day_of_week",
    ]
    feature_columns = [col for col in feature_columns if col in df.columns]

    df, scaler = normalize_features(df, feature_columns)

    X, y, timestamps = create_sequences(df, feature_columns, target_column="server-up", window_size=window_size)
    X_train, X_test, y_train, y_test, timestamps_train, timestamps_test = train_test_split(
        X, y, timestamps, test_size=0.2, shuffle=False
    )

    result = {
        "df": df,
        "feature_columns": feature_columns,
        "scaler": scaler,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "timestamps_train": timestamps_train,
        "timestamps_test": timestamps_test,
    }

    if tf is not None:
        model = build_recurrent_model(input_shape=X_train.shape[1:], model_type=model_type)
        result["model"] = model
    else:
        logging.warning("TensorFlow is not installed; skipping recurrent model creation.")

    result["iforest_df"] = train_isolation_forest(df, feature_columns)
    return result


def run_training(dataset_path: Path, window_size: int, model_type: str, epochs: int) -> None:
    pipeline = build_pipeline(dataset_path, window_size=window_size, model_type=model_type)
    df = pipeline["df"]
    feature_columns = pipeline["feature_columns"]

    if "model" in pipeline:
        model = pipeline["model"]
        train_recurrent_model(model, pipeline["X_train"], pipeline["y_train"], pipeline["X_test"], pipeline["y_test"], epochs=epochs)
        predictions = (model.predict(pipeline["X_test"]) > 0.5).astype(int).reshape(-1)
        evaluate_classification(pipeline["y_test"], predictions)
    else:
        logging.warning("No recurrent model available; computing anomaly detection only.")

    anomaly_fig = plot_anomaly_scores(df)
    anomaly_fig.savefig(Path(__file__).resolve().parent / "anomaly_timeline.png")
    logging.info("Anomaly visualization saved to anomaly_timeline.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predictive maintenance pipeline for server monitoring")
    parser.add_argument("--dataset", default=DEFAULT_DATA_PATH, type=Path, help="Path to the server metrics CSV dataset")
    parser.add_argument("--window-size", default=20, type=int, choices=[10, 20], help="Sequence window size")
    parser.add_argument("--model-type", default="gru", choices=["gru", "lstm"], help="Recurrent model type")
    parser.add_argument("--epochs", default=30, type=int, help="Training epochs for the recurrent model")
    parser.add_argument("--download", action="store_true", help="Download the dataset if not already present")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    if args.download:
        download_dataset(args.dataset)
    run_training(args.dataset, args.window_size, args.model_type, args.epochs)


if __name__ == "__main__":
    main()
