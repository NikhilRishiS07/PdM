# Predictive Maintenance for Server Metrics

This workspace contains a dataset and a production-ready Python pipeline for predictive maintenance using server monitoring time series.

## Files

- `system-1.csv` — downloaded server metrics dataset from the public `westermo/test-system-performance-dataset` repository.
- `pdm_server_maintenance.py` — modular pipeline for feature engineering, normalization, time-series sequence creation, LSTM/GRU modeling, Isolation Forest anomaly detection, and visualization.
- `requirements.txt` — required Python packages.

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the pipeline:

```bash
python pdm_server_maintenance.py --dataset system-1.csv --window-size 20 --model-type gru --epochs 30
```

3. If the dataset is missing, add `--download` to fetch it automatically:

```bash
python pdm_server_maintenance.py --download
```

## Output

- `anomaly_timeline.png` — generated plot of anomaly score and server status over time.
