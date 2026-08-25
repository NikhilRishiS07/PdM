# Cloud System Anomaly Detection Using Machine Learning

A machine learning-based approach for detecting anomalous behavior in cloud system metrics. The project explores multiple unsupervised anomaly detection techniques and compares their ability to identify different types of abnormal system behavior.

## Overview

The project analyzes system monitoring data containing CPU, memory, disk, load, and process-related metrics. Feature engineering is used to capture both system state and temporal behavior before applying multiple anomaly detection algorithms.

Synthetic anomalies are introduced to enable controlled evaluation of the detection methods.

The project considers three anomaly patterns:

* **Point anomalies** — sudden spikes in system metrics
* **Contextual anomalies** — unusual behavior relative to surrounding system activity
* **Collective anomalies** — abnormal behavior occurring across a sequence of observations

## Approach

1. Load and explore the system monitoring dataset
2. Perform exploratory data analysis and feature engineering
3. Generate labeled synthetic anomalies for evaluation
4. Standardize the feature matrix
5. Apply multiple unsupervised anomaly detection algorithms
6. Compare models using precision, recall, F1-score, and ROC-AUC
7. Analyze ROC and Precision-Recall curves
8. Use SHAP to interpret feature contributions

## Models

* Isolation Forest
* Local Outlier Factor (LOF)
* One-Class SVM
* DBSCAN

## Feature Engineering

The analysis incorporates system and temporal features including:

* CPU utilization
* Memory utilization
* System load averages
* Process activity
* Rolling CPU statistics
* CPU rate of change
* Lagged CPU values
* Time-based features
* CPU-memory interactions
* Load-related features

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* SHAP
* Jupyter Notebook

## Evaluation

The models are evaluated against the injected anomaly labels using precision, recall, F1-score, and ROC-AUC. ROC and Precision-Recall curves are used to compare detection performance across models.

SHAP analysis is also used to examine which features contribute most to Isolation Forest's anomaly predictions.


## Purpose

This project demonstrates the use of unsupervised machine learning, temporal feature engineering, anomaly simulation, model comparison, and explainable AI for cloud system monitoring and anomaly detection.
