Cloud System Anomaly Detection Using Machine Learning

A machine learning-based approach for detecting anomalous behavior in cloud system metrics. The project explores multiple unsupervised anomaly detection techniques and compares their ability to identify different types of abnormal system behavior.

Overview

The project analyzes system monitoring data containing CPU, memory, disk, load, and process-related metrics. Feature engineering is used to capture both system state and temporal behavior before applying multiple anomaly detection algorithms.

Synthetic anomalies are introduced to enable controlled evaluation of the detection methods.

The project considers three anomaly patterns:

Point anomalies — sudden spikes in system metrics
Contextual anomalies — unusual behavior relative to the surrounding system activity
Collective anomalies — gradual changes or drift occurring across a sequence of observations
Approach
Load and explore the system monitoring dataset
Engineer resource, temporal, and interaction-based features
Generate labeled synthetic anomalies for evaluation
Standardize the feature matrix
Train multiple unsupervised anomaly detection models
Compare model performance using precision, recall, F1-score, and ROC-AUC
Analyze ROC and Precision-Recall curves
Use SHAP to investigate feature contributions to Isolation Forest predictions
Models

The following anomaly detection techniques are evaluated:

Isolation Forest
Local Outlier Factor (LOF)
One-Class SVM
DBSCAN
Feature Engineering

The model input includes features such as:

CPU utilization
Memory utilization
System load averages
Process fork rate
Rolling CPU statistics
CPU rate of change
Lagged CPU values
Time-based features
CPU-memory interaction
Load imbalance
Technologies
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
SHAP
Jupyter Notebook
Evaluation

Models are evaluated against the injected anomaly labels using precision, recall, F1-score, and ROC-AUC. ROC and Precision-Recall curves are also generated to compare detection behavior across models.

SHAP analysis is used to provide an interpretable view of which engineered features contribute most to Isolation Forest's anomaly detection decisions.

Project Structure
Cloud-Anomaly-Detection/
├── Cloud_Anomaly_Detection_Research.ipynb
└── README.md
Purpose

This project demonstrates the application of unsupervised machine learning, temporal feature engineering, anomaly simulation, model comparison, and explainable AI to a cloud-system monitoring problem.
