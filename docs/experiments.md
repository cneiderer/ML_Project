# Experiments and Evaluation

## Overview

This project evaluates the ability of machine learning models to predict wind turbine failures within a defined future time horizon using SCADA data.

Experiments are designed to:

* assess predictive performance across multiple time horizons
* evaluate model behavior under severe class imbalance
* understand how early failure signals emerge over time

## Problem Formulation

The task is formulated as a **binary classification problem**:

* **Positive class**: a failure event occurs within a future time window
* **Negative class**: no failure event occurs within that window

Each timestamp is treated as an observation with features derived from historical behavior, while labels are determined by future event occurrence.

## Prediction Windows

Three prediction horizons are evaluated:

* **24 hours**
* **48 hours**
* **72 hours**

These horizons reflect different operational objectives:

* Short horizon → higher precision, immediate intervention
* Long horizon → earlier warning, more planning flexibility

## Labeling Strategy

Labels are generated using an event-based approach:

1. For each timestamp, check if a failure event occurs within the prediction window
2. Assign:

   * `1` → failure occurs within window
   * `0` → no failure occurs within window

### Buffer Zones

To avoid ambiguity and data leakage, observations near failure events are excluded:

* Time periods immediately surrounding failure events are removed
* Ensures that training data reflects true pre-failure conditions

## Dataset Splitting

Data is split using a **time-aware strategy**:

* Training, validation, and test sets are defined chronologically
* No random shuffling is applied
* Prevents leakage from future information

This ensures that models are evaluated on their ability to generalize to future data.

## Cross-Turbine Generalization Considerations

The dataset includes multiple turbines across different wind farms, each with potentially different operating conditions, environments, and equipment characteristics.

Although feature harmonization is applied, the data is not strictly homogeneous across turbines.

The time-based splitting strategy preserves temporal ordering but does not explicitly enforce turbine-level separation. As a result:

* Some turbines may appear only in the training set or only in the test set
* Model performance partially reflects the ability to generalize across turbines and locations

In practice, predictive maintenance systems are often developed at the turbine, site, or equipment-type level to account for these differences.

However, due to limited data availability per turbine and per wind farm, this project adopts a global modeling approach to ensure sufficient training data.

This introduces an additional challenge, as the model must learn patterns that generalize across heterogeneous operating conditions.

## Models Evaluated

The following models are compared:

### Logistic Regression

* Baseline linear model
* Provides interpretability and calibration reference

### Random Forest

* Nonlinear ensemble model
* Captures feature interactions and nonlinear relationships

### XGBoost

* Gradient boosting model
* Strong performance on structured/tabular data
* Handles feature interactions and complex patterns effectively

## Alternative Modeling Approaches

This project frames the problem as a binary classification task using features derived from time-series data.

Alternative approaches, such as direct time-series modeling (e.g., ARIMA, GARCH, LSTM, etc.), could potentially capture temporal dynamics more explicitly. However, these approaches introduce additional complexity and require larger amounts of clean, well-labeled sequential data.

Given the project scope and data constraints, a feature-based classification approach was selected to balance interpretability, implementation complexity, and robustness.

Exploring dedicated time-series models remains a potential area for future work.

## Class Imbalance Handling

Failure events are rare, resulting in severe class imbalance.

To address this:

* Class weights are applied during model training
* Evaluation focuses on precision-recall tradeoffs rather than accuracy

## Evaluation Metrics

The following metrics are used:

* **Precision** → reliability of positive predictions
* **Recall** → ability to detect failure events
* **F1 Score** → balance between precision and recall

### Why Not Accuracy?

Accuracy is not used as a primary metric due to class imbalance.
A model predicting only the majority class would achieve high accuracy but no practical value.

## Threshold Optimization

Model outputs are probabilities that must be converted into binary predictions.

Rather than using a default threshold (0.5), thresholds are tuned to:

* balance precision and recall
* reflect operational tradeoffs
* reduce false positives while maintaining useful recall

Threshold sweep analysis is used to identify appropriate operating points.

### Threshold Optimization vs. Probability Calibration

This project includes **threshold optimization**, not formal probability calibration.

Model-specific decision thresholds were selected using validation-set performance to optimize the precision-recall tradeoff. This step was necessary because default classification thresholds produced poor operational behavior, particularly excessive false positives.

However, threshold optimization does not change the underlying probability estimates themselves. The need for relatively high decision thresholds in tree-based models suggests that predicted probabilities are not well calibrated to true event likelihood.

Formal calibration methods such as Platt scaling or isotonic regression were not applied in the current implementation, but remain a logical area for future work.

## Experimental Workflow

Each experiment follows a consistent pipeline:

1. Generate features and labels for a given prediction window
2. Split data chronologically
3. Train models using training data
4. Tune thresholds using validation data
5. Evaluate final performance on test data

This ensures consistent and fair comparison across models and prediction horizons.

## Key Findings

### 1. Similar Performance Across Horizons

Model performance is broadly consistent across 24h, 48h, and 72h windows.

This suggests that failure signals are distributed over time rather than concentrated near the event.

### 2. Temporal Features Are Critical

Models rely heavily on:

* rolling statistics
* lag features
* volatility measures

These features capture gradual system degradation.

### 3. Threshold Selection Is Essential

Default thresholds lead to poor precision-recall balance.

Careful threshold tuning significantly improves usable model performance.

### 4. Performance Reflects Problem Difficulty

Absolute metric values are modest due to:

* severe class imbalance
* noisy real-world sensor data
* diffuse failure signals

Results should be interpreted as early-warning signal detection rather than precise failure prediction.

## Summary

The experimental framework demonstrates that:

* Predictive maintenance is feasible using SCADA data
* Failure signals can be detected prior to events
* Model performance depends heavily on feature engineering and threshold selection
* Threshold tuning improved operational performance, but formal probability calibration remains future work

The approach emphasizes **practical evaluation under real-world constraints** rather than idealized model performance.
