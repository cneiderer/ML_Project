# Data Overview

This project uses publicly available wind turbine SCADA data for early fault detection.

**Source:**
Kasimov, A. (2024)
https://zenodo.org/records/10958775

The dataset consists of multivariate, time-stamped measurements collected from multiple turbines across several wind farms.

## Dataset Structure

The raw dataset is organized by wind farm:

```bash
data/raw/zenodo_windfarm_data/
├── Wind Farm A/
├── Wind Farm B/
└── Wind Farm C/
```

Each wind farm contains:

* `datasets/` → individual turbine SCADA time-series (CSV files)
* `event_info.csv` → failure/anomaly event metadata
* `feature_description.csv` → sensor definitions and descriptions

## Data Size and Storage

The raw SCADA dataset is large (~15–20 GB when extracted), requiring careful data handling and memory-aware processing.

Processed datasets are smaller (~2–3 GB) but still substantial depending on feature engineering configuration.

Users should ensure sufficient disk space and memory before running the full preprocessing pipeline, particularly when generating feature-rich datasets.

## Key Characteristics

* Multivariate time-series data
* Multiple turbines per wind farm
* Environmental, mechanical, electrical, and thermal features
* Labeled failure/anomaly events
* Regular time intervals per turbine

## Important Data Challenges

### 1. Cross-Farm Inconsistencies

Sensor names, availability, and distributions vary across wind farms.
Equivalent physical measurements are not guaranteed to share the same naming or scale.

### 2. Missing Data

Some sensors exhibit substantial missingness or intermittent gaps.
Feature selection and preprocessing decisions account for data quality variability.

### 3. Class Imbalance

Failure events are rare relative to normal operation, creating a highly imbalanced classification problem.

### 4. Diffuse Failure Signals

Failure-related patterns emerge gradually over time rather than at a single point, requiring temporal feature engineering.

## Data Processing Strategy

To address these challenges, the following approach is used:

### 1. Turbine-Level Processing

Each turbine is processed independently to:

* preserve temporal ordering
* avoid memory constraints
* maintain clean event alignment

### 2. Feature Harmonization

Features are aligned across wind farms using:

* mapping definitions (`config/feature_map.yaml`)
* selection of physically meaningful variables
* derived features (more robust than raw signals)

### 3. Temporal Feature Engineering

Features are generated to capture system behavior over time:

* rolling statistics (mean, standard deviation)
* lag features
* rate-of-change metrics
* volatility measures

### 4. Event-Based Labeling

Each timestamp is labeled based on whether a failure occurs within a future window:

* 24-hour horizon
* 48-hour horizon
* 72-hour horizon

### 5. Buffer Zones

Data near failure events is excluded to:

* prevent label ambiguity
* reduce data leakage
* ensure meaningful pre-failure signals

## Processed Data

Processed datasets are stored in:

```bash
data/processed/
```

> Note: This directory is populated during preprocessing. Data files are not version-controlled and are excluded via `.gitignore`.

### Structure

* `*_event_*.parquet` → event-specific datasets
* `master_dataset.parquet` → combined dataset for modeling

Each dataset:

* is time-ordered
* includes engineered features
* contains labels for prediction windows

## Why Parquet?

Parquet format is used instead of CSV because:

* faster read/write performance
* reduced storage size
* efficient columnar access
* better scalability for large datasets

## Reproducibility

All processed datasets are generated programmatically using:

```bash
python scripts/run_preprocessing.py
```

> Raw data is required to reproduce processed datasets.
> Processed data files are not version-controlled and are excluded via `.gitignore`.

This design choice:

- prevents large file bloat in the repository
- ensures the project remains lightweight and maintainable
- reinforces reproducibility by requiring data to be generated from source

Users must download the raw dataset and run the preprocessing pipeline to reproduce all intermediate and final datasets.

## Notes

* Raw data is not modified in-place
* All transformations are applied through the preprocessing pipeline
* Temporal ordering is preserved throughout the pipeline
