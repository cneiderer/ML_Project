# System Architecture

## Overview

This project is structured as a modular, end-to-end machine learning pipeline for wind turbine fault detection. The architecture separates concerns across data processing, feature engineering, modeling, and evaluation while maintaining reproducibility and scalability.

Rather than embedding all logic within notebooks, core functionality is implemented in a reusable Python package (`wtfd`), with notebooks and scripts serving as orchestration layers.

---

## Design Principles

The architecture is guided by the following principles:

### 1. Separation of Concerns

* Data processing, feature engineering, modeling, and evaluation are implemented as distinct components
* Notebooks are used for orchestration and analysis, not core logic
* This improves readability, maintainability, and testability

### 2. Reproducibility

* All datasets and model outputs are generated programmatically
* Preprocessing and modeling pipelines are executed through scripts and notebooks
* No reliance on manually created intermediate artifacts

### 3. Scalability

* The dataset (~15–20 GB raw) requires memory-aware processing
* Data is processed at the turbine level to avoid loading all data into memory
* Columnar storage (Parquet) enables efficient downstream operations

### 4. Consistency

* A unified pipeline ensures consistent preprocessing, feature engineering, and evaluation across experiments
* Prevents data leakage and ensures fair model comparison

## Repository Architecture

The project is organized into the following high-level components:

```bash
ML_Project/
├── src/wtfd/          # Core package (data processing, modeling, utilities)
├── scripts/           # Pipeline entry points
├── notebooks/         # Experiment orchestration and analysis
├── data/              # Raw and processed data (not version-controlled)
├── artifacts/         # Model outputs and experiment results
├── outputs/           # Generated visualizations
├── documents/         # Project documentation
├── config/            # Configuration files
```

## Core Package (`src/wtfd/`)

The `wtfd` package contains the core logic of the pipeline and is organized into subpackages by functional responsibility.

### Data Subpackage (`wtfd.data`)

Handles:

* SCADA data loading
* preprocessing and cleaning
* turbine-level dataset generation

### Modeling Subpackage (`wtfd.models`)

Handles:

* experiment definitions (prediction windows, labeling strategy)
* model training and evaluation
* metrics and threshold optimization
* artifact generation (feature importance, summaries)

This subpackage is composed of multiple modules (e.g., `trainer.py`, `metrics.py`, `experiments.py`) that separate concerns within the modeling pipeline.

### Utilities Subpackage (`wtfd.utils`)

Handles:

* logging
* shared helper functions

> This structure enables separation of concerns within the modeling pipeline, improving maintainability and experiment flexibility.

## Pipeline Architecture

The pipeline follows a structured, sequential flow:

### 1. Data Ingestion

* Load raw SCADA data from multiple wind farms
* Parse semicolon-delimited CSV files
* Organize data by turbine

### 2. Preprocessing

* Handle missing values and inconsistencies
* Harmonize features across wind farms
* Preserve temporal ordering

### 3. Feature Engineering

* Generate rolling statistics, lag features, and rate-of-change metrics
* Create physically meaningful derived features

### 4. Labeling

* Assign binary labels based on future failure windows (24h, 48h, 72h)
* Apply buffer zones to prevent data leakage

### 5. Modeling

* Train multiple classification models (Logistic Regression, Random Forest, XGBoost)
* Apply class weighting to address imbalance

### 6. Evaluation

* Evaluate using precision, recall, and F1-score
* Perform threshold optimization
* Analyze temporal prediction behavior

## Orchestration: Notebooks vs Scripts

### Notebooks

* Used for:

  * experimentation
  * visualization
  * exploratory analysis
* Serve as a transparent interface for understanding pipeline behavior

### Scripts

* `scripts/run_preprocessing.py`
* `scripts/run_modeling.py`

Used for:

* reproducible, end-to-end execution
* consistent pipeline runs without manual intervention

### Why Both?

* Notebooks provide **interpretability and flexibility**
* Scripts provide **reproducibility and automation**

> This combination balances development speed with reproducible, production-style workflow design.

## Data and Artifact Management

### Data

* Raw data is stored under `data/raw/`
* Processed data is generated under `data/processed/`
* Data files are **not version-controlled** and are excluded via `.gitignore`

### Artifacts

* Model outputs are stored under `artifacts/`
* Includes:

  * feature importance
  * threshold sweeps
  * model comparison summaries

### Outputs

* Visualizations are stored under `outputs/`
* Generated dynamically during pipeline execution

## Key Architectural Decisions

### 1. Modular Package Design

Core logic is implemented in `src/wtfd/` rather than notebooks to:

* enable reuse across experiments
* improve maintainability
* support scalable experimentation

### 2. Turbine-Level Processing

Data is processed per turbine to:

* reduce memory usage
* preserve temporal structure
* simplify event alignment

### 3. Parquet-Based Storage

Parquet is used for processed data to:

* improve I/O performance
* reduce storage footprint
* enable efficient columnar operations

### 4. Time-Aware Pipeline Design

* All transformations respect temporal ordering
* Train/test splits are chronological
* Buffer zones prevent leakage near failure events

## Summary

This architecture reflects a transition from a notebook-centric workflow to a modular, reproducible machine learning system.

Key strengths:

* clear separation between logic and orchestration
* scalable data processing for large time-series datasets
* consistent and reproducible experimentation framework

The design supports both exploratory analysis and structured experimentation while remaining aligned with real-world machine learning system practices.
