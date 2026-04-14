# Repository Structure

## Overview

This repository is organized to support a modular, reproducible machine learning workflow for wind turbine fault detection.

The structure separates core logic, data, experimentation, and outputs to improve clarity, maintainability, and usability.

## Top-Level Structure

```bash
ML_Project/
├── src/                # Source code (Python package)
├── scripts/            # Pipeline entry points
├── notebooks/          # Exploratory analysis and experiment orchestration
├── data/               # Raw and processed data (not version-controlled)
├── artifacts/          # Model outputs and experiment results
├── outputs/            # Generated visualizations
├── documents/          # Project documentation
├── assets/             # Images and branding (e.g., README visuals)
├── config/             # Configuration files
```

## Source Code (`src/wtfd/`)

The core logic of the project is implemented as a Python package.

```bash
src/wtfd/
├── data/               # Data loading, preprocessing, and transformation
├── models/             # Model training, evaluation, and experiment logic
├── utils/              # Logging and shared utilities
```

This modular structure ensures that:

* core functionality is reusable across notebooks and scripts
* logic is not duplicated across experiments
* the pipeline is maintainable and extensible

## Scripts (`scripts/`)

Scripts provide reproducible entry points for running the pipeline:

* `run_preprocessing.py` → generates processed datasets
* `run_modeling.py` → runs model training and evaluation experiments

These scripts allow the full pipeline to be executed without manual interaction.

## Notebooks (`notebooks/`)

Notebooks are used for:

* exploratory data analysis (EDA)
* feature inspection and validation
* experiment analysis and visualization

They serve as an interactive interface for understanding model behavior and validating pipeline outputs.

## Data (`data/`)

```bash
data/
├── raw/                # Original SCADA data (external, not version-controlled)
├── processed/          # Generated datasets (not version-controlled)
```

* Raw data must be downloaded separately
* Processed data is generated via the preprocessing pipeline
* Data files are excluded from version control via `.gitignore`

See `data/README.md` for full details.

## Artifacts (`artifacts/`)

Stores structured outputs from modeling experiments:

* feature importance results
* threshold sweep outputs
* model performance summaries

Artifacts are used for analysis and comparison across experiments.

## Outputs (`outputs/`)

Stores generated visualizations, including:

* threshold curves
* failure timeline plots
* other diagnostic figures

All outputs are generated dynamically and are not version-controlled.

## Documents (`documents/`)

Contains project documentation:

* `architecture.md` → system design and structure
* `experiments.md` → modeling approach and evaluation
* `repository_structure.md` → repository organization (this document)

## Assets (`assets/`)

Stores static resources used in the repository:

* README images
* project visuals and branding

## Configuration (`config/`)

Contains configuration files used throughout the project, such as:

* feature mappings
* path definitions
* pipeline settings

These files enable consistent and reproducible experimentation.

## Notes on Testing

Unit testing is not included in this project due to its exploratory and research-focused nature, but would be recommended for production systems.

## Summary

The repository is structured to:

* separate core logic from experimentation and outputs
* support reproducible pipeline execution
* scale to large datasets through modular design

This organization enables both exploratory analysis and structured experimentation while maintaining clarity and maintainability.
