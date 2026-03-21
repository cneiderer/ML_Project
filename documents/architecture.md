# Architecture Overview

## Purpose

This document describes the overall architecture of the Wind Turbine Fault Detection (WTFD) project, including:

- data flow across the system
- module responsibilities
- design decisions
- extensibility patterns

The goal is to make the system easy to understand, extend, and reproduce.

---

## High-Level Pipeline

```text
Raw SCADA Data
      ↓
Preprocessing (wtfd/data/preprocessing.py)
      ↓
Processed Event Dataset (Parquet)
      ↓
Modeling Pipeline (wtfd/models/*)
      ↓
Evaluation + Artifacts
      ↓
Outputs (tables, metrics, reports)