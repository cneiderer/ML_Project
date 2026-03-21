
---

# 📄 `docs/experiments.md`

```markdown
# Experiments

## Overview

This document defines how modeling experiments are structured and configured.

---

## What is an Experiment?

An experiment defines:

- prediction horizon
- positive class definition
- models to evaluate
- metric to optimize
- split strategy

---

## Example

```python
"pre_24h": {
    "description": "Predict faults within 24 hours",
    "positive_states": ["pre_0_24h", "event_occurring"],
    "models": ["logistic", "rf", "xgboost"],
    "optimize_for": "f1",
    "split_method": "event_chronological"
}