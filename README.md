# Agni-v2

Agni-v2 is a research-grade wildfire occurrence and pre-ignition severity forecasting framework for Indonesian peatlands. It is designed for reproducible AAAI-style experiments with explicit configuration, spatiotemporal leakage controls, multi-source feature engineering, and risk ranking via expected loss.

## Scope

The framework supports three aligned tasks:

- `30-day occurrence`: predict whether a patch ignites within the next 30 days.
- `conditional severity`: predict fire severity for rows where a fire occurs and severity is observable.
- `expected risk`: combine both as `R = P(fire) * severity`.

## Repository Layout

Key directories:

- `configs/`: reproducible YAML experiment, data, and model configs.
- `src/agni/`: package source.
- `scripts/`: CLI entry points for build, train, evaluate, and plotting.
- `experiments/`: higher-level experiment runners.
- `tests/`: unit and synthetic integration tests.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

Use Python 3.11 or 3.12 for this repository. The pinned scientific stack is not intended to be installed under Python 3.13.

Example workflow:

```bash
python scripts/build_grid.py configs/experiments/kalimantan_pilot.yaml
python scripts/build_dataset.py configs/experiments/kalimantan_pilot.yaml
python scripts/enrich_features.py configs/experiments/kalimantan_pilot.yaml
python scripts/build_labels.py configs/experiments/kalimantan_pilot.yaml
python scripts/train.py configs/experiments/kalimantan_pilot.yaml
python scripts/evaluate.py configs/experiments/kalimantan_pilot.yaml
```

## Notes

- All experiment parameters live in YAML and deserialize into Pydantic models.
- Feature leakage protection is enforced before every model fit.
- External integrations such as Earth Engine, OSM, and local peat rasters are structured behind adapters so the core pipeline remains testable offline.
- Label builds are task-specific and write separate artifacts such as `labeled_features_occurrence.parquet` and `labeled_features_risk.parquet`.
- `scripts/run_experiment.py` performs the full raw-data-to-labeled-training pipeline in one command.
