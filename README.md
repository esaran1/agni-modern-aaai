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
pip install -e .[dev]
pytest
```

Example workflow:

```bash
python scripts/build_grid.py --config configs/experiments/kalimantan_pilot.yaml
python scripts/build_dataset.py --config configs/experiments/kalimantan_pilot.yaml
python scripts/enrich_features.py --config configs/experiments/kalimantan_pilot.yaml
python scripts/train.py --config configs/experiments/kalimantan_pilot.yaml
python scripts/evaluate.py --config configs/experiments/kalimantan_pilot.yaml
```

## Notes

- All experiment parameters live in YAML and deserialize into Pydantic models.
- Feature leakage protection is enforced before every model fit.
- External integrations such as Earth Engine, OSM, and local peat rasters are structured behind adapters so the core pipeline remains testable offline.
