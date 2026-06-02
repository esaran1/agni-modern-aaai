.PHONY: lint test train evaluate build-grid build-dataset enrich

lint:
	ruff check src tests scripts experiments

test:
	pytest

build-grid:
	python scripts/build_grid.py --config configs/experiments/kalimantan_pilot.yaml

build-dataset:
	python scripts/build_dataset.py --config configs/experiments/kalimantan_pilot.yaml

enrich:
	python scripts/enrich_features.py --config configs/experiments/kalimantan_pilot.yaml

train:
	python scripts/train.py --config configs/experiments/kalimantan_pilot.yaml

evaluate:
	python scripts/evaluate.py --config configs/experiments/kalimantan_pilot.yaml
