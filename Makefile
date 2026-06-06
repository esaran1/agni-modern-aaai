.PHONY: lint test train evaluate build-grid build-dataset enrich build-labels

lint:
	ruff check src tests scripts experiments

test:
	pytest

build-grid:
	python scripts/build_grid.py configs/experiments/kalimantan_pilot.yaml

build-dataset:
	python scripts/build_dataset.py configs/experiments/kalimantan_pilot.yaml

enrich:
	python scripts/enrich_features.py configs/experiments/kalimantan_pilot.yaml

build-labels:
	python scripts/build_labels.py configs/experiments/kalimantan_pilot.yaml

train:
	python scripts/train.py configs/experiments/kalimantan_pilot.yaml

evaluate:
	python scripts/evaluate.py configs/experiments/kalimantan_pilot.yaml
