.PHONY: install sync build lint format clean test

install:
	uv sync

build:
	uv run python scripts/build_dataset.py

eda:
	jupyter lab

train:
	uv run python scripts/train.py

predict:
	uv run python scripts/predict.py

evaluate:
	uv run python scripts/evaluate.py

test:
	uv run pytest

format:
	uv run ruff format .

lint:
	uv run ruff check .

fix:
	uv run ruff check --fix .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

tree:
	tree -I '.venv|__pycache__|.git'

all:
	uv run python scripts/build_dataset.py
	uv run python scripts/train.py
	uv run python scripts/predict.py
	uv run python scripts/evaluate.py