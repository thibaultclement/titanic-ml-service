.PHONY: install sync build train predict evaluate all test format lint fix clean tree eda

install:
	uv sync

sync:
	uv sync

build:
	uv run python scripts/build_dataset.py

train:
	uv run python scripts/train.py

predict:
	uv run python scripts/predict.py

evaluate:
	uv run python scripts/evaluate.py

all: build train predict evaluate

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
	tree -I '.venv|__pycache__|.git|*.egg-info'

eda:
	jupyter lab