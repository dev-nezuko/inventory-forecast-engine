# Run from the repo root; pytest finds src/ through pyproject.toml
test:
	python -m pytest

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .

.PHONY: test lint format
