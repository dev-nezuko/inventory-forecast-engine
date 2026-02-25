# Run unit tests from repo root (PYTHONPATH set by conftest.py)
test:
	pytest tests/ -v

# Run linter (ruff)
lint:
	ruff check src tests
	ruff format --check src tests

# Format code
format:
	ruff format src tests

.PHONY: test lint format
