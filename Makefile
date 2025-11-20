.DEFAULT_GOAL := all

install:
	uv pip install -U pip wheel
	uv pip install -e "."

.PHONY: install-dev
install-dev:
	uv pip install -U pip wheel
	uv pip install -e ".[dev]"
	python -m pre-commit install
	python -m pre-commit autoupdate

.PHONY: lint
lint:
	python -m ruff check src
	python -m ruff format --check src

.PHONY: fmt
fmt:
	python -m ruff check src --fix
	python -m ruff format src

.PHONY: test
test:
	python -m pytest
	open htmlcov/index.html

.PHONY: test-parallel
test-parallel:
	python -m pytest -n auto
	open htmlcov/index.html

.PHONY: test-duration
test-duration:
	python -m pytest --durations=10 --durations-min=1.0
	open htmlcov/index.html

.PHONY: all
all: fmt test

.PHONY: clean
clean:
	rm -rf `find . -name __pycache__`
	rm -rf dist/
	rm -rf build/
	rm -rf docs/
	rm -rf output/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf *.egg-info
	rm -f .coverage
	rm -f .coverage.*
	rm -f result.json