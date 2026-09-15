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

# THE SINGLE VERIFY ENTRY POINT is `verify`, at the bottom of this section. CI calls it and
# nothing else; a human calls it before pushing. A new check belongs in a target HERE, never
# inline as a step in a workflow -- written twice, the two drift, and CI starts failing on
# things a local run could not have caught.
#
# `lint`/`fmt` cover the WHOLE tree, not just `src`. The architecture guards -- the adoption
# test, the runtime-purity guard, the public-surface pins -- all live under `tests/`, so a
# src-only lint left the files that carry this repo's declarations unchecked.

.PHONY: lint
lint:
	python -m ruff check .

.PHONY: fmt-check
fmt-check:
	python -m ruff format --check .

.PHONY: fmt
fmt:
	python -m ruff check . --fix
	python -m ruff format .

.PHONY: test
test:
	python -m pytest

# The same run, then the coverage report in a browser. Kept apart from `test` so that `test` is
# runnable on a headless box (`open` is not a command on CI, and not on Windows either).
.PHONY: test-open
test-open: test
	open htmlcov/index.html

.PHONY: adoption
adoption:
	python -m pytest tests/architecture/test_optimi_lab_adopts_the_shared_registry.py -v

# The runtime of this library must stay numpy/scipy/scikit-learn. `lab-commons[dev]` brings
# structlog, pint and pydantic and is a DEV extra: CI installs it to run the tests, so the one
# thing CI must not do is install everything and then never check that the RUNTIME table stayed
# pure. This target is that check. It is static (pyproject + the import graph), so installing
# the dev extra cannot make it pass by accident.
.PHONY: purity
purity:
	python -m pytest tests/architecture/test_the_runtime_stays_pure.py -v

.PHONY: verify
verify: lint fmt-check test

# NO `open` BELOW, and that is the whole point of the split. A target that launches a browser on
# every run is an interruption rather than feedback: it steals focus on every GREEN run, and it
# reports nothing back -- a human still has to read the page, so nothing is learned unless somebody
# was already looking. `open` is also not a command on Windows or on CI, so these targets were
# headless-hostile as well. Coverage still lands in `htmlcov/` every run; ASKING to see it is now
# exactly one target, `test-open`.
.PHONY: test-parallel
test-parallel:
	python -m pytest -n auto

.PHONY: test-duration
test-duration:
	python -m pytest --durations=10 --durations-min=1.0

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